"""Bounded Refiner adaptation study; only Train-main/Train-dev are accessible."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import numpy as np
import torch
from torch import nn

from .data import load_events
from .h2.constants import SENSOR_HEIGHT, SENSOR_WIDTH
from .h2.losses import refiner_loss
from .h2.numpy_model import H2Export, load_export
from .h2.representation import build_temporal_patch_features, patch_scores_to_heatmaps
from .h2.torch_model import (
    QuantizedMLP, fake_int8_weight, fake_uint8, load_exported_models, prototype_logits,
)
from .metrics import event_metrics, localization_metrics
from .s0p import role_file_paths
from .s0r import build_s0r_candidates, s0r_loss
from .snn import RefinerSNN, load_proposal_snn_export, load_refiner_snn_export


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "results" / "runs"
INDEX = ROOT / "data" / "index.json"
WEIGHTS = {
    "h2": ("H2_D1_seed42.npz", "21019377617e429a599497a0a6b0f541d2c05bf7bc8fee83005deae72bb578b2"),
    "proposal": ("S0P_T4_POS15_KD50_S42.npz", "fb5fd635ee8f4b9660f98d6c71299de9e40b9d14fa83e51b2306853b2db86952"),
    "refiner": ("S0R_T4_POS10_KD50_S42.npz", "35223d0dae3dbfdeac758aec41c6904349937072b8f7e5f35fa1628e170dc0b7"),
}
PROTOCOL = {
    "version": 1, "proposal_threshold": 0.51, "time_steps": 4,
    "beta": 0.875, "threshold": 0.25, "readout": "mean_membrane",
    "epochs": 20, "batch_size": 4096, "learning_rate": 1e-4,
    "positive_fraction": 0.10, "distillation_weight": 0.50,
    "spike_rate_weight": 0.01, "optimizer": "AdamW", "weight_decay": 0.01,
    "gradient_clip": 5.0, "seeds": [42, 43, 44], "max_training_runs": 10,
    "iou_adaptation_gain": 0.005, "full_snn_min_iou": 0.75,
    "iou_tolerance": 0.002, "pd_tolerance": 0.005, "fa_ratio_limit": 1.10,
    "threshold_grid": [0.01, 0.99, 0.01],
    "selection_ties": "higher IoU, higher F1, earlier epoch, lower threshold",
}
ARMS = {"A": "h2", "B": "snn", "C": "snn", "D": "snn"}


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return _digest(handle)


def _digest(handle: Any) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def within(path: Path, parent: Path) -> bool:
    return path.resolve().is_relative_to(parent.resolve())


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def frozen_inputs() -> dict[str, Any]:
    hashes = {"index": sha256(INDEX)}
    if hashes["index"] != "7968011347d9d6291a94e183b4268a84549301003523bca7b4b185a5ca36a397":
        raise ValueError("frozen data index mismatch")
    for key, (name, expected) in WEIGHTS.items():
        actual = sha256(ROOT / "software" / "weights" / name)
        if actual != expected:
            raise ValueError(f"frozen {key} weight mismatch")
        hashes[key] = actual
    return hashes


def weight_path(key: str) -> Path:
    return ROOT / "software" / "weights" / WEIGHTS[key][0]


def study_paths(role: str, limit: int = 0) -> list[Path]:
    paths = role_file_paths(INDEX, role)
    if any(not within(path, ROOT / "data" / "EV-UAV-dataset") for path in paths):
        raise ValueError("study data must come from the official dataset directory")
    return paths[:limit] if limit else paths


@dataclass
class SequenceCache:
    name: str
    events: np.ndarray
    candidates: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    teacher: np.ndarray

    def validate(self) -> None:
        n = len(self.candidates)
        if self.candidates.dtype != np.int64 or self.candidates.shape != (n,):
            raise ValueError("invalid candidate indices")
        if n and (self.candidates[0] < 0 or self.candidates[-1] >= len(self.events)
                  or np.any(np.diff(self.candidates) <= 0)):
            raise ValueError("candidate indices must be unique, ordered and in range")
        if self.features.dtype != np.uint8 or self.features.shape != (n, 32):
            raise ValueError("invalid Refiner features")
        if self.labels.dtype != np.float32 or self.labels.shape != (n,) or not np.array_equal(self.labels, self.events["label"][self.candidates]):
            raise ValueError("candidate labels are misaligned")
        if self.teacher.dtype != np.float32 or self.teacher.shape != (n,) or not np.isfinite(self.teacher).all() or np.any((self.teacher < 0) | (self.teacher > 1)):
            raise ValueError("invalid teacher probabilities")
        for array in (self.events, self.candidates, self.features, self.labels, self.teacher):
            array.flags.writeable = False

    @property
    def nbytes(self) -> int:
        return sum(x.nbytes for x in (self.events, self.candidates, self.features, self.labels, self.teacher))


def load_cache(path: Path, identity: dict[str, Any]) -> SequenceCache:
    with np.load(path, allow_pickle=False) as archive:
        if json.loads(str(archive["identity"].item())) != identity:
            raise ValueError(f"stale or mismatched cache: {path.name}")
        result = SequenceCache(identity["file"], *(archive[key] for key in ("events", "candidates", "features", "labels", "teacher")))
    result.validate()
    return result


def save_cache(path: Path, identity: dict[str, Any], record: SequenceCache) -> None:
    record.validate()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, identity=np.asarray(json.dumps(identity, sort_keys=True)),
                            **{key: getattr(record, key) for key in ("events", "candidates", "features", "labels", "teacher")})
    temporary.replace(path)


def prepare_sequences(
    role: str, source: str, directory: Path, h2: H2Export, proposal: nn.Module,
    device: torch.device, hashes: dict[str, Any], stats: dict[str, Any],
    limit: int = 0, memory_mb: int = 1024,
) -> list[SequenceCache]:
    if source not in ("h2", "snn"):
        raise ValueError("unknown candidate source")
    records = []
    memory = 0
    for path in study_paths(role, limit):
        started = time.perf_counter()
        stat = path.stat()
        identity = {"version": PROTOCOL["version"], "role": role, "source": source,
                    "file": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
                    "frozen_inputs": hashes, "threshold": 0.51 if source == "snn" else h2.proposal_threshold}
        cache_path = directory / f"{role}-{source}-{path.stem}.npz"
        if cache_path.exists():
            record = load_cache(cache_path, identity)
            stats["hits"] += 1
        else:
            events = load_events(path, require_labels=True)
            heatmaps = None
            if source == "snn":
                temporal = build_temporal_patch_features(events, list(range(80)), 4)
                tensor = torch.from_numpy(temporal).permute(1, 0, 2, 3).reshape(4, -1, 128)
                parts = []
                with torch.no_grad():
                    for start in range(0, tensor.shape[1], 4096):
                        parts.append(proposal(tensor[:, start:start + 4096].to(device)).logits.cpu().numpy())
                logits = np.concatenate(parts).astype(np.float64)
                scores = (1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))).astype(np.float32)
                heatmaps = patch_scores_to_heatmaps(scores.reshape(80, 336, 16))
            candidates, features, labels, teacher = build_s0r_candidates(
                events, h2, heatmaps=heatmaps, proposal_threshold=0.51 if source == "snn" else None)
            record = SequenceCache(path.name, events, candidates, features, labels, teacher)
            save_cache(cache_path, identity, record)
            stats["misses"] += 1
        elapsed = time.perf_counter() - started
        stats["sequence_seconds"].append({"role": role, "source": source, "file": path.name, "seconds": elapsed})
        memory += record.nbytes
        # Flattening temporarily retains both sequence arrays and contiguous copies.
        if memory * 2 > memory_mb * 1024 ** 2:
            raise MemoryError("sequence cache exceeds bounded memory budget; increase --memory-mb")
        records.append(record)
        print(f"[{role}/{source}] {len(records)} {path.name}: {len(record.features)} candidates, {elapsed:.2f}s", flush=True)
    stats["peak_sequence_bytes"] = max(stats["peak_sequence_bytes"], memory)
    return records


@dataclass
class TrainingData:
    features: np.ndarray
    labels: np.ndarray
    teacher: np.ndarray


def flatten(records: list[SequenceCache]) -> TrainingData:
    data = TrainingData(*(np.concatenate([getattr(r, key) for r in records]) for key in ("features", "labels", "teacher")))
    if not len(data.labels):
        raise ValueError("training role has no candidates")
    for array in (data.features, data.labels, data.teacher):
        array.flags.writeable = False
    return data


def balanced_indices(labels: np.ndarray, count: int, seed: int, epoch: int) -> np.ndarray:
    if not len(labels) or count <= 0:
        raise ValueError("sampling requires positive source and target counts")
    generator = torch.Generator().manual_seed(seed + 100000 * epoch)
    positive = torch.from_numpy(np.flatnonzero(labels > 0.5))
    negative = torch.from_numpy(np.flatnonzero(labels <= 0.5))
    if not len(positive) or not len(negative):
        raise ValueError("balanced study training requires both classes")
    npos = round(count * 0.10)
    pos = positive[torch.randint(len(positive), (npos,), generator=generator)]
    nneg = count - npos
    neg = negative[torch.randperm(len(negative), generator=generator)[:nneg]] if nneg <= len(negative) else negative[torch.randint(len(negative), (nneg,), generator=generator)]
    indices = torch.cat((pos, neg))[torch.randperm(count, generator=generator)].numpy()
    indices.flags.writeable = False
    return indices


def prune_refiner(teacher: QuantizedMLP) -> tuple[QuantizedMLP, dict[str, list[int]]]:
    weights: list[torch.Tensor] = []
    for layer in teacher.layers:
        if not isinstance(layer, nn.Linear):
            raise TypeError("Refiner pruning requires Linear layers")
        weights.append(layer.weight.detach().cpu())
    selected = []
    for weight, count in zip(weights[:2], (32, 16)):
        scores = weight.abs().sum(dim=1).numpy()
        chosen = np.lexsort((np.arange(len(scores)), -scores))[:count]
        selected.append(np.sort(chosen))
    first, second = selected
    model = QuantizedMLP((32, 32, 16, 4), quantized=True)
    with torch.no_grad():
        for destination, value in zip(model.layers, (weights[0][first], weights[1][second][:, first], weights[2][:, second])):
            if not isinstance(destination, nn.Linear):
                raise TypeError("Refiner pruning requires Linear layers")
            destination.weight.copy_(value)
    return model, {"hidden1": first.tolist(), "hidden2": second.tolist()}


def make_model(arm: str) -> tuple[nn.Module, dict[str, Any]]:
    if arm in ("A", "B", "frozen_snn"):
        model, config, metadata = load_refiner_snn_export(weight_path("refiner"))
        for key in ("time_steps", "beta", "threshold", "readout"):
            if config[key] != PROTOCOL[key]:
                raise ValueError(f"frozen Refiner configuration mismatch: {key}")
        info = {"initialization": WEIGHTS["refiner"][0], "dimensions": list(model.dimensions),
                "frozen_threshold": float(metadata["event_threshold"]), "kind": "snn"}
    else:
        _, model = load_exported_models(weight_path("h2"))
        info = {"initialization": WEIGHTS["h2"][0], "kind": "ann", "quantized": True,
                "frozen_threshold": float(load_export(weight_path("h2")).event_threshold)}
        if arm == "D":
            model, info["pruned_channels"] = prune_refiner(model)
        elif arm not in ("C", "frozen_ann"):
            raise ValueError("unknown study arm")
        info["dimensions"] = list(model.dimensions)
    info["weights"] = sum(p.numel() for p in model.parameters())
    return model, info


def raw_forward(model: nn.Module, features: torch.Tensor, cached: bool = False) -> torch.Tensor:
    if isinstance(model, RefinerSNN):
        return model(features.unsqueeze(0).expand(4, -1, -1), cache_static_input=cached).raw_logits
    return model(features)


def infer(model: nn.Module, features: np.ndarray, device: torch.device, cached: bool = False,
          batch_size: int = 4096) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size].copy()).to(device)
            raw = raw_forward(model, batch, cached)
            parts.append(torch.softmax(prototype_logits(raw), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0, dtype=np.float32)


def full_probabilities(records: list[SequenceCache], candidate_probability: np.ndarray) -> np.ndarray:
    if sum(len(r.candidates) for r in records) != len(candidate_probability):
        raise ValueError("candidate probability count mismatch")
    parts = []
    offset = 0
    for record in records:
        probability = np.zeros(len(record.events), dtype=np.float32)
        count = len(record.candidates)
        probability[record.candidates] = candidate_probability[offset:offset + count]
        offset += count
        parts.append(probability)
    return np.concatenate(parts)


def scan_thresholds(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    if labels.shape != probabilities.shape or not np.isfinite(probabilities).all():
        raise ValueError("invalid threshold scan inputs")
    # Match float32 >= threshold semantics in event_metrics without 99 full-array scans.
    thresholds = np.arange(0.01, 0.99 + 0.5 * 0.01, 0.01)
    pos = np.sort(probabilities[labels.astype(bool)])
    neg = np.sort(probabilities[~labels.astype(bool)])
    tp = len(pos) - np.searchsorted(pos, thresholds.astype(probabilities.dtype), side="left")
    fp = len(neg) - np.searchsorted(neg, thresholds.astype(probabilities.dtype), side="left")
    iou = np.divide(tp, len(pos) + fp, out=np.ones(len(tp), dtype=float), where=(len(pos) + fp) != 0)
    denom = tp + fp + len(pos)
    f1 = np.divide(2 * tp, denom, out=np.ones(len(tp), dtype=float), where=denom != 0)
    best = max(range(99), key=lambda i: (iou[i], f1[i]))
    threshold = float(thresholds[best])
    return {"best_threshold": threshold, "best_event": event_metrics(labels, probabilities, threshold),
            "grid_min": 0.01, "grid_max": 0.99, "grid_step": 0.01, "evaluated_thresholds": 99}


def evaluate_probabilities(records: list[SequenceCache], candidates: np.ndarray,
                           fixed_threshold: float, with_localization: bool) -> dict[str, Any]:
    probabilities = full_probabilities(records, candidates)
    labels = np.concatenate([r.events["label"] for r in records])
    scan = scan_thresholds(labels, probabilities)
    result = {"scan": scan, "fixed_event": event_metrics(labels, probabilities, fixed_threshold)}
    if with_localization:
        per_sequence = []
        offset = 0
        totals = {key: 0.0 for key in ("correct_targets", "targets", "false_components", "frame_denominator")}
        fixed_totals = totals.copy()
        for r in records:
            values = probabilities[offset:offset + len(r.events)]
            offset += len(r.events)
            e = r.events
            def locate(threshold: float) -> dict[str, float]:
                return localization_metrics(e["t"], e["x"], e["y"], e["label"], e["name"], values, threshold)
            loc = locate(scan["best_threshold"])
            fixed_loc = locate(fixed_threshold)
            for key in totals:
                totals[key] += loc[key]
                fixed_totals[key] += fixed_loc[key]
            per_sequence.append({"file": r.name, "event": event_metrics(e["label"], values, scan["best_threshold"]), "localization": loc})
        for name, data in (("localization", totals), ("fixed_localization", fixed_totals)):
            data["pd"] = data["correct_targets"] / data["targets"] if data["targets"] else 1.0
            denominator = data["frame_denominator"] * SENSOR_WIDTH * SENSOR_HEIGHT
            data["fa"] = data["false_components"] / denominator if denominator else 0.0
            result[name] = data
        result["per_sequence"] = per_sequence
    return result


def train_epoch(model: nn.Module, data: TrainingData, indices: np.ndarray,
                optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, Any]:
    model.train()
    started = time.perf_counter()
    wait_seconds = 0.0
    total_loss = 0.0
    for start in range(0, len(indices), 4096):
        transfer_started = time.perf_counter()
        selected = indices[start:start + 4096]
        features = torch.from_numpy(data.features[selected]).to(device)
        labels = torch.from_numpy(data.labels[selected]).to(device)
        teacher = torch.from_numpy(data.teacher[selected]).to(device)
        synchronize(device)
        wait_seconds += time.perf_counter() - transfer_started
        optimizer.zero_grad(set_to_none=True)
        if isinstance(model, RefinerSNN):
            output = model(features.unsqueeze(0).expand(4, -1, -1))
            losses = s0r_loss(output.raw_logits, labels, teacher, output.spike_rates,
                              distillation_weight=0.50, spike_rate_weight=0.01)
        else:
            losses = refiner_loss(model(features), labels, teacher, torch.ones_like(labels), distillation_weight=0.50)
        losses["total"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(losses["total"].detach().cpu()) * len(selected)
    synchronize(device)
    return {"loss": total_loss / len(indices), "samples": len(indices),
            "updates": (len(indices) + 4095) // 4096, "input_wait_s": wait_seconds,
            "seconds": time.perf_counter() - started}


def profile_cost(model: nn.Module, features: np.ndarray, device: torch.device,
                 cached: bool = False) -> dict[str, Any]:
    model.eval()
    counts = {"scheduled_multiplications": 0, "nonzero_multiplication_proxy": 0,
              "scheduled_accumulations": 0, "nonzero_accumulation_proxy": 0,
              "spike_accumulation_proxy": 0}
    handles = []
    snn = isinstance(model, RefinerSNN)
    if snn:
        def hook(index: int):
            def count(layer: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                value = inputs[0]
                fanout = output.shape[-1]
                dense = value.numel() * fanout
                sparse = int(torch.count_nonzero(value)) * fanout
                counts["scheduled_accumulations"] += dense
                counts["nonzero_accumulation_proxy"] += sparse
                if index == 0:
                    counts["scheduled_multiplications"] += dense
                    counts["nonzero_multiplication_proxy"] += sparse
                else:
                    counts["spike_accumulation_proxy"] += sparse
            return count
        for index, layer in enumerate(model.layers):
            handles.append(layer.register_forward_hook(hook(index)))
    try:
        with torch.no_grad():
            for start in range(0, len(features), 4096):
                batch = torch.from_numpy(features[start:start + 4096].copy()).to(device)
                if snn:
                    raw_forward(model, batch, cached)
                else:
                    if not isinstance(model, QuantizedMLP):
                        raise TypeError("study profiling requires RefinerSNN or QuantizedMLP")
                    value = batch.float() / 255.0
                    if model.quantized:
                        value = fake_uint8(value)
                    for index, layer in enumerate(model.layers):
                        if not isinstance(layer, nn.Linear):
                            raise TypeError("ANN profiling requires Linear layers")
                        dense = value.numel() * layer.out_features
                        sparse = int(torch.count_nonzero(value)) * layer.out_features
                        counts["scheduled_multiplications"] += dense
                        counts["nonzero_multiplication_proxy"] += sparse
                        counts["scheduled_accumulations"] += dense
                        counts["nonzero_accumulation_proxy"] += sparse
                        weight = fake_int8_weight(layer.weight) if model.quantized else layer.weight
                        value = nn.functional.linear(value, weight)
                        if index + 1 < len(model.layers):
                            value = nn.functional.hardtanh(value, 0.0, 1.0)
                            if model.quantized:
                                value = fake_uint8(value)
    finally:
        for handle in handles:
            handle.remove()
    inference_updates = len(features) * 4 * (64 + 32 + 4) if snn else 0
    diagnostic_updates = len(features) * 4 * 4 if snn else 0
    # Warmup and timings exclude activity instrumentation, include batch copies/transfers.
    infer(model, features[:4096], device, cached)
    timings = []
    for _ in range(3):
        synchronize(device)
        started = time.perf_counter()
        infer(model, features, device, cached)
        synchronize(device)
        timings.append(time.perf_counter() - started)
    return {**counts, "inference_membrane_updates": inference_updates,
            "diagnostic_membrane_updates": diagnostic_updates,
            "membrane_updates": inference_updates + diagnostic_updates,
            "membrane_state_bytes_per_event": (64 + 32 + 4 + 4) * 4 if snn else 0,
            "cached_current_bytes_per_event": 64 * 4 if snn and cached else 0,
            "parameter_bytes_in_backend": sum(p.numel() * p.element_size() for p in model.parameters()),
            "candidate_events": len(features), "device": str(device), "backend": "PyTorch float32/QAT",
            "cached_first_current": cached, "inference_seconds": timings,
            "median_inference_s": float(np.median(timings)),
            "scope": "Refiner only, same SNN Proposal candidates; no preprocessing, FPGA or energy claims"}


def complete_evaluation(model: nn.Module, info: dict[str, Any], records: list[SequenceCache],
                        features: np.ndarray, device: torch.device,
                        probability: np.ndarray | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    uncached = infer(model, features, device)
    probabilities = uncached if probability is None else probability
    evaluation = evaluate_probabilities(records, probabilities, info["frozen_threshold"], True)
    cost: dict[str, Any] = {"uncached": profile_cost(model, features, device)}
    if isinstance(model, RefinerSNN):
        cached = infer(model, features, device, True)
        if not np.array_equal(uncached, cached):
            raise AssertionError("cached Refiner inference changed probabilities")
        cost["cached"] = profile_cost(model, features, device, True)
        cost["cache_probability_bit_exact"] = True
    if probability is not None:
        cost["official_numpy_vs_torch_max_probability_error"] = float(np.max(np.abs(probabilities - uncached))) if len(uncached) else 0.0
    return {"info": info, "selected": evaluation, "cost": cost,
            "evaluation_seconds": time.perf_counter() - started}


def localization_guard(candidate: dict[str, Any], reference: dict[str, Any]) -> bool:
    c, r = candidate["selected"]["localization"], reference["selected"]["localization"]
    fa_limit = r["fa"] * 1.10 + 1e-15 if r["fa"] else 0.0
    return c["pd"] >= r["pd"] - 0.005 - 1e-12 and c["fa"] <= fa_limit


def iou(record: dict[str, Any]) -> float:
    return float(record["selected"]["scan"]["best_event"]["iou"])


def dominated(snn: dict[str, Any], ann: dict[str, Any]) -> bool:
    snn_loc, ann_loc = snn["selected"]["localization"], ann["selected"]["localization"]
    if iou(ann) < iou(snn) or ann_loc["pd"] < snn_loc["pd"] or ann_loc["fa"] > snn_loc["fa"]:
        return False
    s = snn["cost"].get("cached", snn["cost"]["uncached"])
    a = ann["cost"]["uncached"]
    keys = ("nonzero_multiplication_proxy", "nonzero_accumulation_proxy", "membrane_updates",
            "parameter_bytes_in_backend", "median_inference_s")
    return all(a[k] <= s[k] for k in keys) and any(a[k] < s[k] for k in keys)


def decide(arms: dict[str, Any], baselines: dict[str, Any]) -> dict[str, Any]:
    a, b, c, d = (arms[key] for key in "ABCD")
    frozen = baselines["frozen_ann"]
    mixed = max((c, d, frozen), key=iou)
    adaptation_gain = iou(b) - iou(a)
    adaptation = adaptation_gain >= 0.005 - 1e-12 and localization_guard(b, a)
    full = (iou(b) >= 0.75 - 1e-12 and iou(b) >= iou(mixed) - 0.002 - 1e-12
            and localization_guard(b, mixed))
    small = all(iou(d) >= iou(r) - 0.002 - 1e-12 and localization_guard(d, r) for r in (c, frozen))
    is_dominated = dominated(b, d)
    return {"adaptation_iou_gain": adaptation_gain, "adaptation_pass": adaptation,
            "full_ann_adaptation_iou_gain": iou(c) - iou(frozen),
            "full_ann_adaptation_localization_pass": localization_guard(c, frozen),
            "full_snn_accuracy_pass": full, "compact_ann_pass": small,
            "full_snn_dominated_by_compact_ann": is_dominated,
            "replicate_snn": (adaptation or full) and not is_dominated,
            "replicate_compact_ann": small,
            "hardware_mainline_changed": False}


def replication_arms(decision: dict[str, Any]) -> list[str]:
    if decision["replicate_snn"]:
        # Recheck B against all mixed alternatives. A remains the seed-42 causal control.
        return ["B", "C", "D"]
    if decision["replicate_compact_ann"]:
        return ["C", "D"]
    return []


def final_decision(first: dict[str, Any], repetitions: dict[str, Any], baselines: dict[str, Any]) -> dict[str, Any]:
    snn = bool(first["full_snn_accuracy_pass"] and not first["full_snn_dominated_by_compact_ann"])
    compact = bool(first["compact_ann_pass"])
    details = {}
    for seed, arms in repetitions.items():
        frozen = baselines["frozen_ann"]
        c, d = arms["C"], arms["D"]
        compact_seed = all(iou(d) >= iou(r) - 0.002 - 1e-12 and localization_guard(d, r) for r in (c, frozen))
        compact = compact and compact_seed
        snn_seed = False
        if "B" in arms:
            b = arms["B"]
            mixed = max((c, d, frozen), key=iou)
            snn_seed = (iou(b) >= 0.75 - 1e-12 and iou(b) >= iou(mixed) - 0.002 - 1e-12
                        and localization_guard(b, mixed) and not dominated(b, d))
        snn = snn and snn_seed
        details[seed] = {"full_snn_pass": snn_seed, "compact_ann_pass": compact_seed}
    if len(repetitions) != 2:
        snn = compact = False
    return {"full_snn_ready_for_quantization_research": snn,
            "compact_ann_ready_for_implementation_evaluation": compact,
            "seed42": first, "replications": details,
            "adaptation_gain_replication": "A only ran at seed 42; causal gain is not claimed as replicated",
            "hardware_mainline_changed": False,
            "next": "evaluate_qualified_candidates" if snn or compact else "retain_frozen_mainline_stop_this_search"}


def train_arm(arm: str, seed: int, epochs: int, data: TrainingData,
              samples: int, dev: list[SequenceCache], dev_features: np.ndarray,
              directory: Path, device: torch.device, shared_samples: dict[int, np.ndarray],
              hashes: dict[str, Any], deadline: float) -> dict[str, Any]:
    run_directory = directory / f"{arm}-S{seed}"
    run_directory.mkdir(exist_ok=False)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model, info = make_model(arm)
    model.to(device)
    info.update({"arm": arm, "seed": seed, "training_candidate_source": ARMS[arm],
                 "evaluation_candidate_source": "snn", "samples_per_epoch": samples})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    history: list[dict[str, Any]] = []
    best_key = (-1.0, -1.0)
    best_epoch = 0
    best_state = None
    run = {"schema_version": 1, "status": "running", "experiment_id": f"REFINER-{arm}-S{seed}",
           "stage": "refiner_adaptation", "data_role": ["train_main", "train_dev"],
           "primary_metrics": {}, "decision": "pending", "conclusion": "Training in progress",
           "protocol": PROTOCOL, "info": info, "frozen_inputs": hashes, "history": history}
    write_json(run_directory / "history.json", run)
    try:
        for epoch in range(1, epochs + 1):
            if time.perf_counter() >= deadline:
                raise TimeoutError("study wall-time limit reached between epochs")
            indices = shared_samples[epoch] if arm != "A" else balanced_indices(data.labels, samples, seed, epoch)
            metrics = train_epoch(model, data, indices, optimizer, device)
            evaluation_started = time.perf_counter()
            candidate_probability = infer(model, dev_features, device)
            evaluation = evaluate_probabilities(dev, candidate_probability, info["frozen_threshold"], False)
            record = {"epoch": epoch, "train": metrics, "evaluation": evaluation,
                      "selection_seconds": time.perf_counter() - evaluation_started}
            history.append(record)
            event = evaluation["scan"]["best_event"]
            key = (event["iou"], event["f1"])
            if key > best_key:
                best_key, best_epoch = key, epoch
                best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                torch.save({"model": best_state, "info": info, "epoch": epoch,
                            "protocol": PROTOCOL, "frozen_inputs": hashes,
                            "selection": evaluation}, run_directory / "selected.pt")
            write_json(run_directory / "history.json", run)
            print(f"[{arm}/S{seed}] epoch {epoch}/{epochs}: IoU={event['iou']:.6f}, train={metrics['seconds']:.2f}s", flush=True)
        if best_state is None:
            raise AssertionError("no selected checkpoint")
        model.load_state_dict(best_state)
        result = complete_evaluation(model, info, dev, dev_features, device)
        result["selected"]["epoch"] = best_epoch
        result["checkpoint"] = str((run_directory / "selected.pt").relative_to(ROOT))
        result["training_seconds"] = sum(h["train"]["seconds"] for h in history)
        result["selection_seconds"] = sum(h["selection_seconds"] for h in history)
        result["input_wait_s"] = sum(h["train"]["input_wait_s"] for h in history)
        run.update({"status": "ok", "primary_metrics": result["selected"], "decision": "selected_on_train_dev",
                    "conclusion": "Research checkpoint only; frozen deployment weights unchanged", "result": result})
        write_json(run_directory / "history.json", run)
        return result
    except BaseException as error:
        run.update({"status": "failed", "decision": "incomplete", "conclusion": str(error)})
        write_json(run_directory / "history.json", run)
        raise


def summary_template(experiment_id: str, mode: str, hashes: dict[str, Any]) -> dict[str, Any]:
    return {"schema_version": 1, "status": "running", "experiment_id": experiment_id,
            "stage": "refiner_adaptation", "data_role": ["train_main", "train_dev"],
            "primary_metrics": {}, "decision": "pending", "conclusion": "Study in progress",
            "mode": mode, "complete_roles": mode == "formal", "protocol": PROTOCOL,
            "frozen_inputs": hashes, "validation_used": False, "test_used": False,
            "hardware_mainline_changed": False, "baselines": {}, "runs": {},
            "cost_definitions": {
                "multiplications": "Algorithmic nonbinary products; SNN binary synapses counted as accumulations. PyTorch still executes dense GEMMs, including binary inputs.",
                "nonzero_proxy": "Input nonzero count times fanout; hypothetical zero skipping, not implemented CPU/GPU scheduling or energy.",
                "membrane_updates": "Inference and diagnostic output membranes reported separately; leak/reset arithmetic is included here, not in synaptic products.",
                "state_bytes": "Logical float32 membrane/current state per event, excluding backend workspaces, activation histories and parameter tensors.",
                "latency": "Three uninstrumented same-device Refiner timings after warmup; batch copies/transfers included, common Proposal and feature construction excluded.",
            }}


def data_diagnostics(h2_train: TrainingData, snn_train: TrainingData,
                     dev: list[SequenceCache]) -> dict[str, Any]:
    labels = np.concatenate([r.events["label"] for r in dev])
    retained = sum(int(r.labels.sum()) for r in dev)
    return {"h2_train_candidates": len(h2_train.labels), "snn_train_candidates": len(snn_train.labels),
            "h2_train_positive_fraction": float(h2_train.labels.mean()),
            "snn_train_positive_fraction": float(snn_train.labels.mean()),
            "dev_sequences": len(dev), "dev_events": len(labels), "dev_target_events": int(labels.sum()),
            "dev_candidate_events": sum(len(r.candidates) for r in dev),
            "dev_proposal_recall": retained / int(labels.sum()) if labels.sum() else 1.0,
            "dev_proposal_missed_targets": int(labels.sum()) - retained,
            "context_mean_h2_training": h2_train.features[:, -4:].mean(axis=0).tolist(),
            "context_mean_snn_training": snn_train.features[:, -4:].mean(axis=0).tolist(),
            "interpretation": "Training context means also reflect different candidate populations; not a causal attribution"}


def execute(arguments: argparse.Namespace) -> dict[str, Any]:
    directory = arguments.run_directory.resolve()
    if not within(directory, RUN_ROOT) or directory == RUN_ROOT.resolve():
        raise ValueError("run directory must be a child of results/runs")
    if directory.exists():
        raise FileExistsError("use a new run directory; existing studies are never overwritten")
    output = arguments.output.resolve() if arguments.output else directory / "summary.json"
    if output.exists():
        raise FileExistsError("summary already exists")
    if not within(output, ROOT / "results") or (arguments.mode != "formal" and not within(output, RUN_ROOT)):
        raise ValueError("formal summaries belong in results; pilot/smoke summaries in results/runs")
    cache_directory = arguments.cache_directory.resolve() if arguments.cache_directory else directory / "cache"
    if not within(cache_directory, RUN_ROOT) or cache_directory == RUN_ROOT.resolve():
        raise ValueError("cache directory must be a child of results/runs")
    hashes = frozen_inputs()
    device = choose_device(arguments.device)
    directory.mkdir(parents=True)
    summary = summary_template(arguments.experiment_id, arguments.mode, hashes)
    summary["environment"] = {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
                              "device": str(device), "cuda": getattr(torch, "version").cuda,
                              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                              "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())}
    summary["invocation"] = {key: str(value) if isinstance(value, Path) else value for key, value in vars(arguments).items()}
    stats = {"hits": 0, "misses": 0, "peak_sequence_bytes": 0, "sequence_seconds": []}
    stats["directory"] = str(cache_directory.relative_to(ROOT))
    summary["cache"] = stats
    started = time.perf_counter()
    deadline = started + arguments.wall_time_limit if arguments.wall_time_limit else float("inf")
    def publish() -> None:
        summary["elapsed_s"] = time.perf_counter() - started
        write_json(directory / "summary.json", summary)
        if output != directory / "summary.json":
            write_json(output, summary)
    publish()
    try:
        h2 = load_export(weight_path("h2"))
        proposal, config, metadata = load_proposal_snn_export(weight_path("proposal"))
        if int(config["time_steps"]) != 4 or float(metadata["proposal_threshold"]) != 0.51:
            raise ValueError("frozen Proposal configuration mismatch")
        proposal.to(device).eval()
        train_limit, dev_limit, epochs = {"smoke": (1, 1, 1), "pilot": (3, 2, 1), "formal": (0, 0, 20)}[arguments.mode]
        preparation_started = time.perf_counter()
        train_h2 = flatten(prepare_sequences("train_main", "h2", cache_directory, h2, proposal, device, hashes, stats, train_limit, arguments.memory_mb))
        def data_bytes(data: TrainingData) -> int:
            return sum(x.nbytes for x in (data.features, data.labels, data.teacher))
        remaining_mb = arguments.memory_mb - (data_bytes(train_h2) + 1024 ** 2 - 1) // 1024 ** 2
        train_snn = flatten(prepare_sequences("train_main", "snn", cache_directory, h2, proposal, device, hashes, stats, train_limit, remaining_mb))
        remaining_mb -= (data_bytes(train_snn) + 1024 ** 2 - 1) // 1024 ** 2
        dev = prepare_sequences("train_dev", "snn", cache_directory, h2, proposal, device, hashes, stats, dev_limit, remaining_mb)
        dev_features = np.concatenate([r.features for r in dev])
        summary["preparation_seconds"] = time.perf_counter() - preparation_started
        summary["data"] = data_diagnostics(train_h2, train_snn, dev)
        resident = sum(x.nbytes for data in (train_h2, train_snn) for x in (data.features, data.labels, data.teacher)) + sum(r.nbytes for r in dev) + dev_features.nbytes
        samples = len(train_h2.labels) if arguments.mode == "formal" else min(len(train_h2.labels), 16384)
        # Shared indices for 20 epochs are bounded and reused verbatim by B/C/D.
        if resident + samples * (epochs + 4) * 8 > arguments.memory_mb * 1024 ** 2:
            raise MemoryError("training arrays and shared indices exceed --memory-mb")
        stats["resident_arrays_bytes"] = resident
        stats["shared_sampling_bytes"] = samples * epochs * 8
        summary["samples_per_epoch"] = samples
        for name in ("frozen_ann", "frozen_snn"):
            model, info = make_model(name)
            model.to(device)
            teacher = np.concatenate([r.teacher for r in dev]) if name == "frozen_ann" else None
            summary["baselines"][name] = complete_evaluation(model, info, dev, dev_features, device, teacher)
            publish()
        for seed in (42, 43, 44):
            chosen = list("ABCD") if seed == 42 else replication_arms(summary["seed42_decision"])
            if seed != 42 and (arguments.mode != "formal" or not chosen):
                break
            shared = {epoch: balanced_indices(train_snn.labels, samples, seed, epoch) for epoch in range(1, epochs + 1)}
            arms = summary["runs"][str(seed)] = {}
            for arm in chosen:
                data = train_h2 if arm == "A" else train_snn
                arms[arm] = train_arm(arm, seed, epochs, data, samples, dev, dev_features,
                                      directory, device, shared, hashes, deadline)
                publish()
            if seed == 42:
                summary["seed42_decision"] = decide(arms, summary["baselines"])
                publish()
            del shared
        summary["primary_metrics"] = {seed: {arm: r["selected"]["scan"]["best_event"] for arm, r in arms.items()}
                                      for seed, arms in summary["runs"].items()}
        if arguments.mode == "formal":
            summary["decision"] = final_decision(summary["seed42_decision"],
                                                 {s: a for s, a in summary["runs"].items() if s != "42"}, summary["baselines"])
            summary["conclusion"] = summary["decision"]["next"]
        else:
            summary["decision"] = "pilot_only_no_model_selection_conclusion"
            summary["conclusion"] = "Limited data and epochs; do not interpret gates as full-study results"
            train_paths = study_paths("train_main")
            dev_paths = study_paths("train_dev")
            # Conservative extrapolation; epoch timings include validation separately.
            historical = ROOT / "results/runs/S0R-full-T4-POS10-KD50-S42/config.json"
            full_samples = 2249852
            if historical.exists():
                full_samples = int(json.loads(historical.read_text(encoding="utf-8"))["train_candidate_events"])
            train_seconds = sum(r["training_seconds"] for r in summary["runs"]["42"].values())
            select_seconds = sum(r["selection_seconds"] for r in summary["runs"]["42"].values())
            estimate = (summary["preparation_seconds"] * (2 * len(train_paths) + len(dev_paths)) / max(stats["misses"] + stats["hits"], 1)
                        + train_seconds * 20 * full_samples / samples
                        + select_seconds * 20 * len(dev_paths) / len(dev))
            summary["estimated_formal_seed42_seconds"] = estimate
            summary["estimate_limitations"] = "Extrapolation excludes replications; dataset density and full-role profiling may increase time"
        summary["status"] = "ok"
        publish()
        return summary
    except BaseException as error:
        summary.update({"status": "failed", "decision": "incomplete", "conclusion": str(error)})
        publish()
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", required=True)
    parser.add_argument("--mode", choices=("smoke", "pilot", "formal"), default="pilot")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--cache-directory", type=Path, help="reuse a provenance-checked cache under results/runs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--memory-mb", type=int, default=1024)
    parser.add_argument("--wall-time-limit", type=float, default=0, help="soft limit checked between training epochs; 0 disables")
    args = parser.parse_args(argv)
    if args.memory_mb <= 0 or args.wall_time_limit < 0:
        raise ValueError("invalid resource limits")
    result = execute(args)
    print(json.dumps({"status": result["status"], "decision": result["decision"],
                      "elapsed_s": result["elapsed_s"],
                      "estimated_formal_seed42_seconds": result.get("estimated_formal_seed42_seconds")}, ensure_ascii=False, indent=2))


def select_main(argv: list[str] | None = None) -> None:
    """Verify selected checkpoints and reapply gates without any training/data reads."""
    parser = argparse.ArgumentParser(description="Audit a completed Refiner study and its selected checkpoints")
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    directory = args.run_directory.resolve()
    if not within(directory, RUN_ROOT):
        raise ValueError("study directory must be under results/runs")
    if not within(args.output, ROOT / "results"):
        raise ValueError("selection audit must be written under results")
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    if summary["status"] != "ok" or summary["protocol"] != PROTOCOL or summary["frozen_inputs"] != frozen_inputs():
        raise ValueError("incomplete study or mismatched protocol/frozen inputs")
    if args.output.exists():
        raise FileExistsError("audit output already exists")
    for seed, arms in summary["runs"].items():
        for arm, result in arms.items():
            run_directory = directory / f"{arm}-S{seed}"
            checkpoint = torch.load(run_directory / "selected.pt", map_location="cpu", weights_only=False)
            history = json.loads((run_directory / "history.json").read_text(encoding="utf-8"))
            expected_epochs = 20 if summary["mode"] == "formal" else 1
            if history["status"] != "ok" or history["result"] != result or [r["epoch"] for r in history["history"]] != list(range(1, expected_epochs + 1)):
                raise ValueError("history incomplete or inconsistent with summary")
            best = max(history["history"], key=lambda r: (r["evaluation"]["scan"]["best_event"]["iou"], r["evaluation"]["scan"]["best_event"]["f1"]))
            if best["epoch"] != checkpoint["epoch"]:
                raise ValueError("selected checkpoint is not the best recorded epoch")
            if checkpoint["protocol"] != PROTOCOL or checkpoint["frozen_inputs"] != summary["frozen_inputs"] or checkpoint["info"] != result["info"]:
                raise ValueError("checkpoint provenance mismatch")
            if checkpoint["epoch"] != result["selected"]["epoch"] or checkpoint["selection"]["scan"] != result["selected"]["scan"]:
                raise ValueError("checkpoint selection mismatch")
            model, _ = make_model(arm)
            model.load_state_dict(checkpoint["model"], strict=True)
            if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError("selected checkpoint contains nonfinite values")
    if summary["mode"] == "formal":
        first = decide(summary["runs"]["42"], summary["baselines"])
        summary["decision"] = final_decision(first, {s: a for s, a in summary["runs"].items() if s != "42"}, summary["baselines"])
    elif not within(args.output, RUN_ROOT):
        raise ValueError("pilot audit must stay under results/runs")
    summary["selection_audit"] = "checkpoint metadata, state shapes and decision recomputed; no inference rerun"
    write_json(args.output, summary)
    print(json.dumps({"status": "ok", "output": str(args.output)}, ensure_ascii=False))
