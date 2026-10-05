"""A3 training, exact int8 checkpoint evaluation, and zero-update control."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as functional

from step3_common import (
    CandidateCorpus,
    FLOAT_REFERENCE_AP,
    H2_F_THRESHOLD,
    INT8_REFERENCE_AP,
    RUNTIME_ROOT,
    candidate_gates,
    checkpoint_protection,
    checkpoint_rank,
    evaluate_records,
    export_metrics,
    export_refiner_npz,
    json_write,
    load_refiner_npz,
    make_float_refiner,
    predict_model,
    predict_quantized,
    quantized_layers_from_model,
    remove_private,
    select_h2_r,
    sha256,
    working_point_view,
)
from wangdao.h2.torch_model import prototype_logits
from wangdao.h2_research import select_fair_working_point


MODEL_DESCRIPTIONS = {
    "A3-1": "natural full pass plus unweighted BCE",
    "A3-2": "mutually exclusive density-label strata plus importance-corrected BCE",
}
EVALUATION_EPOCHS = {0, 1, 2, 5, 10, 15, 20}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def environment(device: torch.device) -> Dict[str, Any]:
    return {
        "python_torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


class FullPassSampler:
    """A deterministic epoch permutation with every candidate exactly once."""

    def __init__(self, corpus: CandidateCorpus, seed: int) -> None:
        self.corpus = corpus
        self.seed = int(seed)

    def batches(self, epoch: int, batch_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(self.seed * 1_000_003 + int(epoch))
        indices = rng.permutation(len(self.corpus)).astype(np.int64, copy=False)
        return [
            (indices[first:first + batch_size], np.ones(min(batch_size, len(indices) - first), dtype=np.float32))
            for first in range(0, len(indices), batch_size)
        ]

    def report(self, batches: Sequence[tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        joined = np.concatenate([values for values, _ in batches])
        unique = np.unique(joined)
        return {
            "kind": "natural_full_pass",
            "draws": int(len(joined)),
            "unique_candidates": int(len(unique)),
            "coverage": float(len(unique) / len(self.corpus)),
            "repeat_fraction": float(1.0 - len(unique) / len(joined)),
            "omitted": int(len(self.corpus) - len(unique)),
        }


class CorrectedStratifiedSampler:
    """Four disjoint strata with explicit target-risk importance weights."""

    QUOTAS = {
        "high_positive": 512,
        "high_negative": 1536,
        "ordinary_positive": 512,
        "ordinary_negative": 1536,
    }

    def __init__(self, corpus: CandidateCorpus, seed: int) -> None:
        self.corpus = corpus
        self.seed = int(seed)
        positive = corpus.labels == 1
        high = corpus.high_density.astype(bool)
        masks = {
            "high_positive": high & positive,
            "high_negative": high & ~positive,
            "ordinary_positive": ~high & positive,
            "ordinary_negative": ~high & ~positive,
        }
        self.pools = {name: np.flatnonzero(mask) for name, mask in masks.items()}
        empty = [name for name, values in self.pools.items() if not len(values)]
        if empty:
            raise ValueError("A3-2 has empty required strata: " + ", ".join(empty))
        total_quota = float(sum(self.QUOTAS.values()))
        self.natural = {name: len(values) / len(corpus) for name, values in self.pools.items()}
        self.sampled = {name: self.QUOTAS[name] / total_quota for name in self.QUOTAS}
        self.weights = {name: self.natural[name] / self.sampled[name] for name in self.QUOTAS}

    def batches(self, epoch: int, batch_size: int) -> list[tuple[np.ndarray, np.ndarray]]:
        if batch_size != sum(self.QUOTAS.values()):
            raise ValueError("A3-2 requires batch size 4096 for its frozen quotas")
        updates = int(math.ceil(len(self.corpus) / batch_size))
        result = []
        for update in range(updates):
            rng = np.random.default_rng(
                self.seed * 1_000_003 + int(epoch) * 10_007 + int(update)
            )
            selected = []
            weights = []
            for name, quota in self.QUOTAS.items():
                values = rng.choice(self.pools[name], size=quota, replace=True)
                selected.append(values)
                weights.append(np.full(quota, self.weights[name], dtype=np.float32))
            indices = np.concatenate(selected)
            importance = np.concatenate(weights)
            order = rng.permutation(batch_size)
            result.append((indices[order], importance[order]))
        return result

    def report(self, batches: Sequence[tuple[np.ndarray, np.ndarray]]) -> Dict[str, Any]:
        joined = np.concatenate([values for values, _ in batches])
        unique = np.unique(joined)
        return {
            "kind": "corrected_stratified_with_replacement",
            "draws": int(len(joined)),
            "unique_candidates": int(len(unique)),
            "coverage": float(len(unique) / len(self.corpus)),
            "repeat_fraction": float(1.0 - len(unique) / len(joined)),
            "quotas": dict(self.QUOTAS),
            "natural_stratum_probability": self.natural,
            "sampled_stratum_probability": self.sampled,
            "importance_weight": self.weights,
        }


def class_probability(raw_logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(prototype_logits(raw_logits), dim=1)[:, 1]


def candidate_bce(
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    importance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    probability = class_probability(raw_logits)
    values = functional.binary_cross_entropy(probability, labels, reduction="none")
    if importance is not None:
        values = values * importance
    return values.mean()


def activation_saturation(model: torch.nn.Module, features: np.ndarray) -> list[Dict[str, float]]:
    """Measure exact hardtanh regions on a deterministic Train-main sample."""
    value = torch.from_numpy(features).to(next(model.parameters()).device).to(torch.float32) / 255.0
    rows = []
    with torch.no_grad():
        for index, layer in enumerate(model.layers[:-1], start=1):
            value = torch.clamp(layer(value), 0.0, 1.0)
            rows.append({
                "hidden_layer": index,
                "zero_fraction": float((value <= 0.0).to(torch.float32).mean().cpu()),
                "one_fraction": float((value >= 1.0).to(torch.float32).mean().cpu()),
                "linear_fraction": float(((value > 0.0) & (value < 1.0)).to(torch.float32).mean().cpu()),
            })
    return rows


def _state_clone(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _state_equal(first: Mapping[str, torch.Tensor], second: Mapping[str, torch.Tensor]) -> bool:
    return set(first) == set(second) and all(torch.equal(first[key], second[key]) for key in first)


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    metadata: Mapping[str, Any],
) -> None:
    package = {
        "format_version": 3,
        "dimensions": tuple(model.dimensions),
        "state_dict": _state_clone(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "metadata": dict(metadata),
    }
    with path.open("xb") as stream:
        torch.save(package, stream)


def load_checkpoint_package(path: Path) -> Mapping[str, Any]:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _views(evaluation: Mapping[str, Any], baseline: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    anchor = baseline["curve"][str(float(H2_F_THRESHOLD))]
    selection = select_fair_working_point(
        evaluation["curve"], anchor["localization"]["pd"], anchor["localization"]["fa"]
    )
    return working_point_view(evaluation, selection), selection


def _baseline_views(baseline: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
    h2_f = working_point_view(
        baseline,
        {"eligible": True, "threshold": H2_F_THRESHOLD, "metrics": baseline["curve"][str(float(H2_F_THRESHOLD))]},
    )
    h2_r = working_point_view(baseline, select_h2_r(baseline))
    return h2_f, h2_r


def _candidate_predictions(
    records: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    float_values = [predict_model(model, record["features"], batch_size, device) for record in records]
    layers = quantized_layers_from_model(model)
    int8_values = [predict_quantized(layers, record["features"]) for record in records]
    return float_values, int8_values


def evaluate_both(
    records: Sequence[Mapping[str, Any]],
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    keep_pr_points: bool = False,
) -> Dict[str, Any]:
    float_eval = evaluate_records(records, model, device, batch_size, keep_pr_points=keep_pr_points)
    int8_eval = evaluate_records(
        records,
        None,
        device,
        batch_size,
        quantized_layers=quantized_layers_from_model(model),
        keep_pr_points=keep_pr_points,
    )
    return {"float": float_eval, "int8": int8_eval}


PROGRESS_FIELDS = (
    "epoch", "final_epoch", "loss", "learning_rate", "gradient_norm_mean",
    "gradient_norm_max", "coverage", "repeat_fraction",
    "hidden1_zero_fraction", "hidden1_one_fraction", "hidden1_linear_fraction",
    "hidden2_zero_fraction", "hidden2_one_fraction", "hidden2_linear_fraction",
    "wall_s", "float_ap", "int8_ap", "int8_iou", "int8_high_recall",
    "working_point_eligible", "gates_pass",
)


def _write_progress(path: Path, row: Mapping[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROGRESS_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        stream.flush()


def _print_progress(model_id: str, seed: int, row: Mapping[str, Any]) -> None:
    message = (
        f"[{model_id} seed{seed}] epoch {int(row['epoch']):02d}/{int(row['final_epoch']):02d} "
        f"loss={float(row['loss']):.6f} lr={float(row['learning_rate']):.3e} "
        f"grad={float(row['gradient_norm_mean']):.4f} coverage={float(row['coverage']):.4f} "
        f"repeat={float(row['repeat_fraction']):.4f} wall={float(row['wall_s']):.1f}s"
        f" sat1(z/o/l)={float(row['hidden1_zero_fraction']):.3f}/"
        f"{float(row['hidden1_one_fraction']):.3f}/{float(row['hidden1_linear_fraction']):.3f}"
        f" sat2(z/o/l)={float(row['hidden2_zero_fraction']):.3f}/"
        f"{float(row['hidden2_one_fraction']):.3f}/{float(row['hidden2_linear_fraction']):.3f}"
    )
    if row.get("int8_ap") not in (None, ""):
        message += (
            f" float_ap={float(row['float_ap']):.6f} int8_ap={float(row['int8_ap']):.6f} "
            f"int8_iou={float(row['int8_iou']):.6f} high_r={float(row['int8_high_recall']):.6f} "
            f"gates={bool(row['gates_pass'])}"
        )
    print(message, flush=True)


def _evaluation_row(
    epoch: int,
    pair: Mapping[str, Mapping[str, Any]],
    baseline: Mapping[str, Any],
    h2_f: Mapping[str, Any],
    h2_r: Mapping[str, Any],
) -> Dict[str, Any]:
    int8_view, selection = _views(pair["int8"], baseline)
    float_view, _ = _views(pair["float"], baseline)
    gates = candidate_gates(int8_view, h2_f, h2_r)
    protected = checkpoint_protection(int8_view, h2_f, h2_r)
    return {
        "epoch": int(epoch),
        "float": remove_private(pair["float"]),
        "int8": remove_private(pair["int8"]),
        "float_working_point": float_view,
        "int8_working_point": int8_view,
        "int8_selection": selection,
        "gates": gates,
        "checkpoint_protected": protected,
    }


def train_candidate(
    model_id: str,
    output_dir: Path,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    seed: int = 42,
    epochs: int = 20,
    batch_size: int = 4096,
    learning_rate: float = 1.0e-5,
    smoke: bool = False,
    resume_checkpoint: Optional[Path] = None,
    epoch_offset: int = 0,
) -> Dict[str, Any]:
    if model_id not in MODEL_DESCRIPTIONS:
        raise ValueError("unknown step-3 model")
    output_dir.mkdir(parents=True, exist_ok=False)
    set_seed(seed)
    corpus = CandidateCorpus(train_records)
    sampler = FullPassSampler(corpus, seed) if model_id == "A3-1" else CorrectedStratifiedSampler(corpus, seed)
    model = make_float_refiner(seed).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
    resume_package = None
    if resume_checkpoint is not None:
        resume_package = load_checkpoint_package(Path(resume_checkpoint))
        metadata = resume_package.get("metadata", {})
        if metadata.get("model_id") != model_id or int(metadata.get("seed", -1)) != int(seed):
            raise ValueError("resume checkpoint identity mismatch")
        if int(metadata.get("epoch", -1)) != int(epoch_offset):
            raise ValueError("resume checkpoint epoch does not match --epoch-offset")
        model.load_state_dict(resume_package["state_dict"], strict=True)
        optimizer.load_state_dict(resume_package["optimizer_state_dict"])
        scheduler.load_state_dict(resume_package["scheduler_state_dict"])
        if "torch_rng_state" in resume_package:
            torch.set_rng_state(resume_package["torch_rng_state"])
        if torch.cuda.is_available() and resume_package.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(resume_package["cuda_rng_state"])
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    baseline = evaluate_records(dev_records, None, device, batch_size)
    h2_f, h2_r = _baseline_views(baseline)
    initial_pair = evaluate_both(dev_records, model, device, batch_size)
    initial_eval = _evaluation_row(epoch_offset, initial_pair, baseline, h2_f, h2_r)
    fixed_indices = np.linspace(0, len(corpus) - 1, min(65536, len(corpus)), dtype=np.int64)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "model_id": model_id,
        "description": MODEL_DESCRIPTIONS[model_id],
        "seed": int(seed),
        "smoke": bool(smoke),
        "validation_used": False,
        "test_used": False,
        "environment": environment(device),
        "configuration": {
            "epochs": int(epochs),
            "epoch_offset": int(epoch_offset),
            "final_epoch": int(epoch_offset + epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": 1.0e-4,
            "gradient_clip": 5.0,
            "scheduler": "constant",
            "loss": "binary_cross_entropy" if model_id == "A3-1" else "importance_corrected_binary_cross_entropy",
            "ema": False,
            "qat": False,
            "formal_selection_numeric_path": "exact_int8_replay",
            "resume_checkpoint": str(resume_checkpoint) if resume_checkpoint else None,
        },
        "provenance": {
            "h2": sha256(RUNTIME_ROOT / "software" / "weights" / "H2_D1_seed42.npz"),
            "index": sha256(RUNTIME_ROOT / "data" / "index.json"),
        },
        "candidate_count": len(corpus),
        "h2_f": h2_f,
        "h2_r": h2_r,
        "initial": initial_eval,
        "history": [],
    }
    json_write(output_dir / "summary.json", report)
    initial_checkpoint = output_dir / f"epoch_{epoch_offset:03d}_initial.pt"
    _save_checkpoint(
        initial_checkpoint,
        model,
        optimizer,
        scheduler,
        {"model_id": model_id, "seed": seed, "epoch": epoch_offset, "update_count": 0, "initial": True},
    )
    initial_view = initial_eval["int8_working_point"]
    initial_rank = checkpoint_rank(initial_view, epoch_offset) if initial_view.get("eligible") else (-math.inf,) * 5
    best_any: tuple[tuple[float, ...], Path, Dict[str, Any]] = (initial_rank, initial_checkpoint, initial_eval)
    best_protected = best_any if initial_eval["checkpoint_protected"] else None
    evaluation_absolute = {
        epoch_offset + value for value in EVALUATION_EPOCHS if value <= epochs
    } | {epoch_offset + epochs}
    if smoke:
        evaluation_absolute = {epoch_offset + epochs}
    consecutive_bad = 0
    try:
        final_epoch = epoch_offset + epochs
        for epoch in range(epoch_offset + 1, final_epoch + 1):
            started = time.perf_counter()
            batches = sampler.batches(epoch, batch_size)
            if smoke:
                batches = batches[:1]
            model.train()
            losses = []
            gradient_norms = []
            for selected, importance in batches:
                x = torch.from_numpy(corpus.features[selected]).to(device)
                y = torch.as_tensor(corpus.labels[selected], dtype=torch.float32, device=device)
                w = None if model_id == "A3-1" else torch.as_tensor(importance, dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                loss = candidate_bce(model(x), y, w)
                if not torch.isfinite(loss):
                    raise FloatingPointError("non-finite step-3 BCE")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                gradient_norms.append(float(norm.detach().cpu()))
            scheduler.step()
            sampling = sampler.report(batches)
            saturation = activation_saturation(model, corpus.features[fixed_indices])
            epoch_row: Dict[str, Any] = {
                "epoch": epoch,
                "loss": float(np.mean(losses)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "gradient_norm_mean": float(np.mean(gradient_norms)),
                "gradient_norm_max": float(np.max(gradient_norms)),
                "sampling": sampling,
                "activation_saturation": saturation,
                "wall_s": time.perf_counter() - started,
            }
            progress: Dict[str, Any] = {
                "epoch": epoch,
                "final_epoch": final_epoch,
                "loss": epoch_row["loss"],
                "learning_rate": epoch_row["learning_rate"],
                "gradient_norm_mean": epoch_row["gradient_norm_mean"],
                "gradient_norm_max": epoch_row["gradient_norm_max"],
                "coverage": sampling["coverage"],
                "repeat_fraction": sampling["repeat_fraction"],
                "hidden1_zero_fraction": saturation[0]["zero_fraction"],
                "hidden1_one_fraction": saturation[0]["one_fraction"],
                "hidden1_linear_fraction": saturation[0]["linear_fraction"],
                "hidden2_zero_fraction": saturation[1]["zero_fraction"],
                "hidden2_one_fraction": saturation[1]["one_fraction"],
                "hidden2_linear_fraction": saturation[1]["linear_fraction"],
                "wall_s": epoch_row["wall_s"],
            }
            if epoch in evaluation_absolute:
                pair = evaluate_both(dev_records, model, device, batch_size)
                evaluated = _evaluation_row(epoch, pair, baseline, h2_f, h2_r)
                checkpoint = output_dir / f"epoch_{epoch:03d}.pt"
                _save_checkpoint(
                    checkpoint,
                    model,
                    optimizer,
                    scheduler,
                    {"model_id": model_id, "seed": seed, "epoch": epoch, "update_count": len(batches), "initial": False},
                )
                view = evaluated["int8_working_point"]
                rank = checkpoint_rank(view, epoch) if view.get("eligible") else (-math.inf,) * 5
                item = (rank, checkpoint, evaluated)
                if item[0] > best_any[0]:
                    best_any = item
                if evaluated["checkpoint_protected"] and (best_protected is None or item[0] > best_protected[0]):
                    best_protected = item
                epoch_row["evaluation"] = evaluated
                epoch_row["checkpoint"] = checkpoint.name
                progress.update(
                    float_ap=pair["float"]["average_precision"],
                    int8_ap=pair["int8"]["average_precision"],
                    int8_iou=(view["overall"]["event"]["iou"] if view.get("eligible") else None),
                    int8_high_recall=(view["high_density"]["event"]["recall"] if view.get("eligible") else None),
                    working_point_eligible=bool(view.get("eligible")),
                    gates_pass=bool(evaluated["gates"]["passes"]),
                )
                bad = pair["int8"]["average_precision"] < initial_pair["int8"]["average_precision"] - 0.005 and not view.get("eligible")
                consecutive_bad = consecutive_bad + 1 if bad else 0
            report["history"].append(epoch_row)
            _write_progress(output_dir / "training_progress.csv", progress)
            _print_progress(model_id, seed, progress)
            json_write(output_dir / "summary.json", report, overwrite=True)
            if consecutive_bad >= 2:
                report["early_stop"] = {"epoch": epoch, "reason": "two evaluations >0.5pp AP below epoch-0 and no fair point"}
                break

        chosen = best_protected or best_any
        chosen_path = chosen[1]
        package = load_checkpoint_package(chosen_path)
        model.load_state_dict(package["state_dict"], strict=True)
        final_pair = evaluate_both(dev_records, model, device, batch_size, keep_pr_points=True)
        final_eval = _evaluation_row(int(package["metadata"]["epoch"]), final_pair, baseline, h2_f, h2_r)
        export_path = output_dir / "best_refiner_int8.npz"
        export_refiner_npz(export_path, model)
        reloaded = evaluate_records(
            dev_records, None, device, batch_size, quantized_layers=load_refiner_npz(export_path), keep_pr_points=True
        )
        if abs(float(reloaded["average_precision"]) - float(final_pair["int8"]["average_precision"])) > 1.0e-12:
            raise RuntimeError("reloaded int8 export changed AP")
        export_metrics(output_dir / "metrics" / "int8", final_pair["int8"], final_eval["int8_working_point"])
        export_metrics(output_dir / "metrics" / "float", final_pair["float"], final_eval["float_working_point"])
        report.update(
            status="complete",
            decision="smoke_only" if smoke else ("pass" if final_eval["gates"]["passes"] else "reject"),
            best_checkpoint=chosen_path.name,
            best_checkpoint_sha256=sha256(chosen_path),
            int8_export=export_path.name,
            int8_export_sha256=sha256(export_path),
            best_checkpoint_protected=best_protected is not None,
            final=final_eval,
            continuation_allowed=(not smoke and int(package["metadata"]["epoch"]) == epoch_offset + epochs),
        )
        json_write(output_dir / "summary.json", report, overwrite=True)
        print(f"[{model_id} seed{seed}] complete decision={report['decision']} best={chosen_path.name}", flush=True)
        return report
    except Exception as error:
        report.update(status="failed", decision="stop", error=f"{type(error).__name__}: {error}")
        json_write(output_dir / "summary.json", report, overwrite=True)
        print(f"[{model_id} seed{seed}] failed: {type(error).__name__}: {error}", flush=True)
        raise


def zero_update_control(
    output_dir: Path,
    train_records: Sequence[Mapping[str, Any]],
    dev_records: Sequence[Mapping[str, Any]],
    device: torch.device,
    batch_size: int = 4096,
    smoke: bool = False,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    corpus = CandidateCorpus(train_records)
    baseline_int8 = evaluate_records(dev_records, None, device, batch_size)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "smoke": bool(smoke),
        "validation_used": False,
        "test_used": False,
        "float_reference_ap": FLOAT_REFERENCE_AP,
        "int8_reference_ap": INT8_REFERENCE_AP,
        "controls": {},
    }
    json_write(output_dir / "zero_update_control.json", report)
    try:
        for model_id in ("A3-1", "A3-2"):
            set_seed(42)
            model = make_float_refiner(42).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5, weight_decay=1.0e-4)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
            sampler = FullPassSampler(corpus, 42) if model_id == "A3-1" else CorrectedStratifiedSampler(corpus, 42)
            before_state = _state_clone(model)
            before_float, before_int8 = _candidate_predictions(dev_records, model, device, batch_size)
            before_float_eval = evaluate_records(dev_records, None, device, batch_size, candidate_probabilities=before_float)
            before_int8_eval = evaluate_records(dev_records, None, device, batch_size, candidate_probabilities=before_int8)
            batches = sampler.batches(0, batch_size)
            if smoke:
                batches = batches[:1]
            model.train()
            for selected, importance in batches:
                x = torch.from_numpy(corpus.features[selected]).to(device)
                y = torch.as_tensor(corpus.labels[selected], dtype=torch.float32, device=device)
                w = None if model_id == "A3-1" else torch.as_tensor(importance, dtype=torch.float32, device=device)
                optimizer.zero_grad(set_to_none=True)
                candidate_bce(model(x), y, w).backward()
                optimizer.zero_grad(set_to_none=True)
            after_state = _state_clone(model)
            after_float, after_int8 = _candidate_predictions(dev_records, model, device, batch_size)
            after_float_eval = evaluate_records(dev_records, None, device, batch_size, candidate_probabilities=after_float)
            after_int8_eval = evaluate_records(dev_records, None, device, batch_size, candidate_probabilities=after_int8)
            checkpoint = output_dir / f"zero_update_{model_id}.pt"
            _save_checkpoint(
                checkpoint,
                model,
                optimizer,
                scheduler,
                {"model_id": model_id, "seed": 42, "epoch": 0, "update_count": 0, "scheduler_step_count": 0},
            )
            restored = make_float_refiner(42).to(device)
            package = load_checkpoint_package(checkpoint)
            restored.load_state_dict(package["state_dict"], strict=True)
            reload_state = _state_clone(restored)
            reload_float, reload_int8 = _candidate_predictions(dev_records, restored, device, batch_size)
            reload_float_eval = evaluate_records(dev_records, None, device, batch_size, candidate_probabilities=reload_float)
            reload_int8_eval = evaluate_records(dev_records, None, device, batch_size, candidate_probabilities=reload_int8)
            max_float = max(
                max(float(np.max(np.abs(a - b))) for a, b in zip(before_float, after_float)),
                max(float(np.max(np.abs(a - b))) for a, b in zip(before_float, reload_float)),
            )
            max_int8 = max(
                max(float(np.max(np.abs(a - b))) for a, b in zip(before_int8, after_int8)),
                max(float(np.max(np.abs(a - b))) for a, b in zip(before_int8, reload_int8)),
            )
            sampling = sampler.report(batches)
            if model_id == "A3-1":
                loader_contract = sampling["omitted"] == 0 if not smoke else sampling["repeat_fraction"] == 0.0
            else:
                loader_contract = (
                    sampling["quotas"] == CorrectedStratifiedSampler.QUOTAS
                    and all(value > 0.0 for value in sampling["importance_weight"].values())
                    and abs(sum(sampling["sampled_stratum_probability"].values()) - 1.0) <= 1.0e-12
                )
            checks = {
                "state_unchanged_after_backward": _state_equal(before_state, after_state),
                "state_unchanged_after_reload": _state_equal(before_state, reload_state),
                "optimizer_step_count_zero": int(package["metadata"]["update_count"]) == 0,
                "scheduler_step_count_zero": int(package["metadata"]["scheduler_step_count"]) == 0,
                "float_reference": all(abs(float(value["average_precision"]) - FLOAT_REFERENCE_AP) <= 1.0e-7 for value in (before_float_eval, after_float_eval, reload_float_eval)),
                "int8_reference": all(abs(float(value["average_precision"]) - INT8_REFERENCE_AP) <= 1.0e-7 for value in (before_int8_eval, after_int8_eval, reload_int8_eval, baseline_int8)),
                "float_probability_max_abs_le_1e_7": max_float <= 1.0e-7,
                "int8_probability_max_abs_le_1e_7": max_int8 <= 1.0e-7,
                "loader_contract": loader_contract,
            }
            report["controls"][model_id] = {
                "checks": checks,
                "passes": all(checks.values()),
                "sampling": sampling,
                "float_ap": [value["average_precision"] for value in (before_float_eval, after_float_eval, reload_float_eval)],
                "int8_ap": [value["average_precision"] for value in (before_int8_eval, after_int8_eval, reload_int8_eval)],
                "float_probability_max_abs": max_float,
                "int8_probability_max_abs": max_int8,
                "checkpoint": checkpoint.name,
                "checkpoint_sha256": sha256(checkpoint),
            }
            json_write(output_dir / "zero_update_control.json", report, overwrite=True)
        report["status"] = "pass" if all(value["passes"] for value in report["controls"].values()) else "fail"
        json_write(output_dir / "zero_update_control.json", report, overwrite=True)
        if report["status"] != "pass":
            raise RuntimeError("zero-update control failed; inspect zero_update_control.json")
        return report
    except Exception as error:
        report.update(status="fail", error=f"{type(error).__name__}: {error}")
        json_write(output_dir / "zero_update_control.json", report, overwrite=True)
        raise
