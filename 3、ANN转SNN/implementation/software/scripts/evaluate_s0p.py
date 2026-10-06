"""在固定数据角色上评测冻结的 S0-P Proposal 与 H2 Refiner。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.data import load_events
from wangdao.h2.constants import (
    PROPOSAL_DIMENSIONS,
    REFINER_DIMENSIONS,
    SENSOR_HEIGHT,
    SENSOR_WIDTH,
)
from wangdao.h2.numpy_model import _refiner_probability, load_export, quantized_mlp
from wangdao.h2.representation import (
    build_refiner_features,
    candidate_indices,
    patch_scores_to_heatmaps,
)
from wangdao.metrics import event_metrics, localization_metrics
from wangdao.s0p import build_s0p_sequence
from wangdao.snn import ProposalSNN, load_proposal_snn_export


INDEX_PATH = ROOT / "data" / "index.json"
H2_WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
ALLOWED_ROLES = ("validation", "train_dev", "hw_calibration")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary_name, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def role_paths(role: str) -> List[Path]:
    if role not in ALLOWED_ROLES:
        raise ValueError("S0-P evaluation does not expose Test")
    index = load_json(INDEX_PATH)
    if index.get("status") != "ok":
        raise ValueError("数据索引无效")
    location = index.get("location")
    records = index.get("files")
    names = index.get("roles", {}).get(role)
    if not isinstance(location, str) or not isinstance(records, list):
        raise ValueError("数据索引缺少位置或文件记录")
    if not isinstance(names, list) or not names:
        raise ValueError("所选数据角色为空")
    dataset_root = ROOT / location
    by_name = {
        Path(str(record["path"])).name: dataset_root / str(record["path"])
        for record in records
        if isinstance(record, dict) and "path" in record
    }
    paths = [by_name[str(name)] for name in names]
    absent = [str(path) for path in paths if not path.is_file()]
    if absent:
        raise FileNotFoundError("数据文件缺失：" + ", ".join(absent))
    return paths


def load_student(
    checkpoint_path: Path, device: torch.device
) -> tuple[ProposalSNN, Dict[str, Any], int]:
    if checkpoint_path.suffix.lower() == ".npz":
        student, config, metadata = load_proposal_snn_export(checkpoint_path)
        epoch = int(metadata.get("checkpoint_epoch", 0))
        if not all(
            torch.isfinite(value).all() for value in student.state_dict().values()
        ):
            raise ValueError("冻结权重含非有限参数")
        student.eval()
        return student.to(device), config, epoch
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint 缺少模型参数")
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint 缺少训练配置")
    student = ProposalSNN(
        beta=float(config["beta"]),
        threshold=float(config["threshold"]),
        surrogate_slope=float(config["surrogate_slope"]),
        readout=str(config["readout"]),
    )
    student.load_state_dict(checkpoint["model"])
    if not all(torch.isfinite(value).all() for value in student.state_dict().values()):
        raise ValueError("checkpoint 含非有限参数")
    student.eval()
    return student.to(device), config, int(checkpoint["epoch"])


def evaluate(arguments: argparse.Namespace) -> Dict[str, Any]:
    device = choose_device(arguments.device)
    checkpoint_path = arguments.checkpoint.resolve()
    student, config, epoch = load_student(checkpoint_path, device)
    time_steps = int(config["time_steps"])
    h2 = load_export(H2_WEIGHT_PATH)
    paths = role_paths(arguments.role)
    probability_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    per_sequence: List[Dict[str, Any]] = []
    retained_targets = 0
    total_targets = 0
    total_candidates = 0
    total_events = 0
    correct_targets = 0.0
    localization_targets = 0.0
    false_components = 0.0
    frame_denominator = 0.0
    spike_sum = np.zeros(3, dtype=np.float64)
    patch_count = 0
    input_nonzero = 0
    started = time.perf_counter()
    for number, path in enumerate(paths, start=1):
        sequence_started = time.perf_counter()
        events = load_events(path, require_labels=True)
        sequence = build_s0p_sequence(path, time_steps)
        temporal = torch.from_numpy(sequence.temporal_features).permute(1, 0, 2, 3)
        temporal = temporal.reshape(temporal.shape[0], -1, temporal.shape[-1])
        logits = []
        with torch.no_grad():
            for start in range(0, temporal.shape[1], arguments.chunk_size):
                batch = temporal[:, start:start + arguments.chunk_size].to(
                    device, non_blocking=True
                )
                output = student(batch)
                logits.append(output.logits.cpu().numpy())
                spike_sum += output.spike_rates.cpu().numpy() * batch.shape[1]
                patch_count += batch.shape[1]
                input_nonzero += int(torch.count_nonzero(batch).item())
        flat_logits = np.concatenate(logits).astype(np.float64)
        scores = 1.0 / (1.0 + np.exp(-np.clip(flat_logits, -60.0, 60.0)))
        heatmaps = patch_scores_to_heatmaps(
            scores.astype(np.float32).reshape(80, 336, 16)
        )
        windows = list(range(80))
        candidates = candidate_indices(
            events, windows, heatmaps, arguments.proposal_threshold
        )
        probabilities = np.zeros(len(events), dtype=np.float32)
        if len(candidates):
            features = build_refiner_features(
                events,
                candidates,
                windows,
                heatmaps,
                arguments.proposal_threshold,
            )
            raw = quantized_mlp(features, h2.refiner_layers)
            probabilities[candidates] = _refiner_probability(raw, h2.prototypes)
        labels = events["label"].astype(np.uint8, copy=False)
        event = event_metrics(labels, probabilities, h2.event_threshold)
        localization = localization_metrics(
            events["t"],
            events["x"],
            events["y"],
            labels,
            events["name"],
            probabilities,
            threshold=h2.event_threshold,
        )
        retained = int(labels[candidates].sum())
        retained_targets += retained
        total_targets += int(labels.sum())
        total_candidates += len(candidates)
        total_events += len(events)
        correct_targets += localization["correct_targets"]
        localization_targets += localization["targets"]
        false_components += localization["false_components"]
        frame_denominator += localization["frame_denominator"]
        probability_parts.append(probabilities)
        label_parts.append(labels)
        elapsed = time.perf_counter() - sequence_started
        per_sequence.append(
            {
                "file": path.name,
                "event": event,
                "localization": localization,
                "proposal_recall": retained / int(labels.sum())
                if labels.sum()
                else 1.0,
                "candidate_fraction": len(candidates) / len(events),
                "elapsed_s": elapsed,
            }
        )
        print(
            f"[{number:02d}/{len(paths):02d}] {path.name}: "
            f"IoU={event['iou']:.6f}, candidates={len(candidates)}, "
            f"elapsed={elapsed:.2f}s",
            flush=True,
        )
    labels = np.concatenate(label_parts)
    probabilities = np.concatenate(probability_parts)
    denominator_pixels = frame_denominator * SENSOR_WIDTH * SENSOR_HEIGHT
    mean_spike_rates = spike_sum / patch_count
    spike_counts = [
        int(round(mean_spike_rates[index] * time_steps * patch_count * width))
        for index, width in enumerate(PROPOSAL_DIMENSIONS[1:])
    ]
    event_driven_by_layer = [
        input_nonzero * PROPOSAL_DIMENSIONS[1],
        spike_counts[0] * PROPOSAL_DIMENSIONS[2],
        spike_counts[1] * PROPOSAL_DIMENSIONS[3],
    ]
    proposal_synaptic_proxy = sum(event_driven_by_layer)
    refiner_macs = sum(
        left * right
        for left, right in zip(REFINER_DIMENSIONS, REFINER_DIMENSIONS[1:])
    )
    hybrid_synaptic_proxy = proposal_synaptic_proxy + total_candidates * refiner_macs
    return {
        "status": "ok",
        "experiment_id": arguments.experiment_id,
        "dataset": {
            "index": INDEX_PATH.relative_to(ROOT).as_posix(),
            "index_sha256": sha256_file(INDEX_PATH),
            "role": arguments.role,
            "test_used": False,
        },
        "model": {
            "name": "S0-P Proposal-SNN + H2 Refiner",
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "checkpoint_epoch": epoch,
            "training_config": config,
            "proposal_threshold": arguments.proposal_threshold,
            "event_threshold": h2.event_threshold,
            "h2_refiner_weights_sha256": sha256_file(H2_WEIGHT_PATH),
        },
        "evaluation": {
            "sequences": len(paths),
            "events": total_events,
            "target_events": total_targets,
            "event": event_metrics(labels, probabilities, h2.event_threshold),
            "pd": correct_targets / localization_targets
            if localization_targets
            else 1.0,
            "fa": false_components / denominator_pixels
            if denominator_pixels
            else 0.0,
            "proposal_recall": retained_targets / total_targets
            if total_targets
            else 1.0,
            "candidate_events": total_candidates,
            "candidate_fraction_global": total_candidates / total_events
            if total_events
            else 0.0,
            "patches": patch_count,
            "input_nonzero": input_nonzero,
            "mean_spike_rates": mean_spike_rates.tolist(),
            "spike_counts": spike_counts,
            "event_driven_synaptic_accumulations_by_layer": event_driven_by_layer,
            "event_driven_synaptic_accumulations": proposal_synaptic_proxy,
            "hybrid_synaptic_proxy": hybrid_synaptic_proxy,
            "elapsed_s": time.perf_counter() - started,
            "per_sequence": per_sequence,
        },
        "evaluation_source_sha256": sha256_file(Path(__file__).resolve()),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--role", choices=ALLOWED_ROLES, default="validation")
    parser.add_argument("--proposal-threshold", type=float, default=0.51)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    if not 0.0 <= arguments.proposal_threshold <= 1.0:
        raise ValueError("proposal-threshold must be in [0, 1]")
    return arguments


def main() -> None:
    arguments = parse_arguments()
    result = evaluate(arguments)
    atomic_json(result, arguments.output)
    evaluation = result["evaluation"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "experiment_id": result["experiment_id"],
                "event": evaluation["event"],
                "pd": evaluation["pd"],
                "fa": evaluation["fa"],
                "proposal_recall": evaluation["proposal_recall"],
                "candidate_fraction_global": evaluation[
                    "candidate_fraction_global"
                ],
                "mean_spike_rates": evaluation["mean_spike_rates"],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
