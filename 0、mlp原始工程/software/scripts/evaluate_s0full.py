"""评测冻结的 S0-P Proposal 与 S0-R Refiner 组合；不开放 Test。"""

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
from wangdao.h2.constants import SENSOR_HEIGHT, SENSOR_WIDTH
from wangdao.h2.representation import (
    build_refiner_features,
    candidate_indices,
    patch_scores_to_heatmaps,
)
from wangdao.h2.torch_model import prototype_logits
from wangdao.metrics import event_metrics, localization_metrics
from wangdao.s0p import build_s0p_sequence
from wangdao.snn import load_proposal_snn_export, load_refiner_snn_export


INDEX_PATH = ROOT / "data" / "index.json"
DEFAULT_PROPOSAL = ROOT / "software" / "weights" / "S0P_T4_POS15_KD50_S42.npz"
DEFAULT_REFINER = ROOT / "software" / "weights" / "S0R_T4_POS10_KD50_S42.npz"
ALLOWED_ROLES = ("train_dev", "validation", "hw_calibration")


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
        raise ValueError("S0-Full evaluation does not expose Test")
    index = load_json(INDEX_PATH)
    records = index.get("files")
    location = index.get("location")
    names = index.get("roles", {}).get(role)
    if not isinstance(records, list) or not isinstance(location, str):
        raise ValueError("数据索引缺少文件记录或位置")
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


def scan_event_thresholds(
    labels: np.ndarray,
    probabilities: np.ndarray,
    minimum: float,
    maximum: float,
    step: float,
) -> Dict[str, Any]:
    thresholds = np.arange(minimum, maximum + 0.5 * step, step)
    records = [event_metrics(labels, probabilities, float(value)) for value in thresholds]
    best = max(records, key=lambda item: (item["iou"], item["f1"]))
    return {
        "selection_metric": "event_iou",
        "best_threshold": best["threshold"],
        "best_event": best,
        "grid_min": minimum,
        "grid_max": maximum,
        "grid_step": step,
        "evaluated_thresholds": len(records),
    }


def evaluate(arguments: argparse.Namespace) -> Dict[str, Any]:
    device = choose_device(arguments.device)
    proposal_path = arguments.proposal.resolve()
    refiner_path = arguments.refiner.resolve()
    proposal, proposal_config, proposal_metadata = load_proposal_snn_export(
        proposal_path
    )
    refiner, refiner_config, refiner_metadata = load_refiner_snn_export(refiner_path)
    proposal = proposal.to(device).eval()
    refiner = refiner.to(device).eval()
    proposal_time_steps = int(proposal_config["time_steps"])
    refiner_time_steps = int(refiner_config["time_steps"])
    proposal_threshold = (
        float(proposal_metadata["proposal_threshold"])
        if arguments.proposal_threshold is None
        else arguments.proposal_threshold
    )
    event_threshold = (
        float(refiner_metadata["event_threshold"])
        if arguments.event_threshold is None
        else arguments.event_threshold
    )
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
    proposal_spike_sum = np.zeros(3, dtype=np.float64)
    proposal_patch_count = 0
    refiner_spike_sum = np.zeros(3, dtype=np.float64)
    refiner_event_count = 0
    started = time.perf_counter()
    for number, path in enumerate(paths, start=1):
        sequence_started = time.perf_counter()
        events = load_events(path, require_labels=True)
        sequence = build_s0p_sequence(path, proposal_time_steps)
        temporal = torch.from_numpy(sequence.temporal_features).permute(1, 0, 2, 3)
        temporal = temporal.reshape(temporal.shape[0], -1, temporal.shape[-1])
        proposal_logits = []
        with torch.no_grad():
            for start in range(0, temporal.shape[1], arguments.batch_size):
                batch = temporal[:, start : start + arguments.batch_size].to(
                    device, non_blocking=True
                )
                output = proposal(batch)
                proposal_logits.append(output.logits.cpu().numpy())
                proposal_spike_sum += output.spike_rates.cpu().numpy() * batch.shape[1]
                proposal_patch_count += batch.shape[1]
        flat_logits = np.concatenate(proposal_logits).astype(np.float64)
        proposal_scores = 1.0 / (1.0 + np.exp(-np.clip(flat_logits, -60.0, 60.0)))
        heatmaps = patch_scores_to_heatmaps(
            proposal_scores.astype(np.float32).reshape(80, 336, 16)
        )
        windows = list(range(80))
        candidates = candidate_indices(
            events, windows, heatmaps, proposal_threshold
        )
        probabilities = np.zeros(len(events), dtype=np.float32)
        if len(candidates):
            features = build_refiner_features(
                events, candidates, windows, heatmaps, proposal_threshold
            )
            probability_batches = []
            with torch.no_grad():
                for start in range(0, len(features), arguments.batch_size):
                    batch = torch.from_numpy(
                        features[start : start + arguments.batch_size]
                    ).to(device, non_blocking=True)
                    temporal_refiner = batch.unsqueeze(0).expand(
                        refiner_time_steps, -1, -1
                    )
                    output = refiner(temporal_refiner)
                    probability = torch.softmax(
                        prototype_logits(output.raw_logits), dim=1
                    )[:, 1]
                    probability_batches.append(probability.cpu().numpy())
                    refiner_spike_sum += (
                        output.spike_rates.cpu().numpy() * len(batch)
                    )
                    refiner_event_count += len(batch)
            probabilities[candidates] = np.concatenate(probability_batches)
        labels = events["label"].astype(np.uint8, copy=False)
        event = event_metrics(labels, probabilities, event_threshold)
        localization = localization_metrics(
            events["t"],
            events["x"],
            events["y"],
            labels,
            events["name"],
            probabilities,
            threshold=event_threshold,
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
    threshold_scan = (
        scan_event_thresholds(
            labels,
            probabilities,
            arguments.threshold_min,
            arguments.threshold_max,
            arguments.threshold_step,
        )
        if arguments.scan_thresholds
        else None
    )
    denominator_pixels = frame_denominator * SENSOR_WIDTH * SENSOR_HEIGHT
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
            "name": "S0-P Proposal-SNN + S0-R Refiner-SNN",
            "proposal": {
                "weight": str(proposal_path),
                "weight_sha256": sha256_file(proposal_path),
                "configuration": proposal_config,
                "metadata": proposal_metadata,
                "threshold": proposal_threshold,
            },
            "refiner": {
                "weight": str(refiner_path),
                "weight_sha256": sha256_file(refiner_path),
                "configuration": refiner_config,
                "metadata": refiner_metadata,
                "threshold": event_threshold,
            },
        },
        "evaluation": {
            "sequences": len(paths),
            "events": total_events,
            "target_events": total_targets,
            "event": event_metrics(labels, probabilities, event_threshold),
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
            "proposal_spike_rates": (
                proposal_spike_sum / max(proposal_patch_count, 1)
            ).tolist(),
            "refiner_spike_rates": (
                refiner_spike_sum / max(refiner_event_count, 1)
            ).tolist(),
            "elapsed_s": time.perf_counter() - started,
            "per_sequence": per_sequence,
            "threshold_scan": threshold_scan,
        },
        "evaluation_source_sha256": sha256_file(Path(__file__).resolve()),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--refiner", type=Path, default=DEFAULT_REFINER)
    parser.add_argument("--role", choices=ALLOWED_ROLES, default="train_dev")
    parser.add_argument("--proposal-threshold", type=float)
    parser.add_argument("--event-threshold", type=float)
    parser.add_argument("--scan-thresholds", action="store_true")
    parser.add_argument("--threshold-min", type=float, default=0.01)
    parser.add_argument("--threshold-max", type=float, default=0.99)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for value, name in (
        (arguments.proposal_threshold, "proposal-threshold"),
        (arguments.event_threshold, "event-threshold"),
    ):
        if value is not None and not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if arguments.scan_thresholds and arguments.role != "train_dev":
        raise ValueError("threshold scanning is allowed only on Train-dev")
    if not (
        0.0 <= arguments.threshold_min <= arguments.threshold_max <= 1.0
        and arguments.threshold_step > 0.0
    ):
        raise ValueError("invalid threshold scan range")
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
                "proposal_spike_rates": evaluation["proposal_spike_rates"],
                "refiner_spike_rates": evaluation["refiner_spike_rates"],
                "threshold_scan": evaluation["threshold_scan"],
                "output": str(arguments.output.resolve()),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
