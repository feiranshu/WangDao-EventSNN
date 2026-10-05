"""在固定数据划分上评测锁定的 H2 基线。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.data import load_events
from wangdao.h2.constants import SENSOR_HEIGHT, SENSOR_WIDTH, WINDOW_MS, WINDOWS_PER_SEQUENCE
from wangdao.h2.numpy_model import H2Export, load_export, run_pipeline
from wangdao.h2.representation import window_indices
from wangdao.metrics import event_metrics, localization_metrics


DATA_ROOT = ROOT / "data" / "EV-UAV-dataset"
WEIGHT_PATH = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
INDEX_PATH = ROOT / "data" / "index.json"
DEFAULT_OUTPUT = ROOT / "results" / "baseline_h2.json"
ALLOWED_ROLES = ("validation", "train_dev", "hw_calibration")
ROLE_DIRECTORY = {
    "validation": "val",
    "train_dev": "train",
    "hw_calibration": "train",
}
HISTORICAL_VALIDATION = {
    "iou": 0.861630,
    "recall": 0.910772,
    "precision": 0.941070,
    "f1": 0.925673,
    "pd": 0.900944,
    "fa": 0.0000079163,
}


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


def nonempty_window_workload(events: np.ndarray, candidates: np.ndarray) -> Tuple[List[float], List[int]]:
    assigned = window_indices(events)
    keep = np.zeros(len(events), dtype=bool)
    keep[np.asarray(candidates, dtype=np.int64)] = True
    fractions: List[float] = []
    counts: List[int] = []
    for window in range(WINDOWS_PER_SEQUENCE):
        member = assigned == window
        if np.any(member):
            fractions.append(float(np.mean(keep[member])))
            counts.append(int(np.count_nonzero(keep[member])))
    return fractions, counts


def evaluate_role(
    role: str,
    filenames: Sequence[str],
    model: H2Export,
) -> Dict[str, Any]:
    directory = DATA_ROOT / ROLE_DIRECTORY[role]
    probability_parts: List[np.ndarray] = []
    label_parts: List[np.ndarray] = []
    fractions: List[float] = []
    candidate_counts: List[int] = []
    per_sequence: List[Dict[str, Any]] = []
    correct_targets = 0.0
    targets = 0.0
    false_components = 0.0
    frame_denominator = 0.0
    retained_targets = 0
    total_targets = 0
    total_candidates = 0
    total_events = 0
    started = time.perf_counter()
    for index, filename in enumerate(filenames, start=1):
        path = directory / filename
        events = load_events(path, require_labels=True)
        sequence_started = time.perf_counter()
        result = run_pipeline(events, model)
        elapsed = time.perf_counter() - sequence_started
        probabilities = result["probabilities"]
        candidates = result["candidate_indices"]
        labels = events["label"].astype(np.uint8, copy=False)
        event = event_metrics(labels, probabilities, model.event_threshold)
        localization = localization_metrics(
            events["t"], events["x"], events["y"], labels, events["name"],
            probabilities, threshold=model.event_threshold,
        )
        local_fractions, local_counts = nonempty_window_workload(events, candidates)
        fractions.extend(local_fractions)
        candidate_counts.extend(local_counts)
        correct_targets += localization["correct_targets"]
        targets += localization["targets"]
        false_components += localization["false_components"]
        frame_denominator += localization["frame_denominator"]
        retained = int(labels[np.asarray(candidates, dtype=np.int64)].sum())
        retained_targets += retained
        total_targets += int(labels.sum())
        total_candidates += len(candidates)
        total_events += len(events)
        probability_parts.append(probabilities)
        label_parts.append(labels)
        per_sequence.append({
            "file": filename,
            "events": len(events),
            "target_events": int(labels.sum()),
            "candidate_events": len(candidates),
            "proposal_recall": float(retained / labels.sum()) if labels.sum() else 1.0,
            "candidate_fraction": float(len(candidates) / len(events)),
            "event": event,
            "localization": localization,
            "elapsed_s": elapsed,
        })
        print(
            f"[{index:02d}/{len(filenames):02d}] {filename}: "
            f"events={len(events)}, candidates={len(candidates)}, "
            f"IoU={event['iou']:.6f}, elapsed={elapsed:.2f}s",
            flush=True,
        )
    probabilities = np.concatenate(probability_parts)
    labels = np.concatenate(label_parts)
    aggregate_event = event_metrics(labels, probabilities, model.event_threshold)
    pd = float(correct_targets / targets) if targets else 1.0
    denominator_pixels = frame_denominator * SENSOR_WIDTH * SENSOR_HEIGHT
    fa = float(false_components / denominator_pixels) if denominator_pixels else 0.0
    return {
        "role": role,
        "sequences": len(filenames),
        "events": total_events,
        "target_events": total_targets,
        "event": aggregate_event,
        "pd": pd,
        "fa": fa,
        "proposal_recall": float(retained_targets / total_targets) if total_targets else 1.0,
        "candidate_events": total_candidates,
        "candidate_fraction_global": float(total_candidates / total_events) if total_events else 0.0,
        "candidate_fraction_median": float(np.median(fractions)) if fractions else 0.0,
        "candidate_fraction_p95": float(np.percentile(fractions, 95)) if fractions else 0.0,
        "candidate_count_p95": float(np.percentile(candidate_counts, 95)) if candidate_counts else 0.0,
        "window_ms": WINDOW_MS,
        "elapsed_s": time.perf_counter() - started,
        "per_sequence": per_sequence,
    }


def run(role: str, output: Path) -> Dict[str, Any]:
    if role not in ALLOWED_ROLES:
        raise ValueError(f"role must be one of {ALLOWED_ROLES}; Test is intentionally locked")
    index = load_json(INDEX_PATH)
    if index.get("status") != "ok":
        raise ValueError("数据索引无效")
    fingerprint = str(index["dataset_fingerprint"])
    filenames = index["roles"][role]
    if not isinstance(filenames, list) or not filenames:
        raise ValueError("selected role is empty")
    model = load_export(WEIGHT_PATH)
    evaluation = evaluate_role(role, filenames, model)
    historical_delta = None
    historical_match = None
    if role == "validation":
        current = {
            "iou": evaluation["event"]["iou"],
            "recall": evaluation["event"]["recall"],
            "precision": evaluation["event"]["precision"],
            "f1": evaluation["event"]["f1"],
            "pd": evaluation["pd"],
            "fa": evaluation["fa"],
        }
        historical_delta = {
            key: float(current[key] - HISTORICAL_VALIDATION[key])
            for key in HISTORICAL_VALIDATION
        }
        historical_match = all(
            abs(historical_delta[key]) <= (5.0e-7 if key != "fa" else 5.0e-10)
            for key in historical_delta
        )
    result = {
        "status": "ok",
        "experiment_id": f"B1-H2-ANN-{role}-v1",
        "dataset": {
            "location": "data/EV-UAV-dataset",
            "fingerprint": fingerprint,
            "index": INDEX_PATH.relative_to(ROOT).as_posix(),
            "index_sha256": sha256_file(INDEX_PATH),
            "role": role,
            "test_used": False,
        },
        "model": {
            "name": "H2",
            "recipe": model.recipe,
            "seed": model.seed,
            "weights": WEIGHT_PATH.relative_to(ROOT).as_posix(),
            "weights_sha256": sha256_file(WEIGHT_PATH),
            "proposal_threshold": model.proposal_threshold,
            "event_threshold": model.event_threshold,
        },
        "evaluation": evaluation,
        "historical_validation": HISTORICAL_VALIDATION if role == "validation" else None,
        "historical_delta": historical_delta,
        "historical_match": historical_match,
    }
    atomic_json(result, output)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ALLOWED_ROLES, default="validation")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run(args.role, args.output)
    evaluation = result["evaluation"]
    summary = {
        "status": result["status"],
        "experiment_id": result["experiment_id"],
        "event": evaluation["event"],
        "pd": evaluation["pd"],
        "fa": evaluation["fa"],
        "proposal_recall": evaluation["proposal_recall"],
        "candidate_fraction_global": evaluation["candidate_fraction_global"],
        "elapsed_s": evaluation["elapsed_s"],
        "historical_match": result["historical_match"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
