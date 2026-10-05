"""Data boundaries, official-compatible scoring and records for phase one.

All model development uses the existing training roles. The historical Test
split is deliberately absent from this module's public loading interface.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .data import validate_events
from .phase1_metrics import (PROTOCOL_ID, event_counts, event_summary,
                            localization_counts, localization_summary)

ROOT = Path(__file__).resolve().parents[2]
DEVELOPMENT_ROLES = ("train_main", "train_dev", "hw_calibration")
FROZEN_EVENT_THRESHOLD = 0.2800000011920929
REFERENCE_THRESHOLDS = (FROZEN_EVENT_THRESHOLD, 0.9)
THRESHOLD_GRID = tuple(sorted(set([round(i / 100, 2) for i in range(5, 96, 5)]
                                 + [FROZEN_EVENT_THRESHOLD])))
_VERIFIED: Dict[str, tuple] = {}


def register_experiment(configuration: dict, smoke: bool = False) -> Optional[str]:
    """Count distinct training recipes, not seeds, against the 12-recipe budget."""
    if smoke:
        return None
    recipe = {k: v for k, v in configuration.items()
              if k not in ("seed", "output", "run_dir", "device", "cache_mib")}
    identity = hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()
    path = ROOT / "results/runs/phase1_registry.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_suffix(".lock")
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        registry = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"limit": 12, "recipes": {}}
        if identity not in registry["recipes"]:
            if len(registry["recipes"]) >= 12:
                raise ValueError("Phase-one budget of 12 distinct training configurations reached")
            registry["recipes"][identity] = recipe
            write_json(path, registry, overwrite=path.exists())
    finally:
        os.close(descriptor)
        lock.unlink()
    return identity


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def role_paths(role: str, limit: Optional[int] = None) -> List[Path]:
    if role not in DEVELOPMENT_ROLES:
        raise ValueError("Phase-one development permits only: " + ", ".join(DEVELOPMENT_ROLES))
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("limit must be a positive integer")
    index_path = ROOT / "data/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = index["roles"][role]
    if not names or len(set(names)) != len(names):
        raise ValueError("Invalid indexed role: " + role)
    indexed = {record["path"]: record for record in index["files"]}
    result = []
    for name in names[:limit]:
        if Path(name).name != name or not name.startswith("train_") or not name.endswith(".npz"):
            raise ValueError("Unexpected development filename: " + name)
        key = "train/" + name
        expected = indexed[key]
        path = ROOT / "data/EV-UAV-dataset" / key
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing indexed data: {path}. Place or link the full dataset at "
                f"{ROOT / 'data/EV-UAV-dataset'}; do not change data/index.json."
            )
        stat = path.stat()
        signature = (stat.st_size, stat.st_mtime_ns, expected["sha256"])
        if _VERIFIED.get(str(path)) != signature:
            if sha256(path) != expected["sha256"]:
                raise ValueError("Dataset hash mismatch: " + name)
            _VERIFIED[str(path)] = signature
        result.append(path)
    return result


def load_sequence(path: Path):
    """Load indexed-format arrays for training/scoring; strip fields before inference."""
    with np.load(path, allow_pickle=False) as archive:
        ev = archive["ev"].copy()
        loc = archive["ev_loc"].copy()
        norm = archive["evs_norm"]
        if norm.ndim != 2 or norm.shape[1] < 6:
            raise ValueError("evs_norm must contain labels and identities")
        labels, identities = norm[:, 4].copy(), norm[:, 5].copy()
    validate_events(ev, require_labels=True)
    if loc.shape != (len(ev), 3) or labels.shape != (len(ev),):
        raise ValueError("Event/evaluation array length mismatch")
    if not np.array_equal(ev["label"], labels) or not np.array_equal(ev["name"], identities):
        raise ValueError("Evaluation labels/identities do not match ev")
    if not np.array_equal(np.column_stack((ev["x"], ev["y"])), loc[:, :2]):
        raise ValueError("Event/evaluation coordinates mismatch")
    if not np.array_equal(ev["t"].astype(np.int64), loc[:, 2].astype(np.int64)):
        raise ValueError("Event/evaluation timestamps mismatch")
    return ev, loc, labels, identities


def write_json(path: Path, value: Any, overwrite: bool = False) -> None:
    """Exclusive creation by default; atomic replacement for an owned run history."""
    path = Path(path)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
        return
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def score_predictions(loc, labels, identities, probabilities, threshold):
    return {
        "event": event_summary(event_counts(labels, probabilities, threshold)),
        "localization": localization_summary(
            localization_counts(loc, labels, identities, probabilities, threshold)
        ),
    }


def aggregate_scores(rows):
    if not rows:
        raise ValueError("Cannot aggregate empty sequence scores")
    events = {key: sum(int(row["event"][key]) for row in rows)
              for key in ("tp", "fp", "fn", "tn")}
    localization = {key: sum(int(row["localization"][key]) for row in rows)
                    for key in ("targets", "correct_targets", "false_components",
                                "frame_denominator")}
    for key in ("scored_localization_events", "wrapped_false_pixels"):
        if all(key in row["localization"] for row in rows):
            localization[key] = sum(int(row["localization"][key]) for row in rows)
    return {"event": event_summary(events),
            "localization": localization_summary(localization)}


def select_working_point(curve, baseline_pd, baseline_fa):
    if baseline_pd is None or baseline_fa is None:
        return {"eligible": False, "threshold": None, "metrics": None,
                "reason": "A measured same-protocol baseline Pd/Fa is required"}
    if (not math.isfinite(baseline_pd) or not math.isfinite(baseline_fa)
            or not 0 <= baseline_pd <= 1 or baseline_fa < 0):
        raise ValueError("Invalid baseline Pd/Fa")
    feasible = []
    for threshold, metrics in curve.items():
        value = float(threshold)
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError("Invalid curve threshold")
        iou, pd, fa = (metrics["event"]["iou"], metrics["localization"]["pd"],
                       metrics["localization"]["fa"])
        if any(v is None or not math.isfinite(v) for v in (iou, pd, fa)):
            continue
        if pd >= baseline_pd - 0.005 and fa <= baseline_fa * 1.10:
            feasible.append((float(iou), value, metrics))
    if not feasible:
        return {"eligible": False, "threshold": None, "metrics": None,
                "reason": "No threshold satisfies Pd loss <=0.5pp and Fa rise <=10%"}
    _, threshold, metrics = max(feasible, key=lambda row: (row[0], row[1]))
    return {"eligible": True, "threshold": threshold, "metrics": metrics,
            "reason": "Highest full-event IoU among feasible Train-dev working points"}


def pr_curve(labels, probabilities):
    """Exact threshold-grouped event PR curve, including tied/zero scores."""
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.shape == (0,) and probabilities.shape == (0,):
        return {"thresholds": [], "precision": [], "recall": [],
                "average_precision": None, "target_events": 0,
                "definition": "event average precision; not bounding-box mAP"}
    event_counts(labels, probabilities, FROZEN_EVENT_THRESHOLD)
    order = np.argsort(-probabilities, kind="stable")
    p, y = probabilities[order], (labels[order] == 1)
    ends = np.r_[np.flatnonzero(p[1:] != p[:-1]), len(p) - 1]
    tp = np.cumsum(y, dtype=np.int64)[ends]
    precision = tp / (ends + 1)
    positives = int(y.sum())
    recall = tp / positives if positives else np.zeros(len(ends))
    ap = float(np.sum(np.diff(np.r_[0., recall]) * precision)) if positives else None
    return {"thresholds": p[ends].astype(float).tolist(),
            "precision": precision.tolist(), "recall": recall.tolist(),
            "average_precision": ap, "target_events": positives,
            "definition": "event average precision; not bounding-box mAP"}


def latency_summary(block_ends_ms, processing_ms, event_times_by_block):
    """Single-server FIFO replay, not live camera or board measurement.

    Processing includes feature construction, transfer, network and output
    materialization. Batch evaluation and metric overhead must be excluded.
    """
    if not (len(block_ends_ms) == len(processing_ms) == len(event_times_by_block)):
        raise ValueError("Latency input lengths mismatch")
    completion = -math.inf
    waits, queues, latencies = [], [], []
    processing = []
    pending = []
    maximum_backlog_blocks = 0
    for end, duration, event_times in zip(block_ends_ms, processing_ms, event_times_by_block):
        if not math.isfinite(end) or not math.isfinite(duration) or duration < 0:
            raise ValueError("Invalid replay time")
        processing.append(float(duration))
        arrival = np.asarray(event_times, dtype=np.float64)
        if not np.isfinite(arrival).all() or np.any(arrival > end):
            raise ValueError("Events cannot arrive after their output deadline")
        pending = [value for value in pending if value > end]
        maximum_backlog_blocks = max(maximum_backlog_blocks, len(pending))
        start = max(float(end), completion)
        completion = start + duration
        pending.append(completion)
        waits.extend((end - arrival).tolist())
        queues.extend([start - end] * len(arrival))
        latencies.extend((completion - arrival).tolist())
    def percentiles(values):
        if not values:
            return {"p50": None, "p95": None, "p99": None}
        return dict(zip(("p50", "p95", "p99"),
                        [float(v) for v in np.percentile(values, [50, 95, 99])]))
    return {"measurement_kind": "single_server_fifo_timestamp_replay",
            "processing_ms": percentiles(processing),
            "event_wait_ms": percentiles(waits), "event_queue_ms": percentiles(queues),
            "event_end_to_end_ms": percentiles(latencies),
            "max_backlog_blocks": maximum_backlog_blocks,
            "events": len(latencies), "blocks": len(processing),
            "power_w": None, "energy_j": None,
            "hardware_claim": "software replay only; no KV260 measurements"}
