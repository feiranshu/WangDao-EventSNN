"""Evaluate frozen H2 models with observable event-track HOTA.

EV-SpSegNet can be added later by providing per-event probability NPZ files
under ``<root>/<role>/<sequence>.npz``.  Each archive must contain a 1-D
``probabilities`` array aligned exactly with the original EV-UAV events.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = IMPLEMENTATION_ROOT.parents[1]
sys.path.insert(0, str(IMPLEMENTATION_ROOT / "software"))

from wangdao.a4_2_runtime import A42Pipeline  # noqa: E402
from wangdao.data import load_events  # noqa: E402
from wangdao.h2_release import protected_role_paths  # noqa: E402
from wangdao.phase1_common import sha256  # noqa: E402
from wangdao.target_tracking_metrics import (  # noqa: E402
    HOTA_ALPHAS,
    PROTOCOL_ID,
    WINDOW_MS,
    Detection,
    combine_hota_raw,
    evaluate_sequence,
    ground_truth_detections,
    hota_metrics,
    percentile95,
    predicted_components,
    assign_track_identities,
    hota_raw,
)


ACCURACY_ROOT = PROJECT_ROOT / "5、模型评测指标" / "A、模型准确度"
ACCURACY_DATA_ROOT = ACCURACY_ROOT / "02、评测数据"
PROTOCOL_PATH = ACCURACY_ROOT / "03、评测说明" / "协议与来源" / "目标轨迹协议.json"
INDEX_PATH = IMPLEMENTATION_ROOT / "data" / "index.json"
DATA_ROOT = IMPLEMENTATION_ROOT / "data" / "EV-UAV-dataset"
H2_WEIGHTS = IMPLEMENTATION_ROOT / "software" / "weights" / "H2_D1_seed42.npz"
PLUS_WEIGHTS = IMPLEMENTATION_ROOT / "software" / "weights" / "A4-2_seed42_epoch008_int8.npz"
EXPECTED_HASHES = {
    H2_WEIGHTS: "21019377617e429a599497a0a6b0f541d2c05bf7bc8fee83005deae72bb578b2",
    PLUS_WEIGHTS: "d7d42a3876c37497bdcd5206e6f35a5f949b0230dde0675fd128f989b70d05c4",
}
MODEL_OUTPUTS = {
    "H2": ACCURACY_DATA_ROOT / "H2",
    "H2-plus": ACCURACY_DATA_ROOT / "H2-plus",
    "EV-SpSegNet": ACCURACY_DATA_ROOT / "EV-SpSegNet",
}
ROLE_DIRS = {"validation": "Validation", "test": "Test"}
MODEL_THRESHOLDS = {"H2": 0.30, "H2-plus": 0.40, "EV-SpSegNet": 0.90}
CLUSTER_RADII = (1, 2, 3)
ASSOCIATION_DISTANCES = (8.0, 13.0, 20.0)
MAX_MISSED_WINDOWS = (2, 5, 20)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _verify_frozen_inputs() -> None:
    for path, expected in EXPECTED_HASHES.items():
        actual = sha256(path)
        if actual != expected:
            raise ValueError(f"frozen weight hash mismatch: {path}: {actual}")


def _index() -> Dict[str, Any]:
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def _train_dev_paths() -> List[Path]:
    index = _index()
    records = {record["path"]: record for record in index["files"]}
    paths = []
    for name in index["roles"]["train_dev"]:
        path = DATA_ROOT / "train" / name
        record = records["train/" + name]
        if sha256(path) != record["sha256"]:
            raise ValueError("Train-dev data hash mismatch: " + name)
        paths.append(path)
    return paths


def calibrate_adapter() -> Dict[str, Any]:
    """Freeze the shared adapter using labels only as ideal model output."""

    sequence_data = []
    for path in _train_dev_paths():
        events = load_events(path, require_labels=True)
        gt = ground_truth_detections(events)
        ideal = events["label"].astype(np.float32, copy=False)
        by_radius = {
            radius: predicted_components(events, ideal, 0.5, radius) for radius in CLUSTER_RADII
        }
        sequence_data.append((path.name, gt, by_radius))
    combinations = []
    for radius in CLUSTER_RADII:
        for distance in ASSOCIATION_DISTANCES:
            for max_missed in MAX_MISSED_WINDOWS:
                raw_items = []
                for _, gt, by_radius in sequence_data:
                    tracks = assign_track_identities(by_radius[radius], distance, max_missed)
                    raw_items.append(hota_raw(gt, tracks))
                combined = combine_hota_raw(raw_items)
                metrics = hota_metrics(combined)
                combinations.append(
                    {
                        "cluster_radius_px": radius,
                        "association_distance_px_per_50ms": distance,
                        "max_missed_windows": max_missed,
                        "HOTA": metrics["averaged"]["HOTA"],
                        "DetA": metrics["averaged"]["DetA"],
                        "AssA": metrics["averaged"]["AssA"],
                        "LocA": metrics["averaged"]["LocA"],
                    }
                )
    combinations.sort(
        key=lambda row: (
            -row["HOTA"],
            row["cluster_radius_px"],
            row["association_distance_px_per_50ms"],
            row["max_missed_windows"],
        )
    )
    return {
        "protocol": PROTOCOL_ID,
        "calibration_role": "train_dev",
        "selection_input": "ideal target-event labels used only as prediction mask",
        "sequence_count": len(sequence_data),
        "candidates": {
            "cluster_radius_px": list(CLUSTER_RADII),
            "association_distance_px_per_50ms": list(ASSOCIATION_DISTANCES),
            "max_missed_windows": list(MAX_MISSED_WINDOWS),
        },
        "tie_break": "smaller cluster radius, then distance, then missed-window retention",
        "selected": combinations[0],
        "all_combinations": combinations,
        "data_index_sha256": sha256(INDEX_PATH),
    }


def _counts_by_alpha(raw: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        f"{float(alpha):.2f}": {
            "tp": int(raw["tp"][index]),
            "fn": int(raw["fn"][index]),
            "fp": int(raw["fp"][index]),
        }
        for index, alpha in enumerate(raw["alphas"])
    }


def _detection_rows(
    sequence: str,
    role: str,
    model: str,
    result: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    rows = []
    matches = result["matches_iou_0_5"]
    for entity_type, detections in (
        ("gt", result["ground_truth"]),
        ("pred", result["predictions"]),
    ):
        for item in detections:
            match = matches.get((item.frame, entity_type, item.identity))
            x1, y1, x2, y2 = item.bbox
            rows.append(
                {
                    "data_role": role,
                    "model": model,
                    "sequence": sequence,
                    "window_index": item.frame,
                    "window_start_ms": item.frame * WINDOW_MS,
                    "entity_type": entity_type,
                    "track_id": item.identity,
                    "x1": x1,
                    "y1": y1,
                    "x2_exclusive": x2,
                    "y2_exclusive": y2,
                    "confidence": item.confidence if entity_type == "pred" else "",
                    "matched_track_id_iou_0_5": "" if match is None else match[0],
                    "match_iou": "" if match is None else match[1],
                    "match_status": "matched" if match is not None else "unmatched",
                }
            )
    return rows


def _sequence_row(
    sequence: str, role: str, model: str, event_count: int, result: Mapping[str, Any]
) -> Dict[str, Any]:
    averaged = result["metrics"]["averaged"]
    counts = result["metrics"]["counts"]
    return {
        "data_role": role,
        "model": model,
        "sequence": sequence,
        "events": event_count,
        **averaged,
        "longest_miss_p95_ms": percentile95(result["longest_miss_per_track_ms"]),
        **counts,
    }


def _prediction_probabilities(path: Path, expected_count: int) -> np.ndarray:
    with np.load(str(path), allow_pickle=False) as archive:
        if "probabilities" not in archive.files:
            raise ValueError(f"missing probabilities key: {path}")
        values = np.asarray(archive["probabilities"], dtype=np.float32)
    if values.shape != (expected_count,) or not np.all(np.isfinite(values)):
        raise ValueError(f"invalid probability array: {path}")
    return values


def _write_model_role(
    model: str,
    role: str,
    threshold: float,
    sequence_results: Sequence[tuple[str, int, Mapping[str, Any]]],
    calibration: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    raw = combine_hota_raw([item[2]["raw"] for item in sequence_results])
    metrics = hota_metrics(raw)
    all_misses = [
        value for _, _, result in sequence_results for value in result["longest_miss_per_track_ms"]
    ]
    summary = {
        "status": "complete",
        "metric_protocol": PROTOCOL_ID,
        "data_role": role,
        "model": model,
        "numeric_path": provenance["numeric_path"],
        "threshold": threshold,
        "window_ms": WINDOW_MS,
        "hota_alphas": HOTA_ALPHAS,
        "adapter": calibration["selected"],
        "metrics": metrics,
        "counts_by_alpha": _counts_by_alpha(raw),
        "longest_continuous_miss_p95_ms": percentile95(all_misses),
        "sequence_count": len(sequence_results),
        "events": int(sum(item[1] for item in sequence_results)),
        "data_index_sha256": sha256(INDEX_PATH),
        "provenance": dict(provenance),
        "limitations": [
            "GT boxes are observable target-event support boxes, not manually annotated physical UAV boxes.",
            "A target-silent 50 ms window creates no GT detection and is not counted as a miss.",
            "The score evaluates the model plus the shared frozen clustering and association adapter.",
        ],
    }
    output = MODEL_OUTPUTS[model] / ROLE_DIRS[role] / "目标轨迹级"
    _write_json(output / "summary.json", summary)
    sequence_rows = [
        _sequence_row(name, role, model, event_count, result)
        for name, event_count, result in sequence_results
    ]
    _write_csv(output / "per_sequence.csv", sequence_rows)
    detection_rows = [
        row
        for name, _, result in sequence_results
        for row in _detection_rows(name, role, model, result)
    ]
    _write_csv(output / "detections.csv", detection_rows)
    return summary


def evaluate_role(
    role: str,
    calibration: Mapping[str, Any],
    evuav_prediction_root: Path | None = None,
    evuav_repo_commit: str | None = None,
    evuav_weight_sha256: str | None = None,
) -> Dict[str, Dict[str, Any]]:
    selected = calibration["selected"]
    adapter_args = {
        "cluster_radius": int(selected["cluster_radius_px"]),
        "association_distance": float(selected["association_distance_px_per_50ms"]),
        "max_missed_windows": int(selected["max_missed_windows"]),
    }
    pipeline = A42Pipeline(H2_WEIGHTS, PLUS_WEIGHTS)
    collected: Dict[str, List[tuple[str, int, Mapping[str, Any]]]] = {
        "H2": [],
        "H2-plus": [],
    }
    if evuav_prediction_root is not None:
        collected["EV-SpSegNet"] = []
    paths = protected_role_paths(role)
    for index, path in enumerate(paths, 1):
        events = load_events(path, require_labels=True)
        prepared = pipeline.proposal.prepare(events)
        h2_probability = pipeline.proposal.infer_prepared(prepared, len(events))["probabilities"]
        plus_probability = pipeline.infer_prepared(prepared, len(events))["probabilities"]
        for model, probability in (("H2", h2_probability), ("H2-plus", plus_probability)):
            result = evaluate_sequence(
                events, probability, MODEL_THRESHOLDS[model], **adapter_args
            )
            collected[model].append((path.name, len(events), result))
        if evuav_prediction_root is not None:
            probability_path = evuav_prediction_root / role / path.name
            probability = _prediction_probabilities(probability_path, len(events))
            result = evaluate_sequence(
                events, probability, MODEL_THRESHOLDS["EV-SpSegNet"], **adapter_args
            )
            collected["EV-SpSegNet"].append((path.name, len(events), result))
        print(f"[{role}] {index:02d}/{len(paths):02d} {path.name}", flush=True)
    summaries = {}
    for model, sequence_results in collected.items():
        if model == "H2":
            provenance = {
                "numeric_path": "exact_int8_replay",
                "weights": str(H2_WEIGHTS),
                "weights_sha256": sha256(H2_WEIGHTS),
            }
        elif model == "H2-plus":
            provenance = {
                "numeric_path": "exact_int8_replay",
                "weights": str(PLUS_WEIGHTS),
                "weights_sha256": sha256(PLUS_WEIGHTS),
                "model_id": "A4-2",
                "seed": 42,
                "epoch": 8,
            }
        else:
            if not evuav_repo_commit or not evuav_weight_sha256:
                raise ValueError("EV-SpSegNet import requires repository commit and weight SHA-256")
            provenance = {
                "numeric_path": "official_float_cuda_import",
                "prediction_root": str(evuav_prediction_root),
                "repository": "https://github.com/ChenYichen9527/EV-UAV",
                "repository_commit": evuav_repo_commit,
                "weights_sha256": evuav_weight_sha256,
            }
        summaries[model] = _write_model_role(
            model,
            role,
            MODEL_THRESHOLDS[model],
            sequence_results,
            calibration,
            provenance,
        )
    return summaries


def write_pending_evuav() -> None:
    _write_json(
        ACCURACY_DATA_ROOT / "EV-SpSegNet" / "实测状态.json",
        {
            "status": "pending_cuda_evaluation",
            "model": "EV-SpSegNet",
            "reason": "The current computer has no NVIDIA CUDA runtime or Docker; HOTA cannot be inferred from paper Pd/ACC.",
            "required_probability_layout": "<root>/<validation|test>/<sequence>.npz",
            "required_npz_key": "probabilities",
            "probability_alignment": "one float per original event, in unchanged event order",
            "threshold": MODEL_THRESHOLDS["EV-SpSegNet"],
            "required_provenance": ["official repository commit", "pretrained weight SHA-256"],
            "repository": "https://github.com/ChenYichen9527/EV-UAV",
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", nargs="+", choices=("validation", "test"), default=("validation", "test"))
    parser.add_argument("--evuav-prediction-root", type=Path)
    parser.add_argument("--evuav-repo-commit")
    parser.add_argument("--evuav-weight-sha256")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _verify_frozen_inputs()
    calibration = calibrate_adapter()
    _write_json(PROTOCOL_PATH, calibration)
    all_summaries = {}
    for role in args.roles:
        all_summaries[role] = evaluate_role(
            role,
            calibration,
            args.evuav_prediction_root,
            args.evuav_repo_commit,
            args.evuav_weight_sha256,
        )
    if args.evuav_prediction_root is None:
        write_pending_evuav()
    print(json.dumps(_jsonable(all_summaries), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
