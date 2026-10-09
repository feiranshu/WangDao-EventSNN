"""检查工作区 EV-UAV 数据，并维护唯一的数据索引。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = ROOT / "dataset" / "EV-UAV-dataset"
DEFAULT_INDEX = ROOT / "dataset" / "index.json"
EXPECTED_SPLIT_COUNTS = {"train": 99, "val": 24, "test": 24}
EXPECTED_KEYS = {"ev", "ev_loc", "evs_norm"}
EXPECTED_FIELDS = ("x", "y", "t", "p", "label", "name")
SENSOR_WIDTH = 346
SENSOR_HEIGHT = 260
SPLIT_SEED = 20260828


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    try:
        with open(temporary_name, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=False)
            handle.write("\n")
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: " + str(path))
    return value


def protect_frozen_artifact(
    path: Path, candidate: Mapping[str, Any], identity_key: str
) -> None:
    """Refuse to overwrite a versioned manifest or split after any drift."""
    if not path.is_file():
        return
    existing = load_json(path)
    if existing.get(identity_key) != candidate.get(identity_key):
        raise ValueError(
            f"frozen artifact changed: {path}. Create a new version instead of overwriting v1"
        )
    if existing != candidate:
        raise ValueError(
            f"frozen artifact content drifted: {path}. Review and create a new version"
        )


def _require_binary(values: np.ndarray, field: str, path: Path) -> None:
    unique = set(np.unique(values).astype(np.int64).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"{path}: {field} must be binary, got {sorted(unique)}")


def audit_sequence(path: Path, split: str, data_root: Path) -> Dict[str, Any]:
    with np.load(str(path), allow_pickle=False) as archive:
        if set(archive.files) != EXPECTED_KEYS:
            raise ValueError(f"{path}: unexpected NPZ keys {archive.files}")
        events = archive["ev"]
        locations = archive["ev_loc"]
        normalized = archive["evs_norm"]
    if events.ndim != 1 or len(events) == 0:
        raise ValueError(f"{path}: ev must be a nonempty one-dimensional array")
    if events.dtype.names != EXPECTED_FIELDS:
        raise ValueError(f"{path}: unexpected event fields {events.dtype.names}")
    if locations.shape != (len(events), 3) or locations.dtype != np.int64:
        raise ValueError(f"{path}: ev_loc must have shape [events,3] and dtype int64")
    if normalized.shape != (len(events), 6) or normalized.dtype != np.float32:
        raise ValueError(f"{path}: evs_norm must have shape [events,6] and dtype float32")
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{path}: evs_norm contains non-finite values")

    x = events["x"].astype(np.int64, copy=False)
    y = events["y"].astype(np.int64, copy=False)
    timestamps = events["t"].astype(np.float64, copy=False)
    polarity = events["p"].astype(np.int64, copy=False)
    labels = events["label"].astype(np.int64, copy=False)
    identities = events["name"].astype(np.int64, copy=False)
    if not np.all(np.isfinite(timestamps)):
        raise ValueError(f"{path}: timestamps contain non-finite values")
    if np.any(np.diff(timestamps) < 0.0):
        raise ValueError(f"{path}: timestamps are not monotonic")
    if x.min() < 0 or x.max() >= SENSOR_WIDTH:
        raise ValueError(f"{path}: x coordinate outside [0,{SENSOR_WIDTH - 1}]")
    if y.min() < 0 or y.max() >= SENSOR_HEIGHT:
        raise ValueError(f"{path}: y coordinate outside [0,{SENSOR_HEIGHT - 1}]")
    _require_binary(polarity, "p", path)
    _require_binary(labels, "label", path)
    if np.any((labels == 0) & (identities != 0)):
        raise ValueError(f"{path}: background events must use identity 0")
    if np.any((labels == 1) & (identities <= 0)):
        raise ValueError(f"{path}: target events must use a positive identity")
    reconstructed_locations = np.column_stack((x, y, np.floor(timestamps))).astype(np.int64)
    if not np.array_equal(reconstructed_locations, locations):
        raise ValueError(f"{path}: ev_loc does not match x/y/floor(t)")

    target_mask = labels == 1
    relative = path.relative_to(data_root).as_posix()
    return {
        "split": split,
        "file": path.name,
        "relative_path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "events": int(len(events)),
        "target_events": int(np.count_nonzero(target_mask)),
        "background_events": int(np.count_nonzero(~target_mask)),
        "target_fraction": float(np.mean(target_mask)),
        "target_identities": [int(value) for value in np.unique(identities[target_mask])],
        "start_ms": float(timestamps[0]),
        "end_ms": float(timestamps[-1]),
        "duration_ms": float(timestamps[-1] - timestamps[0]),
        "x_range": [int(x.min()), int(x.max())],
        "y_range": [int(y.min()), int(y.max())],
        "polarity_counts": {
            "negative": int(np.count_nonzero(polarity == 0)),
            "positive": int(np.count_nonzero(polarity == 1)),
        },
    }


def summarize(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    events = sum(int(record["events"]) for record in records)
    targets = sum(int(record["target_events"]) for record in records)
    return {
        "sequences": len(records),
        "bytes": sum(int(record["bytes"]) for record in records),
        "events": events,
        "target_events": targets,
        "background_events": events - targets,
        "target_fraction": float(targets / events) if events else 0.0,
        "duration_ms_total": sum(float(record["duration_ms"]) for record in records),
    }


def dataset_fingerprint(files: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in sorted(files, key=lambda item: str(item["relative_path"])):
        line = f"{record['relative_path']}\0{record['bytes']}\0{record['sha256']}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def build_split(records: Sequence[Mapping[str, Any]], fingerprint: str) -> Dict[str, Any]:
    train_records = [record for record in records if record["split"] == "train"]
    train_names = sorted(str(record["file"]) for record in train_records)
    validation_names = sorted(str(record["file"]) for record in records if record["split"] == "val")
    test_names = sorted(str(record["file"]) for record in records if record["split"] == "test")
    if len(train_names) != 99:
        raise ValueError("frozen split requires exactly 99 official training sequences")
    # Sample dev and calibration sequences across the full target-fraction
    # distribution. Seeded jitter gives deterministic ordering when fractions tie.
    generator = np.random.default_rng(SPLIT_SEED)
    tie_breaker = {
        str(record["file"]): float(value)
        for record, value in zip(train_records, generator.random(len(train_records)))
    }
    ordered = sorted(
        train_records,
        key=lambda record: (
            float(record["target_fraction"]), tie_breaker[str(record["file"])]
        ),
    )
    dev_ranks = {
        int(np.floor((index + 0.25) * len(ordered) / 10.0)) for index in range(10)
    }
    calibration_ranks = {
        int(np.floor((index + 0.75) * len(ordered) / 10.0)) for index in range(10)
    }
    if dev_ranks & calibration_ranks:
        raise AssertionError("stratified role ranks overlap")
    roles = {
        "train_main": sorted(
            str(record["file"])
            for rank, record in enumerate(ordered)
            if rank not in dev_ranks | calibration_ranks
        ),
        "train_dev": sorted(
            str(record["file"])
            for rank, record in enumerate(ordered)
            if rank in dev_ranks
        ),
        "hw_calibration": sorted(
            str(record["file"])
            for rank, record in enumerate(ordered)
            if rank in calibration_ranks
        ),
        "validation": validation_names,
        "test": test_names,
    }
    flattened = [name for role in ("train_main", "train_dev", "hw_calibration") for name in roles[role]]
    if len(flattened) != len(set(flattened)) or set(flattened) != set(train_names):
        raise AssertionError("training roles are not a disjoint cover of official Train")
    by_name = {str(record["file"]): record for record in records}
    role_summaries = {
        role: summarize([by_name[name] for name in names]) for role, names in roles.items()
    }
    return {
        "format_version": 1,
        "dataset": "EV-UAV",
        "dataset_fingerprint": fingerprint,
        "method": "target-fraction stratified quantile sampling with seeded tie breaking",
        "seed": SPLIT_SEED,
        "roles": roles,
        "summaries": role_summaries,
        "policy": {
            "train_main": "model fitting",
            "train_dev": "hyperparameters, thresholds and early stopping",
            "hw_calibration": "fixed-point, approximation and board calibration only",
            "validation": "stage-gated comparison; no fine-grained search",
            "test": "final frozen model only",
        },
    }


def run(data_root: Path, index_path: Path) -> Dict[str, Any]:
    data_root = data_root.resolve()
    expected_workspace_root = (ROOT / "dataset" / "EV-UAV-dataset").resolve()
    if data_root != expected_workspace_root:
        raise ValueError(
            "the formal audit accepts only the workspace dataset: " + str(expected_workspace_root)
        )
    if not data_root.is_dir():
        raise FileNotFoundError("workspace dataset is missing: " + str(data_root))

    sequence_records: List[Dict[str, Any]] = []
    for split, expected_count in EXPECTED_SPLIT_COUNTS.items():
        directory = data_root / split
        files = sorted(directory.glob(f"{split}_*.npz"))
        if len(files) != expected_count:
            raise ValueError(
                f"{split}: expected {expected_count} NPZ files, found {len(files)}"
            )
        sequence_records.extend(audit_sequence(path, split, data_root) for path in files)

    all_files = sorted(path for path in data_root.rglob("*") if path.is_file())
    formal_paths = {str(record["relative_path"]) for record in sequence_records}
    unexpected = [
        path.relative_to(data_root).as_posix()
        for path in all_files
        if path.relative_to(data_root).as_posix() not in formal_paths
    ]
    if unexpected:
        raise ValueError("数据目录包含非正式文件：" + ", ".join(unexpected))
    fingerprint = dataset_fingerprint(sequence_records)
    hash_groups: Dict[str, List[str]] = {}
    for record in sequence_records:
        hash_groups.setdefault(str(record["sha256"]), []).append(str(record["relative_path"]))
    duplicates = [paths for paths in hash_groups.values() if len(paths) > 1]
    if duplicates:
        raise ValueError("duplicate NPZ contents detected: " + repr(duplicates))

    split = build_split(sequence_records, fingerprint)
    index = {
        "format_version": 1,
        "dataset": "EV-UAV",
        "location": "dataset/EV-UAV-dataset",
        "sensor": {"width": SENSOR_WIDTH, "height": SENSOR_HEIGHT},
        "dataset_fingerprint": fingerprint,
        "file_count": len(all_files),
        "total_bytes": sum(path.stat().st_size for path in all_files),
        "official_split_counts": EXPECTED_SPLIT_COUNTS,
        "summaries": {
            split: summarize([record for record in sequence_records if record["split"] == split])
            for split in EXPECTED_SPLIT_COUNTS
        },
        "roles": split["roles"],
        "role_summaries": split["summaries"],
        "policy": split["policy"],
        "files": [
            {
                "path": record["relative_path"],
                "bytes": record["bytes"],
                "sha256": record["sha256"],
                "events": record["events"],
                "target_events": record["target_events"],
            }
            for record in sequence_records
        ],
        "status": "ok",
    }
    protect_frozen_artifact(index_path, index, "dataset_fingerprint")
    atomic_json(index, index_path)
    return index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = run(args.data_root, args.index)
    print(json.dumps({
        "status": "ok",
        "dataset_fingerprint": index["dataset_fingerprint"],
        "files": index["file_count"],
        "summaries": index["summaries"],
        "role_summaries": index["role_summaries"],
        "index": str(args.index.resolve()),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
