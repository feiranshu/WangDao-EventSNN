"""Frozen inputs and strict development-only data access for the new research."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Optional

PACKAGE = Path(__file__).absolute().parent
IMPLEMENTATION = PACKAGE.parents[2]
REPOSITORY = IMPLEMENTATION.parents[1]
ANN_ROOT = REPOSITORY / "1_H2_model_improvement" / "implementation"
DEVELOPMENT_ROLES = ("train_main", "train_dev")
ROLE_COUNTS = {"train_main": 79, "train_dev": 10, "hw_calibration": 10,
               "validation": 24, "test": 24}


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(path: Path, report: dict) -> None:
    """Never replace an existing experiment record."""
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)


def research_sources() -> dict:
    """Record the diagnostic implementation used for a durable experiment record."""
    paths = sorted(PACKAGE.glob('*.py')) + sorted(PACKAGE.glob('*.json'))
    paths.append(IMPLEMENTATION / 'software/scripts/h2pro_snn.py')
    paths.append(IMPLEMENTATION / 'software/scripts/train_h2pro_refiner.py')
    return {path.relative_to(IMPLEMENTATION).as_posix():
            hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            for path in paths}


def verify_baseline(ann_root: Path = ANN_ROOT) -> dict:
    """Verify the selected teacher, never silently adopt new weights or code."""
    lock = read_json(PACKAGE / "baseline_lock.json")
    checked = {}
    for kind in ("files_sha256", "source_sha256_utf8_lf"):
        for relative, expected in lock[kind].items():
            path = Path(ann_root) / relative
            if kind == "files_sha256":
                actual = sha256(path)
            else:
                actual = hashlib.sha256(path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
            if actual != expected:
                raise ValueError("Frozen H2-pro input changed: " + relative)
            checked[relative] = actual
    return {"model_id": lock["model_id"], "threshold": lock["threshold"],
            "verified_files": checked, "baseline_lock_sha256": sha256(PACKAGE / "baseline_lock.json"),
            "research_protocol_sha256": sha256(PACKAGE / "protocol.json")}


def development_index(ann_root: Path = ANN_ROOT) -> dict:
    path = Path(ann_root) / "data/index.json"
    expected = read_json(PACKAGE / "baseline_lock.json")["files_sha256"]["data/index.json"]
    if sha256(path) != expected:
        raise ValueError("Frozen H2-pro data index changed")
    index = read_json(path)
    seen = set()
    for role, count in ROLE_COUNTS.items():
        names = index["roles"][role]
        if len(names) != count or len(set(names)) != count or seen.intersection(names):
            raise ValueError("Invalid or overlapping split: " + role)
        seen.update(names)
    return index


def role_records(role: str, limit: Optional[int] = None, ann_root: Path = ANN_ROOT) -> list:
    if role not in DEVELOPMENT_ROLES:
        raise ValueError("This research entrypoint permits only train_main and train_dev")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise ValueError("limit must be a positive integer")
    index = development_index(ann_root)
    indexed = {record["path"]: record for record in index["files"]}
    result = []
    for name in index["roles"][role][:limit]:
        if (PurePosixPath(name).name != name or "\\" in name or
                not name.startswith("train_") or not name.endswith(".npz")):
            raise ValueError("Unsafe indexed filename")
        result.append(indexed["train/" + name])
    return result


def verified_sequence(dataset_root: Path, record: dict) -> Path:
    relative = PurePosixPath(record["path"])
    if len(relative.parts) != 2 or relative.parts[0] != "train" or "\\" in record["path"]:
        raise ValueError("Only indexed development sequences are allowed")
    allowed = {r["path"]: r for role in DEVELOPMENT_ROLES for r in role_records(role)}
    if relative.as_posix() not in allowed or record != allowed[relative.as_posix()]:
        raise ValueError("Sequence is not a frozen development record")
    path = Path(dataset_root) / relative
    if not path.is_file():
        raise FileNotFoundError("Missing dataset sequence: " + str(path))
    if sha256(path) != record["sha256"]:
        raise ValueError("Dataset content hash mismatch: " + str(relative))
    return path


def audit() -> dict:
    baseline = verify_baseline()
    index = development_index()
    return {"schema_version": 1, "experiment_id": "h2pro_snn_stage0_audit",
            "status": "baseline_verified_training_not_started", "stage": "baseline_audit",
            "data_role": "metadata_only", "primary_metrics": {"role_counts": ROLE_COUNTS},
            "decision": "development_only", "training_started": False,
            "reserved_sequences_read": 0, "baseline": baseline, "research_sources_utf8_lf": research_sources(),
            "dataset_fingerprint": index["dataset_fingerprint"],
            "protocol": read_json(PACKAGE / "protocol.json"),
            "conclusion": "Inputs are locked. This audit never starts the separate manual training entrypoint."}
