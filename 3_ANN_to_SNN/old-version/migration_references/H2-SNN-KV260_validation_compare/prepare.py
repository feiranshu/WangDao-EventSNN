"""Freeze read-only baseline copies and audit/copy Validation only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parent
OFFICIAL_COMMIT = "33e2affb51a204676536e119803ab8494e2eb0a9"
OFFICIAL_FILES = ("utils/eval.py", "dataset/basedataset.py", "dataset/ev_uav.py",
                  "configs/evisseg_evuav.yaml", "test.py", "README.md")
WEIGHTS = ("H2_D1_seed42.npz", "S1P_INT8_ACC24_MEM24_Q12.npz")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def frozen_copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(source) != sha256(destination):
            raise ValueError(f"Refusing to overwrite a differing snapshot: {destination}")
    else:
        shutil.copy2(source, destination)
    return sha256(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    args = parser.parse_args()
    source = args.source_project.resolve()
    dataset = args.dataset_root.resolve()
    index_path = source / "data/index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = index["roles"]["validation"]
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError("Expected exactly 24 distinct frozen Validation sequences")
    records = {r["path"]: r for r in index["files"]}
    audit = []
    for name in names:
        if Path(name).name != name or not name.startswith("val_"):
            raise ValueError("Only val_* files are permitted")
        rel = "val/" + name
        path = dataset / rel
        record = records[rel]
        digest = sha256(path)
        if digest != record["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {name}")
        with np.load(path, allow_pickle=False) as z:
            ev, norm, loc = z["ev"], z["evs_norm"], z["ev_loc"]
            checks = {
                "event_count": len(ev) == len(norm) == len(loc) == record["events"],
                "labels": np.array_equal(ev["label"], norm[:, 4]),
                "identities": np.array_equal(ev["name"], norm[:, 5]),
                "xy": np.array_equal(np.column_stack((ev["x"], ev["y"])), loc[:, :2]),
                "integer_ms": np.array_equal(ev["t"].astype(np.int64), loc[:, 2].astype(np.int64)),
                "ordered": bool(np.all(np.diff(ev["t"]) >= 0)),
                "label_identity": bool(np.all((ev["label"] == 0) == (ev["name"] == 0))),
                "time_range": bool(ev["t"].min() >= 0 and ev["t"].max() < 8000),
            }
            if not all(checks.values()):
                raise ValueError(f"Event correspondence failure: {name}: {checks}")
            audit.append({"file": name, "sha256": digest, "events": len(ev),
                          "target_events": int(ev["label"].sum()), "checks": checks,
                          "raw_timestamp_range_ms": [float(ev["t"].min()), float(ev["t"].max())],
                          "official_timestamp_range_ms": [int(loc[:, 2].min()), int(loc[:, 2].max())],
                          "official_boundary_events": int(np.count_nonzero(loc[:, 2].astype(np.int64) % 50 == 0))})
        frozen_copy(path, ROOT / "data/EV-UAV-dataset" / rel)

    manifest = []
    sources = list((source / "software/wangdao").rglob("*.py"))
    sources += [source / "software/scripts" / name for name in ("evaluate.py", "evaluate_s1.py")]
    sources += [source / "software/weights" / name for name in WEIGHTS]
    sources += [index_path, source / "results/baseline_h2.json", source / "results/s1_validation.json"]
    for path in sorted(sources):
        relative = path.relative_to(source)
        destination = ROOT / "baseline" / relative
        digest = frozen_copy(path, destination)
        manifest.append({"path": relative.as_posix(), "sha256": digest})
    official = []
    for relative in OFFICIAL_FILES:
        path = ROOT / "official_reference" / relative
        if not path.exists():
            url = f"https://raw.githubusercontent.com/ChenYichen9527/EV-UAV/{OFFICIAL_COMMIT}/{relative}"
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as f:
                f.write(content)
        official.append({"path": relative, "sha256": sha256(path)})
    result = {
        "schema_version": 1, "status": "ok", "experiment_id": "validation-input-audit-v1",
        "stage": "evaluation_alignment", "data_role": "validation",
        "primary_metrics": {"sequences": len(audit), "events": sum(r["events"] for r in audit),
                            "target_events": sum(r["target_events"] for r in audit),
                            "all_hashes_match": True, "all_event_rows_match": True},
        "decision": "use_frozen_validation_only", "conclusion": "24 Validation files match the original index; Test contents were not read or copied.",
        "source_project": str(source), "source_dataset": str(dataset),
        "index_sha256": sha256(index_path), "test_used": False,
        "baseline_manifest": manifest, "official_commit": OFFICIAL_COMMIT,
        "official_reference_manifest": official, "per_sequence": audit,
    }
    save_json(ROOT / "results/input_audit.json", result)
    print(json.dumps(result["primary_metrics"], indent=2))


if __name__ == "__main__":
    main()
