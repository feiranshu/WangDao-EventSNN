"""Read-only S0 audit of the frozen H2-pro contract.

Run from any directory: python 3_ANN_to_SNN/s0_audit.py
The script never opens Validation/Test event files or changes upstream artifacts.
"""
from __future__ import annotations

import hashlib
import json
import csv
from pathlib import Path


HERE = Path(__file__).absolute().parent
ROOT = HERE.parent.parent
H2 = ROOT / "1_H2_model_improvement"
STUDY = H2 / "seven_step_improvement_study" / "H2-pro"
FROZEN = STUDY / "frozen" / "H2-pro-Causal-ANN-v1"
EVAL = ROOT / "5_model_evaluation"
OUT = HERE / "audit.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def checked_hash(path: Path, expected: str) -> dict:
    actual = sha256(path)
    check(actual == expected, f"SHA-256 mismatch: {path}")
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": actual}


def main() -> None:
    manifest = read(FROZEN / "model_manifest.json")
    decision = read(H2 / "implementation" / "results" / "final_H2-pro" / "deployment_decision.json")
    metadata = manifest["metadata"]
    config = decision["configuration"]
    check(manifest["model_id"] == decision["model_id"] == "H2-pro-Causal-ANN-v1", "model identity")
    for key in ("variant", "epoch", "threshold", "window_ms"):
        check(metadata[key] == config[key], f"frozen {key}")
    check(config["seed"] == 42 and config["numeric_path"] == "exact_int8_replay", "seed or numeric path")
    check(metadata["dimensions"] == [36, 64, 32, 4] and metadata["input_dtype"] == "uint8", "Refiner shape/input")
    check(metadata["feature_protocol"] == "H2-pro-stage1-v1", "feature protocol")
    check(config["weight_sha256"] == manifest["teacher_int8_sha256"], "decision weight digest")

    index_path = H2 / "implementation" / "data" / "index.json"
    files = [
        checked_hash(FROZEN / "teacher_exact_int8.npz", manifest["teacher_int8_sha256"]),
        checked_hash(FROZEN / "teacher_float.pt", manifest["teacher_float_sha256"]),
        checked_hash(STUDY / "protocol.json", manifest["protocol_sha256"]),
        checked_hash(STUDY / "selection_protocol_v2.json", manifest["selection_protocol_sha256"]),
        checked_hash(index_path, manifest["index_sha256"]),
    ]
    source_dir = H2 / "implementation" / "software" / "weights"
    files.append(checked_hash(source_dir / "H2_D1_seed42.npz", manifest["source_weights_sha256"]["H2_D1_seed42.npz"]))
    lock = read(FROZEN / "acceptance_lock.json")
    for suffix in ("src/stream_state.py", "src/causal_feature_builder.py", "src/motion_coherence.py"):
        path = STUDY / suffix
        expected = next(value for key, value in lock["digests"].items() if key.replace("\\", "/").endswith(suffix))
        files.append(checked_hash(path, expected))
    index = read(index_path)
    roles = {name: len(index["roles"][name]) for name in
             ("train_main", "train_dev", "hw_calibration", "validation", "test")}
    check(roles == {"train_main": 79, "train_dev": 10, "hw_calibration": 10,
                    "validation": 24, "test": 24}, "data roles")
    role_names = [name for names in (index["roles"][key] for key in roles) for name in names]
    check(len(role_names) == len(set(role_names)) == index["file_count"] == 147, "role partition")
    check(index["sensor"] == {"width": 346, "height": 260}, "sensor geometry")

    protocol = read(STUDY / "protocol.json")
    acceptance = read(STUDY / "acceptance_protocol.json")
    check(protocol["protocol_id"] == metadata["feature_protocol"], "protocol id")
    check(protocol["output_policy"] == "publish only after watermark reaches right boundary; discard final partial window", "publication rule")
    check(acceptance["candidate_threshold"] == 0.35 and acceptance["data_prefix"].startswith("Only complete 100 ms"), "acceptance rule")

    accuracy = {}
    costs = {}
    profile_path = EVAL / "B_model_complexity" / "02_model_records" / "formal" / "h2_pro__software_exact_int8" / "common" / "model_profile.json"
    profile = read(profile_path)
    check(profile["source_weights_sha256"] == manifest["teacher_int8_sha256"], "B profile weights")
    check(profile["summary"] == {"total_weight_parameters": 12160,
                                  "total_tensor_payload_bytes": 12912,
                                  "persistent_proposal_history_bytes": 11968}, "B profile storage")
    for role in ("validation", "test"):
        apath = EVAL / "A_model_accuracy" / "02_model_records" / "formal" / "h2_pro__software_exact_int8" / role / "event_level" / "summary.json"
        cpath = EVAL / "B_model_complexity" / "02_model_records" / "formal" / "h2_pro__software_exact_int8" / role / "summary.json"
        a = read(apath)["working_point"]
        c = read(cpath)["summary"]
        check(a["threshold"] == 0.35, f"{role} threshold")
        e = a["overall"]["event"]
        check(abs(e["iou"] - e["tp"] / (e["tp"] + e["fp"] + e["fn"])) < 1e-12, f"{role} IoU")
        check(c["sequences"] == 24 and c["effective_acs_per_sequence"] == 0, f"{role} B counts")
        check(c["dense_macs_per_sequence"] >= c["effective_macs_per_sequence"] > 0, f"{role} B order")
        comparison = EVAL / "B_model_complexity" / "01_metric_comparisons"
        for rel, expected in (
            (f"B03_dense_mac/{role}.csv", c["dense_macs_per_sequence"]),
            (f"B04_effective_operations/B04a_effective_mac/{role}.csv", c["effective_macs_per_sequence"]),
            (f"B04_effective_operations/B04b_effective_ac/{role}.csv", 0),
            (f"B04_effective_operations/B04c_mac_plus_ac/{role}.csv", c["effective_macs_per_sequence"]),
        ):
            with (comparison / rel).open("r", encoding="utf-8-sig", newline="") as stream:
                row = next(item for item in csv.DictReader(stream) if item["evaluation_id"] == "h2_pro__software_exact_int8")
            check(row["status"] == "complete" and abs(float(row["value"]) - expected) < 1e-6,
                  f"{role} {rel} comparison")
        accuracy[role] = {"source": apath.relative_to(ROOT).as_posix(), "event_iou": e["iou"],
                          "ap": a["average_precision"], "precision": e["precision"],
                          "recall": e["recall"], "high_density_recall": a["high_density"]["event"]["recall"]}
        costs[role] = {"source": cpath.relative_to(ROOT).as_posix(), **c,
                       "scope": "Proposal + conditional Refiner exact-int8 MAC; excludes features, routing, postprocessing, latency and memory workspace"}

    report = {
        "stage": "S0", "status": "verified_read_only", "model_id": manifest["model_id"],
        "variant": config["variant"], "seed": config["seed"], "epoch": config["epoch"],
        "threshold": config["threshold"], "numeric_path": config["numeric_path"],
        "window_ms": config["window_ms"], "feature_dimensions": metadata["dimensions"],
        "input_dtype": metadata["input_dtype"], "source_hashes": files,
        "roles": roles, "sensor": index["sensor"],
        "model_profile": {"source": profile_path.relative_to(ROOT).as_posix(), **profile["summary"]},
        "input_fields": ["x", "y", "t", "p"],
        "publication_rule": protocol["output_policy"],
        "time_origin": protocol["time_origin"],
        "accuracy_historical_locked": accuracy, "b03_b04_historical_locked": costs,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"S0 frozen contract verified: {OUT}")


if __name__ == "__main__":
    main()
