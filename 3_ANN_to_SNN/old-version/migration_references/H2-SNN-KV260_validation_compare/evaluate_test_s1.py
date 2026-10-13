"""One-shot S1 Test evaluation at two thresholds frozen before Test access.

This script never selects a threshold from Test.  It verifies the frozen source,
weights, official evaluator and all 24 Test files, runs S1 once per sequence,
saves the continuous probabilities, and scores the same predictions at 0.28
and 0.9 with both the compatible and pinned-official metric implementations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

from adapters import Models
from metrics import (PROTOCOL_ID, event_counts, event_summary, localization_counts,
                     localization_summary, sum_counts)
from prepare import ROOT, OFFICIAL_COMMIT, save_json, sha256
from reference_check import ReferenceEvaluator
from wangdao.data import validate_events


THRESHOLDS = (
    ("s1_frozen_0p28", 0.2800000011920929),
    ("official_public_0p9", 0.9),
)


def verify_frozen_test_inputs():
    validation_audit_path = ROOT / "results/input_audit.json"
    validation_audit = json.loads(validation_audit_path.read_text(encoding="utf-8"))

    for record in validation_audit["baseline_manifest"]:
        path = ROOT / "baseline" / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Frozen baseline changed: " + record["path"])
    for record in validation_audit["official_reference_manifest"]:
        path = ROOT / "official_reference" / record["path"]
        if sha256(path) != record["sha256"]:
            raise ValueError("Pinned official reference changed: " + record["path"])

    index_path = ROOT / "baseline/data/index.json"
    index_digest = sha256(index_path)
    if index_digest != validation_audit["index_sha256"]:
        raise ValueError("Frozen dataset index changed")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    names = index["roles"]["test"]
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError("Expected exactly 24 distinct Test sequences")

    indexed = {record["path"]: record for record in index["files"]}
    audited = []
    for name in names:
        if Path(name).name != name or not name.startswith("test_"):
            raise ValueError("Unexpected Test filename: " + name)
        relative = "test/" + name
        if relative not in indexed:
            raise ValueError("Test file absent from frozen index: " + relative)
        expected = indexed[relative]
        path = ROOT / "data/EV-UAV-dataset" / relative
        if not path.is_file():
            raise FileNotFoundError("Missing Test file: " + str(path))
        digest = sha256(path)
        if digest != expected["sha256"]:
            raise ValueError("Test dataset hash mismatch: " + name)

        with np.load(path, allow_pickle=False) as z:
            ev, norm, loc = z["ev"], z["evs_norm"], z["ev_loc"]
            checks = {
                "event_count": len(ev) == len(norm) == len(loc) == expected["events"],
                "labels": np.array_equal(ev["label"], norm[:, 4]),
                "identities": np.array_equal(ev["name"], norm[:, 5]),
                "xy": np.array_equal(np.column_stack((ev["x"], ev["y"])), loc[:, :2]),
                "integer_ms": np.array_equal(ev["t"].astype(np.int64),
                                                loc[:, 2].astype(np.int64)),
                "ordered": bool(np.all(np.diff(ev["t"]) >= 0)),
                "label_identity": bool(np.all((ev["label"] == 0) == (ev["name"] == 0))),
                "time_range": bool(ev["t"].min() >= 0 and ev["t"].max() < 8000),
            }
            if not all(checks.values()):
                raise ValueError(f"Test event correspondence failure: {name}: {checks}")
            audited.append({
                "file": name,
                "sha256": digest,
                "events": len(ev),
                "target_events": int(ev["label"].sum()),
                "checks": checks,
            })

    return validation_audit, index_digest, audited


def s1_prediction(models, events):
    # Reuse the exact adapter path already regression-tested on all Validation data.
    # Models.predict also computes the other frozen baselines, but only S1 is retained.
    for name, probabilities, candidates in models.predict(events):
        if name == "s1_mixed":
            return probabilities, candidates
    raise RuntimeError("S1 prediction was not produced")


def run(output):
    started = time.perf_counter()
    torch.set_num_threads(1)
    validation_audit, index_digest, test_records = verify_frozen_test_inputs()
    models = Models()
    if models.threshold != THRESHOLDS[0][1]:
        raise ValueError(f"Frozen S1 threshold changed: {models.threshold!r}")

    predictions_root = output.parent / (output.stem + "_predictions")
    if predictions_root.exists():
        raise FileExistsError("Will not overwrite Test predictions: " + str(predictions_root))

    rows = {name: [] for name, _ in THRESHOLDS}
    references = {name: ReferenceEvaluator() for name, _ in THRESHOLDS}
    prediction_manifest = []

    for number, record in enumerate(test_records, 1):
        path = ROOT / "data/EV-UAV-dataset/test" / record["file"]
        with np.load(path, allow_pickle=False) as z:
            ev = z["ev"].copy()
            loc = z["ev_loc"].copy()
            norm = z["evs_norm"].copy()
        validate_events(ev, require_labels=True)
        labels, identities = norm[:, 4], norm[:, 5]

        inference_started = time.perf_counter()
        probabilities, candidates = s1_prediction(models, ev)
        probabilities = np.asarray(probabilities, dtype=np.float32)
        candidates = np.asarray(candidates, dtype=np.int64)
        if probabilities.shape != labels.shape or not np.isfinite(probabilities).all():
            raise ValueError("Invalid S1 probabilities for " + record["file"])

        destination = predictions_root / record["file"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as f:
            np.savez_compressed(
                f,
                probabilities=probabilities,
                event_index=np.arange(len(ev), dtype=np.int64),
                source_sha256=np.array(record["sha256"]),
                candidate_indices=candidates,
                frozen_thresholds=np.array([value for _, value in THRESHOLDS]),
            )
        prediction_manifest.append({
            "file": destination.name,
            "sha256": sha256(destination),
            "source_sha256": record["sha256"],
            "events": len(ev),
            "candidate_events": len(candidates),
        })

        message = []
        for name, threshold in THRESHOLDS:
            event = event_counts(labels, probabilities, threshold)
            localization = localization_counts(
                loc, labels, identities, probabilities, threshold
            )
            references[name].add(
                record["file"], loc, labels, identities, probabilities, threshold
            )
            rows[name].append({
                "file": record["file"],
                "source_sha256": record["sha256"],
                "prediction_sha256": prediction_manifest[-1]["sha256"],
                "events": len(ev),
                "event_counts": event,
                "localization_counts": localization,
            })
            summary = event_summary(event)
            message.append(f"{threshold:g}:IoU={summary['iou']:.6f}")
        print(
            f"[{number:02d}/{len(test_records):02d}] {path.name} | "
            + "; ".join(message)
            + f" | {time.perf_counter() - inference_started:.2f}s",
            flush=True,
        )

    results = {}
    for name, threshold in THRESHOLDS:
        event = event_summary(sum_counts([row["event_counts"] for row in rows[name]]))
        localization = localization_summary(
            sum_counts([row["localization_counts"] for row in rows[name]])
        )
        reference = references[name].summary(threshold)
        deltas = {
            "iou": event["iou"] - reference["iou"],
            "official_acc": event["official_acc"] - reference["official_acc"],
            "pd": localization["pd"] - reference["pd"],
            "fa": localization["fa"] - reference["fa"],
        }
        counts_equal = all(
            localization[key] == reference[key]
            for key in ("targets", "correct_targets", "false_components",
                        "frame_denominator")
        )
        parity = (
            counts_equal
            and max(abs(deltas[key]) for key in ("iou", "official_acc")) <= 1e-7
            and abs(deltas["pd"]) <= 1e-7
            and abs(deltas["fa"]) <= 1e-12
        )
        if not parity:
            raise AssertionError(f"Official Test metric parity failed: {name}: {deltas}")
        results[name] = {
            "threshold": threshold,
            "event": event,
            "localization": localization,
            "official_reference": reference,
            "official_parity_passed": True,
            "integer_counts_exact": counts_equal,
            "deltas": deltas,
            "per_sequence": rows[name],
        }

    weight_hashes = {
        path.name: sha256(path)
        for path in (
            ROOT / "baseline/software/weights/H2_D1_seed42.npz",
            ROOT / "baseline/software/weights/S1P_INT8_ACC24_MEM24_Q12.npz",
        )
    }
    official_eval = ROOT / "official_reference/utils/eval.py"
    result = {
        "schema_version": 1,
        "status": "complete",
        "experiment_id": "s1-test-two-prefrozen-thresholds-v1",
        "stage": "final_test_evaluation",
        "data_role": "test",
        "test_used": True,
        "model": "s1_mixed",
        "threshold_selection": {
            "selected_before_test": True,
            "selection_from_test": False,
            "primary_public_code_working_point": 0.9,
            "secondary_frozen_deployment_working_point": THRESHOLDS[0][1],
        },
        "primary_metrics": {
            name: {
                "event_iou": value["event"]["iou"],
                "official_acc": value["event"]["official_acc"],
                "overall_accuracy": value["event"]["overall_accuracy"],
                "pd": value["localization"]["pd"],
                "fa": value["localization"]["fa"],
                "fa_times_1e4": value["localization"]["fa_times_1e4"],
            }
            for name, value in results.items()
        },
        "paper_reference_not_recomputed": {
            "model": "EV-SpSegNet",
            "split": "Test",
            "paper_threshold": None,
            "current_public_evaluator_default_threshold": 0.9,
            "iou_percent": 55.18,
            "official_acc_percent": 65.02,
            "pd_percent": 77.53,
            "fa_times_1e4": 1.63,
        },
        "protocol": {
            "id": PROTOCOL_ID,
            "official_commit": OFFICIAL_COMMIT,
            "aggregation": "micro counts summed across all 24 Test sequences before division",
            "event_support": "all original events; proposal-rejected events have probability 0",
            "pd_window_ms": 50,
            "pd_correct_threshold": 0.0001,
            "thresholds": [value for _, value in THRESHOLDS],
            "one_inference_two_rescorings": True,
            "reference_is_metric_code_not_official_network_inference": True,
        },
        "audit": {
            "sequence_count": len(test_records),
            "event_count": sum(record["events"] for record in test_records),
            "dataset_index_sha256": index_digest,
            "weight_sha256": weight_hashes,
            "official_eval_sha256": sha256(official_eval),
            "test_files": test_records,
            "predictions": prediction_manifest,
            "frozen_validation_audit": str(ROOT / "results/input_audit.json"),
        },
        "models": results,
        "runtime": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "torch_threads": 1,
            "total_wall_s": time.perf_counter() - started,
        },
    }
    save_json(output, result)

    print("\nS1 Test threshold comparison")
    print("=" * 86)
    print(f"{'working point':26s} {'IoU(%)':>10s} {'ACC/Recall(%)':>15s} "
          f"{'Overall Acc(%)':>15s} {'Pd(%)':>10s} {'Fa x1e4':>10s}")
    for name, _ in THRESHOLDS:
        metric = result["primary_metrics"][name]
        print(f"{name:26s} {100 * metric['event_iou']:10.4f} "
              f"{100 * metric['official_acc']:15.4f} "
              f"{100 * metric['overall_accuracy']:15.4f} "
              f"{100 * metric['pd']:10.4f} {metric['fa_times_1e4']:10.6f}")
    print("=" * 86)
    print("Official metric parity: PASS for both thresholds")
    print("Saved JSON:", output)
    print("Saved predictions:", predictions_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output = output.resolve()
    if ROOT not in output.parents or output.exists():
        raise ValueError("Output must be a new file inside validation_compare")
    run(output)
