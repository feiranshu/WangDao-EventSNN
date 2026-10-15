"""Frozen ANN/r2-T8 interface smoke checks on declared clips, never training."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
ANN = ROOT.parents[1] / "1_H2_model_improvement/implementation"
sys.path.insert(0, str(ANN / "software"))
import wangdao
wangdao.__path__.append(str(ROOT / "software/wangdao"))
from wangdao.data import load_events
from wangdao.h2_pro.stream_state import CausalPipeline, EVENT_DTYPE
from wangdao.h2pro_snn.foundation import verify_baseline
from wangdao.h2pro_direct_v1.core import forward, probabilities, validate_layers
from wangdao.fred_adapter_v1.download import atomic_json, sha256


def run(dataset, student, output):
    dataset, student, output = Path(dataset), Path(student), Path(output)
    baseline = verify_baseline()
    student_hash = sha256(student)
    if output.exists():
        previous = json.loads(output.read_text("utf-8"))
        if (previous.get("status") == "passed" and
                previous.get("dataset_index_sha256") == sha256(dataset / "index.json") and
                previous.get("student_sha256") == student_hash and
                previous.get("baseline_lock_sha256") == baseline["baseline_lock_sha256"]):
            print(json.dumps({"stage": "reuse_verified_interface_report", "report": str(output)}), flush=True)
            return previous
        raise ValueError("Existing model report has another identity; preserve it and choose a new --report path")
    with np.load(student, allow_pickle=False) as z:
        if str(z["stage"]) != "r2" or int(z["steps"]) != 8 or float(z["threshold"]) != .35:
            raise ValueError("Expected frozen exported r2/T8 student at threshold .35")
        layers = tuple((z[f"layer{i}_weight"].copy(), z[f"layer{i}_scale"].copy()) for i in range(3))
    validate_layers(layers)
    for pair in layers:
        for array in pair: array.setflags(write=False)
    index = json.loads((dataset / "index.json").read_text("utf-8"))
    rows = []
    for sequence in index["sequences"]:
        t0 = sequence["source_first_timestamp_us"] / 1000.0
        clips = [("dense_opening", t0, t0 + 100),
                 ("cross_shard_boundary", 9900.0, 10200.0),
                 ("uav_scene", 15000.0, 16000.0)]
        for purpose, begin, end in clips:
            started = time.perf_counter()
            pipe = CausalPipeline(ANN / "software/weights/H2_D1_seed42.npz",
                ANN / "software/weights/H2-pro-Causal-ANN-v1_int8.npz",
                variant="C1-proposal-motion", threshold=.35, initial_refiner=False)
            pipe.reset(begin)
            windows, counts, input_count, valid_count = [], [], 0, 0
            input_paths = []
            for row in sequence["shards"]:
                if row["last_timestamp_ms"] < begin or row["first_timestamp_ms"] >= end:
                    continue
                source = load_events(dataset / row["path"], require_labels=True)
                lo, hi = np.searchsorted(source["t"], [begin, end], side="left")
                selected = source[lo:hi]
                with np.load(dataset / row["path"], allow_pickle=False) as z:
                    valid_count += int(z["eval_mask"][lo:hi].sum())
                input_count += len(selected)
                input_paths.append(row["path"])
                # Labels and eval_mask are deliberately never passed to the model.
                packet = np.empty(len(selected), dtype=EVENT_DTYPE)
                for field in ("x", "y", "t", "p"): packet[field] = selected[field]
                pipe.push(packet, min(end, row["shard_end_ms"]))
                windows.extend(pipe.flush_completed_windows())
            closed, trailing = pipe.finalize(end)
            windows.extend(closed)
            candidate_count = 0
            explicit_checked = 0
            ann_positive, snn_positive = 0, 0
            for window in windows:
                x = window.features
                if x.shape != (int(window.candidate_mask.sum()), 36) or x.dtype != np.uint8:
                    raise AssertionError("Original feature interface changed")
                if not np.isfinite(window.probabilities).all():
                    raise FloatingPointError("Nonfinite frozen ANN probabilities")
                ann_positive += int(window.predictions.sum())
                candidate_count += len(x)
                for start in range(0, len(x), 32768):
                    raw, cost = forward(x[start:start + 32768], layers, "r2", 8, "count_reference")
                    p = probabilities(raw)
                    if not np.isfinite(p).all():
                        raise FloatingPointError("Nonfinite frozen SNN probabilities")
                    snn_positive += int((p >= .35).sum())
                if len(x) and explicit_checked == 0:
                    probe = x[:128]
                    dense, _ = forward(probe, layers, "r2", 8, "count_reference")
                    explicit, _ = forward(probe, layers, "r2", 8, "spike_add")
                    np.testing.assert_array_equal(dense, explicit)
                    explicit_checked = len(probe)
            covered = sum(len(w.event_indices) for w in windows)
            if covered + trailing != input_count:
                raise AssertionError("Stream coverage mismatch")
            if covered:
                np.testing.assert_array_equal(np.concatenate([w.event_indices for w in windows]), np.arange(covered))
            row = {"sequence": sequence["source_sequence"], "purpose": purpose,
                "begin_ms": begin, "end_ms": end, "input_events": input_count,
                "closed_events": covered, "trailing_events": trailing,
                "closed_windows": len(windows), "candidate_events": candidate_count,
                "eval_mask_events": valid_count, "source_shards": input_paths,
                "ann_predicted_positive": ann_positive, "snn_predicted_positive": snn_positive,
                "explicit_vs_count_reference_events_checked": explicit_checked,
                "probabilities_finite": True, "seconds": time.perf_counter() - started}
            rows.append(row)
            print(json.dumps({"stage": "frozen_interface_check", **row}), flush=True)
    if sha256(student) != student_hash or verify_baseline() != baseline:
        raise AssertionError("Frozen model changed during check")
    report = {"status": "passed", "dataset_index_sha256": sha256(dataset / "index.json"),
        "baseline_lock_sha256": baseline["baseline_lock_sha256"], "student_sha256": student_hash,
        "student": str(student.absolute()), "training_started": False,
        "scope": "Declared short clips only; state reset per clip, preserved across dataset shards",
        "formal_metrics_reported": False, "backend": "ANN exact-int8; SNN dense count_reference, explicit subset parity",
        "checks": rows}
    atomic_json(output, report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--student", type=Path, default=ROOT / "results/runs/h2pro_distill_r2t8_v1_s42/best_int8.npz")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    run(args.dataset, args.student, args.report)
