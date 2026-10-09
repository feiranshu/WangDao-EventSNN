"""Replay actual RTL Proposal transfers through frozen A1-H2 on calibration window 0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))

from host_tools.rtl_verification.generate_rtl_vectors import (
    INDEX_PATH, S1_WEIGHT_PATH, S3_WEIGHT_PATH, frozen_first_layer, hardware_calibration_path,
)
from host_tools.rtl_verification.run_rtl_simulation import parse_success, run as simulate, sha256, validate_manifest
from wangdao.data import load_events
from wangdao.fixed import FixedH2, FixedH2Config, load_approximate_proposal_snn, load_fixed_proposal_snn
from wangdao.h2.numpy_model import load_export
from wangdao.metrics import event_metrics, localization_metrics
from wangdao.rtl_bridge import assert_pipeline_equal, read_proposal_capture, route_window
from wangdao.s0p import build_s0p_sequence

H2_PATH = ROOT / "model_runtime/weights/H2_D1_seed42.npz"


def save_json(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes((json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    temporary.replace(path)


def prepare_window(directory: Path) -> tuple:
    manifest, _ = validate_manifest(directory, "proposal")
    config, source = manifest["configuration"], manifest["source"]
    required = {"window": 0, "patches": 346, "real_patches": 336, "complete_window": True,
                "synthetic_cascade_patches": 10, "time_steps": 4, "channels": [48, 24, 16]}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Integration requires the complete calibration window 0 after 10 synthetic patches")
    if (source.get("dataset_role") != "hw_calibration"
            or source.get("dataset_sequence") != "dataset/EV-UAV-dataset/train/train_022.npz"):
        raise ValueError("Integration only accepts the official HW-calibration sequence")
    path = hardware_calibration_path()
    required_source = {"dataset_role": "hw_calibration", "dataset_sequence": path.relative_to(ROOT).as_posix(),
                       "dataset_index_sha256": sha256(INDEX_PATH), "dataset_sequence_sha256": sha256(path),
                       "exact_weight_sha256": sha256(S1_WEIGHT_PATH), "frozen_weight_sha256": sha256(S3_WEIGHT_PATH)}
    if any(source.get(key) != value for key, value in required_source.items()):
        raise ValueError("Calibration vector provenance mismatch")
    models = frozen_first_layer()
    _, _, exact_meta = load_fixed_proposal_snn(S1_WEIGHT_PATH)
    _, _, approx_meta = load_approximate_proposal_snn(S3_WEIGHT_PATH)
    for key in ("h2_fixed_configuration", "proposal_threshold", "event_threshold"):
        if exact_meta[key] != approx_meta[key]:
            raise ValueError("S1/S3 host configuration differs")
    if approx_meta["h2_weight_sha256"] != sha256(H2_PATH):
        raise ValueError("Frozen H2 weight identity mismatch")
    # Validate ROM semantics, not just agreement with a possibly edited manifest.
    for layer in range(3):
        if (not np.array_equal(models[0].weights[layer], models[1].weights[layer])
                or not np.array_equal(models[0].weight_fraction_bits[layer], models[1].weight_fraction_bits[layer])):
            raise ValueError("Exact/approximate frozen weights or scales differ")
        for bank in range(8):
            words = [int(word, 16) for word in (directory / f"layer{layer + 1}_bank{bank}.txt").read_text().split()]
            signed = np.asarray([word - 256 if word >= 128 else word for word in words])
            if not np.array_equal(signed, models[0].weights[layer][:, bank::8].reshape(-1)):
                raise ValueError("ROM differs from frozen model")
        shifts = [int(word, 16) for word in (directory / f"layer{layer + 1}_shifts.txt").read_text().split()]
        expected_shifts = (models[0].weight_fraction_bits[layer] - 4 if layer == 0
                           else 12 - models[0].weight_fraction_bits[layer])
        if not np.array_equal(shifts, expected_shifts):
            raise ValueError("ROM channel scaling mismatch")
    sequence = build_s0p_sequence(path, 4, windows=(0,))
    features = sequence.temporal_features[0].transpose(1, 0, 2)
    words = (directory / "proposal_inputs.txt").read_text(encoding="ascii").split()
    inputs = np.asarray([int(word, 16) for word in words], dtype=np.uint8).reshape(346, 4, 128)
    if not np.array_equal(inputs[10:], features):
        raise ValueError("Simulated features do not match all 336 calibration patches in order")
    refiner = FixedH2(load_export(H2_PATH), FixedH2Config(**approx_meta["h2_fixed_configuration"]))
    return manifest, models, features.transpose(1, 0, 2), load_events(path, require_labels=True), refiner, approx_meta


def checked_simulation(result: dict, directory: Path):
    manifest, files = validate_manifest(directory, "proposal")
    count = manifest["configuration"]["patches"] * 16
    if (result.get("status") != "ok" or result.get("decision") != "behavioral_alignment_passed"
            or result["source"]["vector_manifest_sha256"] != sha256(directory / "proposal_manifest.json")):
        raise ValueError("Missing successful simulation for this manifest")
    artifact = result["actual_output"]
    if (artifact.get("origin") != "DUT_ready_valid_transfers" or artifact.get("format") != "WD_CAPTURE_v1"
            or artifact.get("records_per_mode") != count):
        raise ValueError("Invalid actual-output provenance")
    work = ROOT / result["simulation"]["work_directory"]
    # Verify the inputs actually copied into XSim's work directory and its success log.
    for source in files:
        if sha256(work / source.name) != sha256(source):
            raise ValueError("Simulation input copy differs from manifest")
    parsed = parse_success((work / "tb_proposal_engine_simulate.log").read_text(encoding="utf-8"),
                           "tb_proposal_engine", count)
    if parsed not in result["simulation"]["tests"]:
        raise ValueError("Simulation log differs from recorded checks")
    path = ROOT / artifact["path"]
    if path.resolve() != (work / "proposal_actual_passed.txt").resolve() or sha256(path) != artifact["sha256"]:
        raise ValueError("Actual output capture identity mismatch")
    return read_proposal_capture(path, count // 16), parsed


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1, "status": "failed", "experiment_id": args.experiment_id,
        "stage": "P4-rtl-python-window-integration", "data_role": "hw_calibration",
        "primary_metrics": {}, "decision": "integration_failed",
        "conclusion": "RTL/host integration did not complete.",
        "test_used": False, "validation_used": False, "hardware_mainline_changed": False,
        "scope": {"sequence": "train_022.npz", "windows": [0], "patches": 336,
                  "rtl": "three-layer Proposal", "host": "Python candidate routing and frozen A1-H2 Refiner",
                  "device_verification": False, "training": False, "threshold_scan": False},
    }
    try:
        manifest, models, features, events, refiner, metadata = prepare_window(args.vector_directory)
        result["timing_seconds"] = {"prepare": round(time.monotonic() - started, 3)}
        if args.simulation_result is None:
            if args.simulation_output.exists():
                raise ValueError("Refusing to overwrite simulation output; use a new path or --simulation-result")
            simulation = simulate(argparse.Namespace(
                vector_directory=args.vector_directory, suite="proposal", vivado_root=args.vivado_root,
                timeout_seconds=args.timeout_seconds, experiment_id=args.experiment_id + "-XSIM", fault_checks=False))
            save_json(args.simulation_output, simulation)
            simulation_path = args.simulation_output
        else:
            simulation_path = args.simulation_result
            simulation = json.loads(simulation_path.read_text(encoding="utf-8"))
        capture, bench = checked_simulation(simulation, args.vector_directory)
        result["source"] = {**manifest["source"], "h2_weight_sha256": sha256(H2_PATH),
                            "simulation_result": str(simulation_path), "simulation_result_sha256": sha256(simulation_path),
                            "actual_output": simulation["actual_output"], "command": sys.argv}
        result["configuration"] = {"proposal_threshold": metadata["proposal_threshold"],
                                   "event_threshold": metadata["event_threshold"],
                                   "h2_fixed_configuration": metadata["h2_fixed_configuration"],
                                   "refiner_batch_size": args.batch_size, "golden_refiner_batch_size": 257,
                                   "excluded_event_probability": 0.0}
        result["simulation"] = {"tool": simulation["simulation"], "proposal_checks": bench}
        modes = {}
        for index, (name, model) in enumerate(zip(("exact", "approximate"), models)):
            golden_started = time.monotonic()
            golden = model.forward(features)
            golden_seconds = time.monotonic() - golden_started
            actual_q = capture.logits_q[index, 10:]
            if not np.array_equal(actual_q, golden.logits_q):
                raise ValueError(f"{name}: actual Proposal logits differ from Python golden")
            tree = int(capture.tree_saturations[index, 10:].sum())
            membrane = int(capture.membrane_saturations[index, 10:].sum())
            if (tree, membrane) != (golden.accumulator_saturations[2], golden.membrane_saturations[2]):
                raise ValueError(f"{name}: captured readout saturation mismatch")
            host_started = time.monotonic()
            actual = route_window(events, 0, actual_q, refiner, metadata["proposal_threshold"],
                                  metadata["event_threshold"], args.batch_size)
            host_seconds = time.monotonic() - host_started
            expected = route_window(events, 0, golden.logits_q, refiner, metadata["proposal_threshold"],
                                    metadata["event_threshold"], 257)
            assert_pipeline_equal(actual, expected)
            members, candidates = np.asarray(actual["event_indices"]), np.asarray(actual["candidates"])
            window_events = events[members]
            labels = window_events["label"].astype(bool)
            probability = np.asarray(actual["probabilities"])
            target_count = int(labels.sum())
            candidate_targets = int(events["label"][candidates].sum())
            modes[name] = {
                "status": "passed", "logits_checked": int(actual_q.size), "pipeline_mismatches": 0,
                "checked_fields": list(expected), "events": len(members), "target_events": target_count,
                "candidates": len(candidates), "candidate_ratio": len(candidates) / len(members),
                "proposal_recall": candidate_targets / target_count if target_count else 1.0,
                "excluded_target_events": target_count - candidate_targets,
                "event": event_metrics(labels, probability, metadata["event_threshold"]),
                "localization": localization_metrics(window_events["t"], window_events["x"], window_events["y"],
                    window_events["label"], window_events["name"], probability, metadata["event_threshold"]),
                "readout_tree_saturations": tree, "readout_membrane_saturations": membrane,
                "refiner_saturations": list(actual["refiner_saturations"]),
                "python_golden_seconds": round(golden_seconds, 3), "actual_host_seconds": round(host_seconds, 3),
                # Retain the accepted DUT words in the formal JSON even if tmp is cleaned.
                "actual_logits_q": actual_q.tolist(),
            }
            print(f"{name}: {actual_q.size} RTL logits, {len(candidates)} candidates, {len(members)} events aligned", flush=True)
        result.update(status="ok", decision="single_window_rtl_host_alignment_passed",
                      conclusion="Actual RTL Proposal outputs match Python through routing and A1-H2 on 1 calibration window; no target-device verification.")
        result["primary_metrics"] = {"windows": 1, "patches_per_mode": 336, "logits_per_mode": 5376,
                                     "pipeline_mismatches": 0, "modes_passed": 2}
        result["modes"] = modes
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as error:
        result["error"] = str(error)
    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-directory", type=Path, default=ROOT / "pl_design/test_vectors")
    parser.add_argument("--vivado-root", type=Path, default=Path("C:/Xilinx/Vivado/2019.1"))
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--simulation-result", type=Path, help="Reuse and verify an existing successful capture and work directory")
    parser.add_argument("--simulation-output", type=Path, default=ROOT / "verification_evidence/pl_simulation/rtl_host_simulation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "verification_evidence/host_integration/rtl_host_integration.json")
    parser.add_argument("--experiment-id", default="RTL-HOST-WINDOW-v1")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing integration result")
    if args.batch_size < 1 or not 1 <= args.timeout_seconds <= 300:
        parser.error("batch-size must be positive and timeout-seconds must be in [1, 300]")
    targets = [args.output.resolve(), args.simulation_output.resolve()]
    if len(set(targets)) != 2 or (args.simulation_result and args.output.resolve() == args.simulation_result.resolve()):
        parser.error("Integration and simulation result paths must differ")
    result = run(args)
    save_json(args.output, result)
    print(json.dumps({key: result.get(key) for key in ("status", "primary_metrics", "elapsed_seconds", "error")}))
    raise SystemExit(0 if result["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
