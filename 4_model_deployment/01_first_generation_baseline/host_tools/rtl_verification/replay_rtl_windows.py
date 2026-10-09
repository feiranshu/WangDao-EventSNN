"""Bounded multiwindow Proposal replay, using frozen calibration inputs and CPU Refiner."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))
from host_tools.rtl_verification.generate_readout_vectors import write_records
from host_tools.rtl_verification.generate_rtl_vectors import DEFAULT_OUTPUT, INDEX_PATH, S1_WEIGHT_PATH, S3_WEIGHT_PATH
from host_tools.rtl_verification.integrate_rtl_proposal import H2_PATH, checked_simulation, prepare_window, save_json
from host_tools.rtl_verification.profile_rtl_sparsity import calibration_sources, profile_mode, read_fixture
from host_tools.rtl_verification.run_rtl_simulation import run as simulate, sha256, validate_manifest
from wangdao.data import load_events
from wangdao.h2.representation import build_temporal_patch_features, window_indices
from wangdao.rtl_bridge import assert_pipeline_equal, route_window

WINDOWS = (("train_093.npz", 70), ("train_093.npz", 71),
           ("train_022.npz", 0), ("train_058.npz", 66), ("train_093.npz", 70))
STREAMS = (("proposal_layer1", (346, 4, 48, 6), [16, 10, 10] * 2),
           ("proposal_layer2", (346, 4, 24, 6), [16, 10, 10] * 2),
           ("proposal_trace", (346, 4, 16, 8), [16, 16, 10, 10] * 2),
           ("proposal_expected", (346, 16, 6), [16, 10, 10] * 2))


def window_slices(windows: list[dict], patches: int) -> list[slice]:
    start, slices = 10, []
    for row in windows:
        if row["patch_start"] != start or row["patches"] != 336 or not 0 <= row["window"] < 80:
            raise ValueError("Noncontiguous or incomplete window mapping")
        slices.append(slice(start, start + 336))
        start += 336
    if not windows or start != patches:
        raise ValueError("Window map does not cover all real patches")
    return slices


def prepare(directory: Path, pilot: bool):
    if directory.exists():
        raise ValueError("Refusing to overwrite replay vectors")
    original, models, _, anchor_events, refiner, metadata = prepare_window(DEFAULT_OUTPUT)
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    sources = {p.name: (p, r) for p, r in calibration_sources(index, ROOT)}
    selected = WINDOWS[:1] if pilot else WINDOWS
    events_by_file = {}
    source_rows = []
    for name in dict.fromkeys(name for name, _ in selected):
        path, record = sources[name]
        if sha256(path) != record["sha256"]:
            raise ValueError("Calibration sequence hash mismatch")
        events_by_file[name] = anchor_events if name == "train_022.npz" else load_events(path, require_labels=True)
        source_rows.append({"file": path.relative_to(ROOT).as_posix(), "sha256": record["sha256"]})
    arrays = {name: [read_fixture(name + ".txt", shape, bases)[:10]] for name, shape, bases in STREAMS}
    inputs = [read_fixture("proposal_inputs.txt", (346, 4, 128), [16] * 512)[:10].astype(np.uint8)]
    cache, windows, offset = {}, [], 10
    for name, window in selected:
        key = (name, window)
        if key not in cache:
            events = events_by_file[name]
            features = build_temporal_patch_features(events, [window], 4)[0].transpose(1, 0, 2)
            traces = [profile_mode(features, model, k)[2] for model, k in zip(models, (0, 2))]
            combined = [np.concatenate((traces[0][i], traces[1][i]), axis=-1) for i in range(4)]
            cache[key] = (features, combined)
        features, combined = cache[key]
        inputs.append(features)
        for (stream, _, _), value in zip(STREAMS, combined):
            arrays[stream].append(value)
        windows.append({"file": name, "window": window, "patch_start": offset, "patches": 336,
                        "events": int(np.count_nonzero(window_indices(events_by_file[name]) == window))})
        offset += 336
    window_slices(windows, offset)
    directory.mkdir(parents=True)
    manifest = copy.deepcopy(original)
    for key, entry in original["files"].items():
        if not key.startswith("proposal_"):
            shutil.copyfile(DEFAULT_OUTPUT / entry["path"], directory / entry["path"])
    for entry in original["weight_files"].values():
        shutil.copyfile(DEFAULT_OUTPUT / entry["path"], directory / entry["path"])
    manifest["files"]["proposal_inputs"] = write_records(directory / "proposal_inputs.txt",
        np.concatenate(inputs).reshape(offset, 512), [16] * 512)
    for stream, _, bases in STREAMS:
        manifest["files"][stream] = write_records(directory / (stream + ".txt"), np.concatenate(arrays[stream]), bases)
    manifest.update(experiment_id="RTL-MULTIWINDOW-VECTORS-" + ("PILOT" if pilot else "v1"),
                    conclusion="Frozen full forward and layer traces checked per window; repeated windows share identical inputs.")
    manifest["configuration"].update(patches=offset, real_patches=offset - 10, window=None, windows=windows,
                                      external_reset_between_windows=False)
    manifest["primary_metrics"].update(cascade_patches=offset, logit_records=(134 + offset) * 16,
        readout_step_records=(134 + offset) * 64, full_forward_checked_logits_per_mode=offset * 16)
    manifest["source"] = {"dataset_role": "hw_calibration", "dataset_index_sha256": sha256(INDEX_PATH),
                           "exact_weight_sha256": sha256(S1_WEIGHT_PATH), "frozen_weight_sha256": sha256(S3_WEIGHT_PATH),
                           "h2_weight_sha256": sha256(H2_PATH), "sequences": source_rows,
                           "base_manifest_sha256": sha256(DEFAULT_OUTPUT / "proposal_manifest.json")}
    save_json(directory / "proposal_manifest.json", manifest)
    validate_manifest(directory, "proposal")
    return manifest, models, events_by_file, refiner, metadata, cache


def validate_outputs(manifest: dict, captures: dict, events: dict, refiner, metadata: dict, cache: dict) -> list[dict]:
    windows = manifest["configuration"]["windows"]
    slices = window_slices(windows, manifest["configuration"]["patches"])
    reference = captures["fixed"]
    for capture in captures.values():
        for field in ("logits_q", "tree_saturations", "membrane_saturations"):
            if not np.array_equal(getattr(capture, field), getattr(reference, field)):
                raise ValueError("Fixed/skip/stressed actual capture mismatch: " + field)
    rows = []
    for entry, part in zip(windows, slices):
        name, window = entry["file"], entry["window"]
        _, traces = cache[(name, window)]
        row = dict(entry, modes={})
        for mode, label in enumerate(("exact", "approximate")):
            golden = traces[-1][..., mode * 3].copy()
            golden[golden >= (1 << 31)] -= 1 << 32
            expected = route_window(events[name], window, golden, refiner,
                                    metadata["proposal_threshold"], metadata["event_threshold"], 257)
            for capture in captures.values():
                actual = route_window(events[name], window, capture.logits_q[mode, part], refiner,
                                      metadata["proposal_threshold"], metadata["event_threshold"], 4096)
                assert_pipeline_equal(actual, expected)
            row["modes"][label] = {"logits_checked": 5376, "events": len(expected["event_indices"]),
                                    "candidates": len(expected["candidates"]), "pipeline_mismatches": 0,
                                    "actual_logits_q": reference.logits_q[mode, part].tolist(),
                                    "refiner_saturations": list(expected["refiner_saturations"])}
        rows.append(row)
    if windows[0]["file"] != windows[-1]["file"] or windows[0]["window"] != windows[-1]["window"]:
        raise ValueError("Missing repeat-window isolation control")
    for capture in captures.values():
        if not np.array_equal(capture.logits_q[:, slices[0]], capture.logits_q[:, slices[-1]]):
            raise ValueError("Repeated window differs after intervening sequences")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("pilot", "formal"), required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-result", type=Path)
    parser.add_argument("--allow-long-run", action="store_true", help="For user-executed runs estimated above 5 minutes")
    parser.add_argument("--vivado-root", type=Path, default=Path("C:/Xilinx/Vivado/2019.1"))
    args = parser.parse_args()
    directory, output = args.run_directory.resolve(), args.output.resolve()
    if directory.exists() or not directory.is_relative_to(ROOT / "experiments/workspace"):
        parser.error("run-directory must be new and under experiments/workspace")
    pilot = args.mode == "pilot"
    estimate = None
    if not pilot:
        if args.pilot_result is None:
            parser.error("formal mode requires a successful pilot result")
        prior = json.loads(args.pilot_result.read_text(encoding="utf-8"))
        if prior.get("status") != "pilot" or prior.get("experiment_id") != "RTL-MULTIWINDOW-PILOT":
            parser.error("pilot did not pass")
        estimate = float(prior["estimated_formal_seconds"])
        if not np.isfinite(estimate) or estimate <= 0:
            parser.error("pilot estimate must be finite and positive")
        if estimate > 300 and not args.allow_long_run:
            parser.error(f"estimated {estimate:.1f}s; user should execute with --allow-long-run")
    output_root = (ROOT / "experiments/workspace" if pilot
                   else ROOT / "verification_evidence" / "host_integration")
    if output.exists() or not output.is_relative_to(output_root) or output.suffix != ".json":
        destination = "experiments/workspace" if pilot else "verification_evidence/host_integration"
        parser.error(f"output must be a new JSON under {destination}")
    started = time.monotonic()
    result = {"schema_version": 1, "status": "failed", "experiment_id": "RTL-MULTIWINDOW-" + ("PILOT" if pilot else "v1"),
              "stage": "P4-multiwindow-rtl-host-replay", "data_role": "hw_calibration_and_synthetic",
              "primary_metrics": {}, "decision": "replay_failed", "conclusion": "not completed",
              "test_used": False, "validation_used": False, "training_run": False, "device_verification": False,
              "source": {"command": sys.argv}, "simulations": {}}
    try:
        vectors = directory / "vectors"
        manifest, _, events, refiner, metadata, cache = prepare(vectors, pilot)
        result["source"].update(manifest["source"], vector_manifest_sha256=sha256(vectors / "proposal_manifest.json"))
        result["configuration"] = {"windows": manifest["configuration"]["windows"], "external_reset_between_windows": False,
                                   "proposal_threshold": metadata["proposal_threshold"], "event_threshold": metadata["event_threshold"],
                                   "h2_fixed_configuration": metadata["h2_fixed_configuration"]}
        result["prepare_seconds"] = time.monotonic() - started
        captures = {}
        variants = ("skip",) if pilot else ("fixed", "skip", "stressed")
        for variant in variants:
            path = directory / (variant + ".json")
            simulation = simulate(argparse.Namespace(vector_directory=vectors, suite="proposal", vivado_root=args.vivado_root,
                timeout_seconds=max(240, int(estimate or 240)) if args.allow_long_run else 240,
                experiment_id=result["experiment_id"] + "-" + variant.upper(), fault_checks=False,
                skip_zero_groups=variant != "fixed", measure_cycles=variant != "stressed", proposal_only=True,
                fixed_reference=directory / "fixed.json" if variant == "skip" and not pilot else None))
            save_json(path, simulation)
            result["simulations"][variant] = {"path": str(path.relative_to(ROOT)), "sha256": sha256(path),
                                               "elapsed_seconds": simulation["simulation"]["elapsed_seconds"]}
            captures[variant], _ = checked_simulation(simulation, vectors)
            print({"variant": variant, "status": simulation["status"], "elapsed_seconds": simulation["simulation"]["elapsed_seconds"]}, flush=True)
        if pilot:
            times = simulation["simulation"]["command_seconds"]
            sim_seconds = times["tb_proposal_engine_simulate.log"]
            overhead = simulation["simulation"]["elapsed_seconds"] - sim_seconds
            # Scale even XSim startup as simulation work, assume fixed schedule and add 25% margin.
            ratio = (10 + len(WINDOWS) * 336) * 9601 / simulation["cycle_measurements"]["measured_total_cycles"]
            result["estimated_formal_seconds"] = 1.25 * (3 * (overhead + sim_seconds * ratio) + result["prepare_seconds"] * len(WINDOWS) + 10)
            real_cycles = simulation["cycle_measurements"]["rows"][10:]
            mean = sum(row["measured_cycles"] - 1 for row in real_cycles) / 336
            result["primary_metrics"] = {"complete_windows": 1, "real_patches_per_mode": 336,
                "logits_checked_per_mode": 5376, "mean_proposal_core_cycles": mean,
                "core_cycle_reduction": 1 - mean / 9600, "host_pipeline_checked": False}
            result.update(status="pilot", decision="ready_for_formal_replay", conclusion="Dense-window pilot passed; full repeat-window study not yet run.")
        else:
            result["windows"] = validate_outputs(manifest, captures, events, refiner, metadata, cache)
            optimized = json.loads((directory / "skip.json").read_text(encoding="utf-8"))
            cycles = optimized["cycle_measurements"]["rows"]
            for row in result["windows"]:
                subset = cycles[row["patch_start"]:row["patch_start"] + 336]
                mean = sum(r["measured_cycles"] - 1 for r in subset) / 336
                row["mean_proposal_core_cycles"] = mean
                row["core_cycle_reduction"] = 1 - mean / 9600
            result["primary_metrics"] = {"unique_windows": len(cache), "window_transfers": len(WINDOWS),
                "real_patches_per_mode_per_schedule": len(WINDOWS) * 336, "pipeline_mismatches": 0,
                "repeat_window_identical": True, "stressed_outputs_identical": True}
            result.update(status="ok", decision="multiwindow_alignment_passed",
                          conclusion="Fixed/skip/stressed RTL and A1-H2 align on selected calibration windows, including a repeated window without external reset; no target-device verification.")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, AssertionError) as error:
        result["error"] = str(error)
    result["elapsed_seconds"] = time.monotonic() - started
    save_json(output, result)
    print(json.dumps({key: result.get(key) for key in ("status", "primary_metrics", "elapsed_seconds", "estimated_formal_seconds", "error")}))
    raise SystemExit(0 if result["status"] in ("ok", "pilot") else 1)


if __name__ == "__main__":
    main()
