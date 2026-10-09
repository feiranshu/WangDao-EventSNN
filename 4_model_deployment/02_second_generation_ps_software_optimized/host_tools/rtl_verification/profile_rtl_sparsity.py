"""Estimate zero-group bypass opportunity using frozen S1/S3; never change or run RTL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))
from host_tools.rtl_verification.generate_readout_vectors import readout_trace
from host_tools.rtl_verification.generate_rtl_vectors import (DEFAULT_OUTPUT, INDEX_PATH, S1_WEIGHT_PATH, S3_WEIGHT_PATH,
                                         engine_trace, frozen_first_layer, second_layer_trace, sha256_file)
from host_tools.rtl_verification.run_rtl_simulation import validate_manifest
from wangdao.data import load_events
from wangdao.h2.representation import build_temporal_patch_features, window_indices
from wangdao.rtl_sparsity import (BENCH_BOUNDARY_CYCLES, COMPUTE_CYCLES, PATCH_CYCLES,
                                 cycle_contract, group_activity, recommend, summarize_cycles)


def calibration_sources(index: dict, root: Path) -> list[tuple[Path, dict]]:
    names = index.get("roles", {}).get("hw_calibration", [])
    if index.get("status") != "ok" or not names or len(names) != len(set(names)):
        raise ValueError("valid nonempty HW-calibration role required")
    if any(set(names) & set(v) for k, v in index["roles"].items() if k != "hw_calibration"):
        raise ValueError("HW-calibration overlaps another data role")
    records = {Path(r["path"]).name: r for r in index["files"]}
    official = root / "dataset/EV-UAV-dataset"
    official_resolved = official.resolve()
    location = Path(index["location"])
    if location.parts and location.parts[0] == "data":
        location = Path("dataset", *location.parts[1:])
    if (root / location).resolve() != official_resolved:
        raise ValueError("only official repository dataset allowed")
    result = []
    for name in names:
        record = records[name]
        path = official / record["path"]
        resolved = path.resolve()
        if not resolved.is_relative_to(official_resolved) or path.name != name:
            raise ValueError("dataset path escapes official root")
        result.append((path, record))
    return result


def choose_windows(counts: np.ndarray, anchor: bool) -> dict[int, list[str]]:
    if counts.shape != (80,) or np.any(counts < 0):
        raise ValueError("expected counts for all 80 frozen 100-ms windows")
    order = np.lexsort((np.arange(80), counts))
    selected: dict[int, list[str]] = {}
    for label, index in (("minimum_events", 0), ("median_events", 40), ("maximum_events", 79)):
        selected.setdefault(int(order[index]), []).append(label)
    if anchor:
        selected.setdefault(0, []).append("historical_rtl_anchor")
    return selected


def historical_cycles() -> dict:
    expected = {"tb_first_layer_engine": 512 + COMPUTE_CYCLES[0],
                "tb_second_layer_engine": 192 + COMPUTE_CYCLES[1],
                "tb_two_layer_engine": 512 + sum(COMPUTE_CYCLES[:2]),
                "tb_readout_layer_engine": 96 + COMPUTE_CYCLES[2], "tb_proposal_engine": PATCH_CYCLES}
    rows = {}
    for name in ("rtl_first_layer_simulation.json", "rtl_two_layer_simulation.json", "rtl_proposal_simulation.json"):
        path = ROOT / "verification_evidence/inherited_first_gen_pl_baseline" / "pl_simulation" / name
        result = json.loads(path.read_text(encoding="utf-8"))
        if result["status"] != "ok":
            raise ValueError("historical simulation did not pass")
        if name == "rtl_proposal_simulation.json" and result["source"]["vector_manifest_sha256"] != sha256_file(DEFAULT_OUTPUT / "proposal_manifest.json"):
            raise ValueError("current Proposal vectors differ from historical XSim inputs")
        for test in result["simulation"]["tests"]:
            bench = test["testbench"]
            if bench not in expected:
                continue
            predicted = expected[bench] + BENCH_BOUNDARY_CYCLES
            if test["status"] != "passed" or test["no_stall_patch_cycles"] != predicted:
                raise ValueError("cycle budget disagrees with historical XSim")
            rows[bench] = {"core_cycles": expected[bench], "historical_xsim_cycles": predicted,
                           "source": f"verification_evidence/pl_simulation/{name}", "source_sha256": sha256_file(path)}
    if set(rows) != set(expected):
        raise ValueError("incomplete historical cycle evidence")
    return rows


def profile_mode(features: np.ndarray, model, k: int) -> tuple[dict, np.ndarray, list[np.ndarray]]:
    first = engine_trace(features, model, k)
    first_spikes = ((first[..., 0] >> 24) & 1).astype(np.uint8)
    second = second_layer_trace(first_spikes, model, k)
    second_spikes = ((second[..., 0] >> 24) & 1).astype(np.uint8)
    third, final = readout_trace(second_spikes, model, k)
    public = model.forward(features.transpose(1, 0, 2))
    q = final[..., 0].copy()
    q[q >= 1 << 31] -= 1 << 32
    trees = [int(t[..., c].sum()) for t, c in ((first, 1), (second, 1), (third, 2))]
    membranes = [int(t[..., c].sum()) for t, c in ((first, 2), (second, 2), (third, 3))]
    if not np.array_equal(q, public.logits_q) or trees != list(public.accumulator_saturations) or membranes != list(public.membrane_saturations[:3]):
        raise ValueError("layer traces disagree with complete frozen forward")
    layers, empty = {}, []
    for layer, (values, trace) in enumerate(zip((features, first_spikes, second_spikes), (first, second, third))):
        activity, counts = group_activity(values, layer)
        encoded = trace[..., 0] & 0xFFFFFF
        membrane = np.where(encoded >= 1 << 23, encoded - (1 << 24), encoded)
        previous = np.concatenate((np.zeros_like(membrane[:, :1]), membrane[:, :-1]), axis=1)
        zero_step = np.all(values == 0, axis=-1)
        activity.update(zero_input_steps=int(zero_step.sum()),
                        zero_input_nonzero_previous_membranes=int(np.count_nonzero(previous[zero_step])),
                        zero_input_changed_membranes=int(np.count_nonzero((membrane != previous)[zero_step])),
                        zero_input_output_spikes=int(((trace[..., 0] >> 24) & 1)[zero_step].sum()) if layer < 2 else 0,
                        tree_saturations=trees[layer], membrane_saturations=membranes[layer])
        layers[f"layer{layer + 1}"] = activity
        empty.append(counts)
    empty_counts = np.stack(empty, axis=1)
    return {"layers": layers, "empty_groups_per_patch_layer": empty_counts.tolist(),
            "cycle_budget": summarize_cycles(empty_counts), "golden_forward_checked_logits": int(q.size),
            "hidden_spikes": [int(first_spikes.sum()), int(second_spikes.sum())]}, empty_counts, [first, second, third, final]


def read_fixture(name: str, shape: tuple, bases: list[int]) -> np.ndarray:
    rows = [[int(v, bases[i]) for i, v in enumerate(line.split())]
            for line in (DEFAULT_OUTPUT / name).read_text(encoding="ascii").splitlines()]
    return np.asarray(rows, dtype=np.int64).reshape(shape)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("anchor", "representative"), default="representative")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() or output.suffix != ".json" or not output.is_relative_to(ROOT / "verification_evidence" / "pl_simulation"):
        raise ValueError("output must be a new JSON under results")
    if args.scope == "anchor" and not output.is_relative_to(ROOT / "experiments/workspace"):
        raise ValueError("anchor pilot belongs in ignored experiments/workspace")
    started = time.perf_counter()
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    sources = calibration_sources(index, ROOT)
    if args.scope == "anchor":
        sources = [r for r in sources if r[0].name == "train_022.npz"]
    models = frozen_first_layer()
    for layer in range(3):
        if not np.array_equal(models[0].weights[layer], models[1].weights[layer]) or not np.array_equal(models[0].weight_fraction_bits[layer], models[1].weight_fraction_bits[layer]):
            raise ValueError("S1/S3 weights or channel scales differ")
    manifest, _ = validate_manifest(DEFAULT_OUTPUT, "proposal")
    expected_configuration = {"patches": 346, "real_patches": 336, "window": 0,
                              "synthetic_cascade_patches": 10, "time_steps": 4,
                              "channels": [48, 24, 16], "lanes": 8}
    if any(manifest["configuration"].get(key) != value for key, value in expected_configuration.items()):
        raise ValueError("historical anchor layout differs from frozen fixture contract")
    for key, path in (("dataset_index_sha256", INDEX_PATH), ("exact_weight_sha256", S1_WEIGHT_PATH),
                      ("frozen_weight_sha256", S3_WEIGHT_PATH)):
        if manifest["source"][key] != sha256_file(path):
            raise ValueError("historical anchor provenance differs from current frozen sources")
    anchor_source = next((record for path, record in sources if path.name == "train_022.npz"), None)
    if anchor_source is None or manifest["source"]["dataset_sequence_sha256"] != anchor_source["sha256"]:
        raise ValueError("historical anchor is not the indexed calibration source")
    cycle_evidence = historical_cycles()
    fixture_input = read_fixture("proposal_inputs.txt", (346, 4, 128), [16] * 512).astype(np.uint8)
    fixtures = [read_fixture("proposal_layer1.txt", (346, 4, 48, 6), [16, 10, 10] * 2),
                read_fixture("proposal_layer2.txt", (346, 4, 24, 6), [16, 10, 10] * 2),
                read_fixture("proposal_trace.txt", (346, 4, 16, 8), [16, 16, 10, 10] * 2),
                read_fixture("proposal_expected.txt", (346, 16, 6), [16, 10, 10] * 2)]
    result = {"schema_version": 1, "status": "running", "experiment_id": "RTL-SPARSITY-" + args.scope.upper() + "-v1",
              "stage": "P4-static-zero-group-cycle-assessment", "data_role": "hw_calibration",
              "primary_metrics": {}, "decision": {}, "conclusion": "pending",
              "configuration": {"scope": args.scope, "cycle_contract": cycle_contract(),
                                "window_selection": "per calibration sequence: event-count min, upper median, max; ties by window index; plus train_022 window0",
                                "engineering_screen_mean_reduction": .05,
                                "non_regression_margin": "one extra cycle per output/time step on selected layer",
                                "labels_used_for_selection": False, "simulation_run": False, "rtl_modified": False},
              "source": {"index_sha256": sha256_file(INDEX_PATH), "s1_sha256": sha256_file(S1_WEIGHT_PATH),
                         "s3_sha256": sha256_file(S3_WEIGHT_PATH), "vector_manifest_sha256": sha256_file(DEFAULT_OUTPUT / "proposal_manifest.json"),
                         "sequences": []}, "historical_cycle_checks": cycle_evidence,
              "windows": [], "timing_seconds": {"read_and_hash": 0., "representation": 0., "trace_and_profile": 0.},
              "validation_used": False, "test_used": False, "training_run": False}
    totals: dict[str, list[np.ndarray]] = {"exact": [], "approximate": []}
    for path, metadata in sources:
        mark = time.perf_counter()
        if sha256_file(path) != metadata["sha256"]:
            raise ValueError("calibration source hash mismatch")
        events = load_events(path, require_labels=True)
        counts = np.asarray(np.bincount(window_indices(events), minlength=80), dtype=np.int64)
        selected = {0: ["historical_rtl_anchor"]} if args.scope == "anchor" else choose_windows(counts, path.name == "train_022.npz")
        result["source"]["sequences"].append({"file": path.relative_to(ROOT).as_posix(), "sha256": metadata["sha256"],
                                               "events": len(events), "window_event_counts": counts.tolist()})
        result["timing_seconds"]["read_and_hash"] += time.perf_counter() - mark
        mark = time.perf_counter()
        windows = sorted(selected)
        features = build_temporal_patch_features(events, windows, 4).transpose(0, 2, 1, 3)
        result["timing_seconds"]["representation"] += time.perf_counter() - mark
        for w, value in zip(windows, features):
            row = {"file": path.name, "window": w, "selection": selected[w], "events": int(counts[w]), "modes": {}}
            mark = time.perf_counter()
            anchor = path.name == "train_022.npz" and w == 0
            if anchor and not np.array_equal(value, fixture_input[10:]):
                raise ValueError("anchor features differ from previously simulated fixture")
            for mode, (name, model) in enumerate(zip(("exact", "approximate"), models)):
                profile, empty_counts, traces = profile_mode(value, model, mode * 2)
                row["modes"][name] = profile
                totals[name].append(empty_counts)
                if anchor:
                    for trace, fixture in zip(traces, fixtures):
                        width = trace.shape[-1]
                        if not np.array_equal(trace, fixture[10:, ..., mode * width:(mode + 1) * width]):
                            raise ValueError("anchor trace differs from previously simulated golden records")
                    row["historical_vector_alignment"] = "all layer words, readout sums, logits and saturation counts"
            result["timing_seconds"]["trace_and_profile"] += time.perf_counter() - mark
            result["windows"].append(row)
            print({"file": path.name, "window": w, "events": int(counts[w]), "profiled": True}, flush=True)
    complete = args.scope == "representative"
    result["primary_metrics"] = {"sequences": len(sources), "windows": len(result["windows"]),
                                  "patches_per_mode": len(result["windows"]) * 336,
                                  "cycles_by_mode": {name: summarize_cycles(np.concatenate(parts)) for name, parts in totals.items()}}
    activities = {}
    for mode in totals:
        activities[mode] = {}
        for layer in ("layer1", "layer2", "layer3"):
            rows = [w["modes"][mode]["layers"][layer] for w in result["windows"]]
            aggregate: dict = {key: sum(row[key] for row in rows) for key in (
                "input_values", "nonzero_inputs", "unique_input_groups", "empty_input_groups",
                "scheduled_group_visits", "skippable_group_sum_visits", "scheduled_lane_products_or_selects",
                "nonzero_input_lane_products_or_selects", "zero_input_steps", "zero_input_nonzero_previous_membranes",
                "zero_input_changed_membranes", "zero_input_output_spikes", "tree_saturations", "membrane_saturations")}
            aggregate["input_nonzero_fraction"] = aggregate["nonzero_inputs"] / aggregate["input_values"]
            aggregate["empty_group_fraction"] = aggregate["empty_input_groups"] / aggregate["unique_input_groups"]
            aggregate["group_active_lane_histogram"] = np.asarray([r["group_active_lane_histogram"] for r in rows]).sum(axis=0).tolist()
            activities[mode][layer] = aggregate
    result["primary_metrics"]["activity_by_mode"] = activities
    result["primary_metrics"]["golden_forward_checked_logits_per_mode"] = sum(
        w["modes"]["exact"]["golden_forward_checked_logits"] for w in result["windows"])
    result["decision"] = recommend(result["windows"], complete)
    result["status"] = "ok" if complete else "pilot"
    result["conclusion"] = result["decision"]["next"] + "; static budget only; RTL equivalence, implementation overhead and FPGA benefit remain unverified"
    result["elapsed_seconds"] = time.perf_counter() - started
    result["estimated_representative_seconds"] = result["elapsed_seconds"] * 31 if not complete else None
    result["configuration"]["historical_manifest_experiment"] = manifest["experiment_id"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes((json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8"))
    print({"status": result["status"], "output": str(output), "elapsed_seconds": result["elapsed_seconds"],
           "estimated_representative_seconds": result["estimated_representative_seconds"], "decision": result["decision"]}, flush=True)


if __name__ == "__main__":
    main()
