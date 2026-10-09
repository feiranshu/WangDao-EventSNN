"""Run bounded Vivado behavioral simulations with checked vector manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))
UNIT_TESTS = ("fixed_add", "fixed_add_tree", "lif_neuron", "linear_lif_neuron")
MANIFEST_NAMES = {"units": "manifest.json", "engine": "first_layer_manifest.json",
                  "two_layers": "two_layer_manifest.json", "proposal": "proposal_manifest.json"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_file(directory: Path, entry: dict[str, Any]) -> Path:
    name = Path(entry["path"]).name
    path = directory / name
    if not path.is_file() or sha256(path) != entry["sha256"]:
        raise ValueError(f"Vector/weight hash mismatch: {name}")
    return path


def check_rows(path: Path, count: int, schema: list[tuple[int, int]]) -> None:
    """Schema entries are (base, maximum); reject X/Z, overflow and extra fields."""
    if count <= 0:
        raise ValueError(f"Expected positive record count: {path.name}")
    rows = path.read_text(encoding="ascii").splitlines()
    if len(rows) != count:
        raise ValueError(f"Record count mismatch: {path.name}: {len(rows)} != {count}")
    for number, line in enumerate(rows, 1):
        fields = line.split()
        if len(fields) != len(schema):
            raise ValueError(f"Malformed record: {path.name}:{number}")
        for value, (base, maximum) in zip(fields, schema):
            pattern = r"[0-9a-fA-F]+" if base == 16 else r"[0-9]+"
            if not re.fullmatch(pattern, value) or int(value, base) > maximum:
                raise ValueError(f"Invalid field: {path.name}:{number}: {value}")


def validate_manifest(directory: Path, suite: str) -> tuple[dict[str, Any], list[Path]]:
    manifest_path = directory / MANIFEST_NAMES[suite]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "ok":
        raise ValueError("Vector manifest is not ready")
    files: list[Path] = []
    hex24, flag = (16, 0xFFFFFF), (10, 1)
    if suite == "units":
        schemas = {
            "fixed_add": [hex24, hex24, hex24, flag, hex24, flag],
            "fixed_add_tree": [hex24] * 6 + [(10, 4), hex24, (10, 4)],
            "lif_neuron": [hex24, hex24, flag, hex24, flag, flag, hex24, flag, flag],
            "linear_lif_neuron": [(16, 255)] * 128
                + [hex24, hex24, hex24, flag, (10, 127), flag,
                   hex24, flag, (10, 127), flag] * 3,
        }
        for name in UNIT_TESTS:
            entry = manifest["files"][name]
            path = checked_file(directory, entry)
            check_rows(path, int(entry["records"]), schemas[name])
            files.append(path)
        for name in ("layer1_weight_f4", "layer1_weight_f5", "layer1_weight_f6"):
            entry = manifest["weight_files"][name]
            path = checked_file(directory, entry)
            if int(entry["values"]) != 128:
                raise ValueError("Wrong first-layer weight count")
            check_rows(path, 128, [(16, 255)])
            files.append(path)
    elif suite == "proposal":
        schemas = {}
        for prefix, patch_key, width, maximum in (("readout", "standalone_patches", 96, 1),
                                                ("proposal", "patches", 512, 255)):
            patches = int(manifest["configuration"][patch_key])
            if not 2 <= patches <= (2048 if prefix == "proposal" else 346):
                raise ValueError("Unsupported Proposal patch count")
            schemas[prefix + "_inputs"] = (patches, [(16, maximum)] * width)
            schemas[prefix + "_expected"] = (patches * 16, [(16, 0xFFFFFFFF), (10, 92), (10, 4)] * 2)
            schemas[prefix + "_trace"] = (patches * 64, [(16, 0xFFFFFF), (16, 0x3FFFFFF), (10, 23), flag] * 2)
        patches = int(manifest["configuration"]["patches"])
        schemas["proposal_layer1"] = (patches * 192, [(16, 0x1FFFFFF), (10, 127), flag] * 2)
        schemas["proposal_layer2"] = (patches * 96, [(16, 0x1FFFFFF), (10, 47), flag] * 2)
        schemas["readout_rounding"] = (23, [(16, 0x3FFFFFF), (16, 0xFFFFFFFF)])
        for name, (count, schema) in schemas.items():
            entry = manifest["files"][name]
            if int(entry["records"]) != count:
                raise ValueError("Inconsistent Proposal manifest count")
            path = checked_file(directory, entry)
            check_rows(path, count, schema)
            files.append(path)
        for layer, count, channels, maximum in ((1, 768, 48, 2), (2, 144, 24, 8), (3, 48, 16, 7)):
            weights = [(f"layer{layer}_bank{bank}", count, 255) for bank in range(8)]
            weights.append((f"layer{layer}_shifts", channels, maximum))
            for name, count, maximum in weights:
                entry = manifest["weight_files"][name]
                if int(entry["values"]) != count:
                    raise ValueError("Inconsistent Proposal ROM count")
                path = checked_file(directory, entry)
                check_rows(path, count, [(16, maximum)])
                files.append(path)
    elif suite == "two_layers":
        schemas = {}
        for prefix, patch_key, width, maximum in (("second_layer", "standalone_patches", 192, 1),
                                                ("two_layer", "patches", 512, 255)):
            patches = int(manifest["configuration"][patch_key])
            if not 2 <= patches <= 346:
                raise ValueError("Unsupported two-layer patch count")
            schemas[prefix + "_inputs"] = (patches, [(16, maximum)] * width)
            schemas[prefix + "_expected"] = (patches * 96, [(16, 0x1FFFFFF), (10, 47), flag] * 2)
        schemas["two_layer_intermediate"] = (int(manifest["configuration"]["patches"]) * 192,
                                               [(16, 0x1FFFFFF), (10, 127), flag] * 2)
        for name, (count, schema) in schemas.items():
            entry = manifest["files"][name]
            if int(entry["records"]) != count:
                raise ValueError("Inconsistent two-layer manifest count")
            path = checked_file(directory, entry)
            check_rows(path, count, schema)
            files.append(path)
        for layer, count, channels, maximum in ((1, 768, 48, 2), (2, 144, 24, 8)):
            weights = [(f"layer{layer}_bank{bank}", count, 255) for bank in range(8)]
            weights.append((f"layer{layer}_shifts", channels, maximum))
            for name, count, maximum in weights:
                entry = manifest["weight_files"][name]
                if int(entry["values"]) != count:
                    raise ValueError("Inconsistent two-layer ROM count")
                path = checked_file(directory, entry)
                check_rows(path, count, [(16, maximum)])
                files.append(path)
    else:
        patches = int(manifest["configuration"]["patches"])
        schemas = {
            "first_layer_inputs": (patches, [(16, 255)] * 512),
            "first_layer_expected": (patches * 192,
                [(16, 0x1FFFFFF), (10, 127), flag] * 2),
        }
        for name, (count, schema) in schemas.items():
            entry = manifest["files"][name]
            if int(entry["records"]) != count:
                raise ValueError("Inconsistent engine manifest count")
            path = checked_file(directory, entry)
            check_rows(path, count, schema)
            files.append(path)
        for bank in range(8):
            entry = manifest["weight_files"][f"layer1_bank{bank}"]
            path = checked_file(directory, entry)
            check_rows(path, 768, [(16, 255)])
            files.append(path)
        path = checked_file(directory, manifest["weight_files"]["layer1_shifts"])
        check_rows(path, 48, [(16, 2)])
        files.append(path)
    return manifest, files


def run_command(command: list[str], work: Path, log_name: str, deadline: float) -> str:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Simulation deadline exceeded")
    with (work / log_name).open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=work, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            process.wait(timeout=10)
            raise TimeoutError(f"Timed out: {log_name}") from error
    output = (work / log_name).read_text(encoding="utf-8", errors="replace")
    if code != 0:
        raise RuntimeError(f"Tool exit {code}: {log_name}\n{output[-3000:]}")
    return output


def parse_success(output: str, top: str, expected: int) -> dict[str, int | str]:
    matches = re.findall(r"WD_PASS top=(\w+) records=(\d+)", output)
    if matches != [(top, str(expected))] or re.search(r"\b(Fatal|FATAL|ERROR):", output):
        raise RuntimeError(f"Missing/incorrect success record for {top}\n{output[-2000:]}")
    result: dict[str, int | str] = {"testbench": top, "records": int(matches[0][1]), "status": "passed"}
    for key, value in re.findall(r"WD_STAT (\w+)=(\d+)", output):
        result[key] = int(value)
    return result


def check_cycle_measurements(output: str, inputs: Path, top: str, patches: int, skip: bool) -> dict:
    """Compare every measured no-stall transaction against its actual eight-lane inputs."""
    import numpy as np
    from wangdao.rtl_sparsity import group_activity
    if top not in ("tb_first_layer_engine", "tb_proposal_engine"):
        raise ValueError("Cycle measurement requires first-layer or Proposal bench")
    values = np.asarray([[int(v, 16) for v in row.split()] for row in inputs.read_text(encoding="ascii").splitlines()],
                        dtype=np.int64).reshape(patches, 4, 128)
    _, empty = group_activity(values, 0)
    matches = re.findall(r"^WD_CYCLE patch=(\d+) cycles=(\d+)\s*$", output, re.MULTILINE)
    if (len(matches) != patches or len(re.findall(r"^WD_CYCLE\b", output, re.MULTILINE)) != patches
            or [int(p) for p, _ in matches] != list(range(patches))):
        raise ValueError("Missing, duplicate or unordered patch cycle measurements")
    baseline = 7425 if top == "tb_first_layer_engine" else 9601
    rows = []
    for (patch, actual), zero in zip(matches, empty):
        expected = baseline - (int(zero) * 48 if skip else 0)
        if int(actual) != expected:
            raise ValueError(f"Patch {patch} measured {actual} cycles, expected {expected}")
        rows.append({"patch": int(patch), "zero_groups": int(zero), "measured_cycles": int(actual),
                     "fixed_cycles": baseline, "saved_cycles": baseline - int(actual)})
    total = sum(row["measured_cycles"] for row in rows)
    return {"testbench": top, "patches": patches, "boundary_cycles_per_patch": 1,
            "fixed_cycles_per_patch": baseline, "measured_total_cycles": total,
            "mean_measured_cycles": total / patches,
            "cycle_reduction": 1 - total / (patches * baseline),
            "mismatches": 0, "rows": rows}


def compare_fixed_proposal(reference: Path, directory: Path, current: dict) -> dict:
    import numpy as np
    from host_tools.rtl_verification.integrate_rtl_proposal import checked_simulation
    from wangdao.rtl_bridge import read_proposal_capture
    fixed = json.loads(reference.read_text(encoding="utf-8"))
    if fixed["simulation"].get("first_layer_skip_zero_groups") is not False or fixed["simulation"].get("measure_cycles") is not True:
        raise ValueError("Reference must measure the fixed Proposal schedule")
    capture, _ = checked_simulation(fixed, directory)
    manifest, _ = validate_manifest(directory, "proposal")
    patches = int(manifest["configuration"]["patches"])
    work = ROOT / fixed["simulation"]["work_directory"]
    rechecked = check_cycle_measurements((work / "tb_proposal_engine_simulate.log").read_text(encoding="utf-8"),
        work / "proposal_inputs.txt", "tb_proposal_engine", patches, False)
    if rechecked != fixed["cycle_measurements"]:
        raise ValueError("Reference cycles disagree with simulation log")
    actual_path = ROOT / current["actual_output"]["path"]
    if sha256(actual_path) != current["actual_output"]["sha256"]:
        raise ValueError("Optimized capture hash mismatch")
    actual = read_proposal_capture(actual_path, patches)
    for field in ("logits_q", "tree_saturations", "membrane_saturations"):
        if not np.array_equal(getattr(capture, field), getattr(actual, field)):
            raise ValueError(f"Fixed/skip RTL captures differ: {field}")
    start = int(manifest["configuration"]["synthetic_cascade_patches"])
    rows = current["cycle_measurements"]["rows"]
    scopes = {}
    for name, subset in (("all_patches", rows), ("hw_calibration", rows[start:])):
        if not subset:
            continue
        fixed_core = sum(r["fixed_cycles"] - 1 for r in subset)
        measured_core = sum(r["measured_cycles"] - 1 for r in subset)
        scopes[name] = {"patches": len(subset), "fixed_core_cycles_per_patch": 9600,
                        "mean_measured_core_cycles": measured_core / len(subset),
                        "saved_cycles": fixed_core - measured_core,
                        "core_cycle_reduction": 1 - measured_core / fixed_core}
    return {"reference": str(reference), "reference_sha256": sha256(reference), "mismatches": 0,
            "checked_logits_per_mode": patches * 16, "scopes": scopes,
            "scope": "measured behavioral cycles; no clock, power or device claim"}


def exercise_failures(binary: Path, work: Path, manifest: dict[str, Any], deadline: float) -> list[dict[str, str]]:
    """Reuse compiled unit snapshots to test the testbenches, bypassing preflight."""
    completed = []
    expected_columns = {"fixed_add": 2, "fixed_add_tree": 5, "lif_neuron": 3, "linear_lif_neuron": 130}
    for name in UNIT_TESTS:
        vector = work / (name + ".txt")
        original = vector.read_text(encoding="ascii")
        rows = original.splitlines()
        damaged = rows[0].split()
        damaged[expected_columns[name]] = f"{int(damaged[expected_columns[name]], 16) ^ 1:X}"
        cases = {
            "empty": "", "truncated": "\n".join(rows[:-1]) + "\n",
            "malformed": "INVALID " + original,
            "wrong_expected": " ".join(damaged) + "\n" + "\n".join(rows[1:]) + "\n",
            "extra_record": original + rows[0] + "\n",
        }
        (work / "expected_records.txt").write_text(str(manifest["files"][name]["records"]) + "\n", encoding="ascii")
        try:
            for case, content in cases.items():
                vector.write_text(content, encoding="ascii")
                log_name = f"fault_{name}_{case}.log"
                try:
                    output = run_command([str(binary / "xsim.bat"), f"tb_{name}_snapshot", "-runall"], work, log_name, deadline)
                except RuntimeError:
                    output = (work / log_name).read_text(encoding="utf-8", errors="replace")
                if not re.search(r"\b(Fatal|FATAL):", output) or "WD_PASS" in output:
                    raise RuntimeError(f"Fault did not cause an explicit testbench failure: {name}/{case}")
                completed.append({"testbench": "tb_" + name, "fault": case, "status": "rejected"})
        finally:
            vector.write_text(original, encoding="ascii")
    return completed


def exercise_two_layer_failures(binary: Path, work: Path, manifest: dict[str, Any], deadline: float,
                               suite: str = "two_layers") -> list[dict[str, str]]:
    """Corrupt only temporary copies and require an explicit failure from each bench."""
    completed = []
    for prefix in (("readout", "proposal") if suite == "proposal" else ("second_layer", "two_layer")):
        top = "tb_readout_layer_engine" if prefix == "readout" else "tb_" + prefix + "_engine"
        input_path, expected_path = work / (prefix + "_inputs.txt"), work / (prefix + "_expected.txt")
        inputs, expected = input_path.read_text(encoding="ascii"), expected_path.read_text(encoding="ascii")
        rows = expected.splitlines()
        wrong = rows[0].split()
        wrong[0] = f"{int(wrong[0], 16) ^ 1:08X}"
        cases = [
            ("empty_input", input_path, ""),
            ("truncated_input", input_path, "\n".join(inputs.splitlines()[:-1]) + "\n"),
            ("malformed_input", input_path, "INVALID " + inputs),
            ("wrong_expected", expected_path, " ".join(wrong) + "\n" + "\n".join(rows[1:]) + "\n"),
            ("extra_expected", expected_path, expected + rows[0] + "\n"),
        ]
        if prefix == "two_layer":
            cases.append(("truncated_intermediate", work / "two_layer_intermediate.txt", ""))
        if suite == "proposal":
            cases.append(("truncated_readout_trace", work / (prefix + "_trace.txt"), ""))
        if prefix == "proposal":
            cases.extend((f"truncated_layer{layer}", work / f"proposal_layer{layer}.txt", "") for layer in (1, 2))
        count = int(manifest["files"][prefix + "_expected"]["records"])
        (work / "expected_records.txt").write_text(str(count) + "\n", encoding="ascii")
        for case, path, damaged in cases:
            original = path.read_bytes()
            log_name = f"fault_{prefix}_{case}.log"
            try:
                path.write_text(damaged, encoding="ascii")
                try:
                    output = run_command([str(binary / "xsim.bat"), top + "_snapshot", "-runall"], work, log_name, deadline)
                except RuntimeError:
                    output = (work / log_name).read_text(encoding="utf-8", errors="replace")
                if not re.search(r"\b(Fatal|FATAL):", output) or "WD_PASS" in output:
                    raise RuntimeError(f"Fault did not explicitly fail: {prefix}/{case}")
                completed.append({"testbench": top, "fault": case, "status": "rejected"})
            finally:
                path.write_bytes(original)
    return completed


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": 1, "status": "failed", "experiment_id": args.experiment_id,
        "stage": "P4-rtl-behavioral-simulation", "data_role": "synthetic_and_hw_calibration_golden_vectors",
        "primary_metrics": {}, "decision": "simulation_failed", "conclusion": "Behavioral verification did not complete.",
        "test_used": False, "validation_used": False,
        "simulation": {"tool": "Vivado Simulator", "version": args.vivado_root.name,
                       "scope": "behavioral_simulation_only", "device_target": None, "board_target": None},
    }
    try:
        skip = bool(getattr(args, "skip_zero_groups", False))
        measure = bool(getattr(args, "measure_cycles", False))
        if (skip or measure) and args.suite not in ("engine", "proposal"):
            raise ValueError("Zero-group options support engine and proposal suites only")
        reference = getattr(args, "fixed_reference", None)
        if reference and (not skip or not measure or args.suite != "proposal"):
            raise ValueError("fixed-reference requires measured zero-skip Proposal simulation")
        result["simulation"]["first_layer_skip_zero_groups"] = skip
        result["simulation"]["measure_cycles"] = measure
        result["simulation"]["proposal_only"] = bool(getattr(args, "proposal_only", False))
        if result["simulation"]["proposal_only"] and args.fault_checks:
            raise ValueError("proposal-only does not compile the standalone fault-check snapshots")
        manifest, files = validate_manifest(args.vector_directory, args.suite)
        manifest_name = MANIFEST_NAMES[args.suite]
        result["source"] = {"vector_manifest": str(args.vector_directory / manifest_name),
                            "vector_manifest_sha256": sha256(args.vector_directory / manifest_name),
                            "command": sys.argv, "manifest_source": manifest.get("source", {})}
        temporary_root = ROOT / "experiments" / "workspace"
        temporary_root.mkdir(exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix="rtl_sim_", dir=temporary_root))
        result["simulation"]["work_directory"] = str(work.relative_to(ROOT))
        for path in files:
            shutil.copyfile(path, work / path.name)
        tops = ["tb_" + name for name in UNIT_TESTS] if args.suite == "units" else ["tb_first_layer_engine"]
        if args.suite == "two_layers":
            tops = ["tb_second_layer_engine", "tb_two_layer_engine"]
        if args.suite == "proposal":
            tops = ["tb_readout_rounding", "tb_readout_layer_engine", "tb_proposal_engine"]
        if getattr(args, "proposal_only", False):
            if args.suite != "proposal":
                raise ValueError("proposal-only requires proposal suite")
            tops = ["tb_proposal_engine"]
        sources = sorted((ROOT / "pl_design" / "rtl").glob("*.sv"))
        sources += ([ROOT / "pl_design/testbenches/tb_proposal_engine.sv"] if args.suite == "proposal"
                    else [ROOT / "pl_design/testbenches/tb_two_layer_engine.sv"] if args.suite == "two_layers"
                    else [ROOT / "pl_design" / "testbenches" / (top + ".sv") for top in tops])
        deadline = started + args.timeout_seconds
        binary = args.vivado_root / "bin"
        command_seconds = {}
        result["simulation"]["command_seconds"] = command_seconds
        def timed_command(command: list[str], log: str) -> str:
            mark = time.monotonic()
            try:
                return run_command(command, work, log, deadline)
            finally:
                command_seconds[log] = time.monotonic() - mark
        # Preserve the deep hierarchical readout observation points in XSim 2019.1.
        elaboration_options = ["-debug", "all"] if args.suite == "proposal" else []
        if elaboration_options:
            result["simulation"]["elaboration_options"] = elaboration_options
        defines = (["WD_ZERO_SKIP"] if skip else []) + (["WD_MEASURE_CYCLES"] if measure else [])
        result["simulation"]["compile_defines"] = defines
        define_options = [option for name in defines for option in ("-d", name)]
        timed_command([str(binary / "xvlog.bat"), "--sv", *define_options, *map(str, sources)], "compile.log")
        completed = []
        for top in tops:
            count = (int(manifest["files"][top[3:]]["records"]) if args.suite == "units"
                     else int(manifest["configuration"]["patches"]) * 192)
            if args.suite == "two_layers":
                count = int(manifest["files"][top[3:].replace("_engine", "_expected")]["records"])
            if args.suite == "proposal":
                key = {"tb_readout_rounding": "readout_rounding", "tb_readout_layer_engine": "readout_expected",
                       "tb_proposal_engine": "proposal_expected"}[top]
                count = int(manifest["files"][key]["records"])
            (work / "expected_records.txt").write_text(str(count) + "\n", encoding="ascii")
            snapshot = top + "_snapshot"
            timed_command([str(binary / "xelab.bat"), top, *elaboration_options, "-s", snapshot], top + "_elaborate.log")
            output = timed_command([str(binary / "xsim.bat"), snapshot, "-runall"], top + "_simulate.log")
            completed.append(parse_success(output, top, count))
            if top in ("tb_first_layer_engine", "tb_proposal_engine"):
                if completed[-1].get("skip_zero_groups") != int(skip) or completed[-1].get("measured_cycle_patches") != (count // (192 if args.suite == "engine" else 16) if measure else 0):
                    raise ValueError("Bench did not confirm requested zero-group/cycle options")
                if measure:
                    key = "first_layer_inputs" if args.suite == "engine" else "proposal_inputs"
                    result["cycle_measurements"] = check_cycle_measurements(output, work / manifest["files"][key]["path"],
                        top, int(manifest["configuration"]["patches"]), skip)
            if top == "tb_proposal_engine":
                from wangdao.rtl_bridge import read_proposal_capture
                capture = work / "proposal_actual.txt"
                read_proposal_capture(capture, count // 16)
                # Fault replays reopen the bench's output; preserve the successful run first.
                saved = work / "proposal_actual_passed.txt"
                shutil.copyfile(capture, saved)
                result["actual_output"] = {"path": str(saved.relative_to(ROOT)),
                                           "sha256": sha256(saved), "records_per_mode": count,
                                           "format": "WD_CAPTURE_v1", "origin": "DUT_ready_valid_transfers"}
            print(f"{top}: {count} records passed", flush=True)
        if reference:
            result["fixed_schedule_comparison"] = compare_fixed_proposal(reference, args.vector_directory, result)
        if args.fault_checks:
            result["fault_checks"] = (exercise_two_layer_failures(binary, work, manifest, deadline, args.suite)
                                      if args.suite in ("two_layers", "proposal")
                                      else exercise_failures(binary, work, manifest, deadline))
        total = sum(int(item["records"]) for item in completed)
        result.update(status="ok", decision="behavioral_alignment_passed",
                      conclusion="RTL matches checked Python golden vectors; no target-device verification performed.")
        result["primary_metrics"] = {"testbenches": len(completed), "passed_records": total, "failed_records": 0}
        result["simulation"]["tests"] = completed
        if args.suite != "units":
            result["coverage"] = manifest["configuration"]
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, TimeoutError) as error:
        result["error"] = str(error)
    result["simulation"]["elapsed_seconds"] = round(time.monotonic() - started, 6)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vivado-root", type=Path, default=Path("C:/Xilinx/Vivado/2019.1"))
    parser.add_argument("--vector-directory", type=Path, default=ROOT / "pl_design" / "test_vectors")
    parser.add_argument("--suite", choices=tuple(MANIFEST_NAMES), default="units")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--experiment-id")
    parser.add_argument("--fault-checks", action="store_true")
    parser.add_argument("--skip-zero-groups", action="store_true", help="Compile first-layer zero-group bypass; default fixed schedule")
    parser.add_argument("--measure-cycles", action="store_true", help="Measure every regular patch without stalls; reset replays still stress flow control")
    parser.add_argument("--fixed-reference", type=Path, help="Verify actual Proposal captures and measured cycles against a fixed-schedule result")
    parser.add_argument("--proposal-only", action="store_true", help="Replay full Proposal only; retain prior standalone arithmetic evidence")
    args = parser.parse_args()
    if args.timeout_seconds <= 0:
        parser.error("timeout-seconds must be positive")
    if args.fault_checks and args.suite == "engine":
        parser.error("fault-checks supports the units, two_layers and proposal suites")
    if (args.skip_zero_groups or args.measure_cycles) and args.suite not in ("engine", "proposal"):
        parser.error("zero-group/cycle options require engine or proposal suite")
    if (args.skip_zero_groups or args.measure_cycles) and (not args.output or not args.experiment_id):
        parser.error("zero-group/cycle runs require explicit output and experiment-id")
    args.vector_directory = args.vector_directory.resolve()
    args.experiment_id = args.experiment_id or ("RTL-UNIT-REGRESSION-X2019-v3" if args.suite == "units"
                                               else "RTL-PROPOSAL-X2019-v1" if args.suite == "proposal"
                                               else "RTL-TWO-LAYER-X2019-v1" if args.suite == "two_layers"
                                               else "RTL-FIRST-LAYER-X2019-v1")
    output_names = {"units": "rtl_unit_regression.json", "engine": "rtl_first_layer_simulation.json",
                    "two_layers": "rtl_two_layer_simulation.json", "proposal": "rtl_proposal_simulation.json"}
    output = args.output or ROOT / "verification_evidence" / "pl_simulation" / output_names[args.suite]
    if output.exists():
        parser.error(f"Refusing to overwrite existing result: {output}")
    result = run(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(json.dumps({"status": result["status"], "output": str(output), "error": result.get("error")}, ensure_ascii=True))
    if result["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
