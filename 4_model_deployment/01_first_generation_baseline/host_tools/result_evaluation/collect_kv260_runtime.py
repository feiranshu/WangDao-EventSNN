"""Aggregate KV260 per-sequence timing, implementation area and optional power data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IMPLEMENTATIONS = (
    ROOT / "verification_evidence" / "implementation" / "kv260_synthesis_implementation.json",
    ROOT / "verification_evidence" / "inherited_first_gen_pl_baseline" / "implementation" / "kv260_synthesis_implementation.json",
)
DEFAULT_BITSTREAM = ROOT / "kv260_deployment" / "firmware" / "artifacts" / "wangdao_pl_flat.bit.bin"
DEFAULT_DTBO = ROOT / "kv260_deployment" / "firmware" / "artifacts" / "wangdao_pl_flat.dtbo"
DEFAULT_SHELL = ROOT / "kv260_deployment" / "firmware" / "artifacts" / "shell.json"
DEFAULT_OUTPUT = ROOT / "verification_evidence" / "runtime_measurements" / "KV260-RUNTIME-EXACT-v2" / "runtime.json"

SUMMARY_FIELDS = (
    "sd_or_page_cache_read_ns",
    "ps_pack_and_axi_write_ns",
    "storage_to_pl_observed_ns",
    "pl_start_to_done_observed_ns",
    "axi_output_read_ns",
    "page_cache_write_ns",
    "patch_total_to_page_cache_ns",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent,
        mode="w", encoding="utf-8", newline="\n", delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def median_int(values: Iterable[int]) -> int:
    return int(statistics.median(list(values)))


def default_implementation() -> Path:
    for candidate in DEFAULT_IMPLEMENTATIONS:
        if candidate.is_file():
            return candidate
    return DEFAULT_IMPLEMENTATIONS[0]


def aggregate_summary(reports: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    summaries = [report["timing"][field] for report in reports]
    count = sum(int(summary["count"]) for summary in summaries)
    total = sum(int(summary["total"]) for summary in summaries)
    return {
        "count": count,
        "total": total,
        "weighted_mean": total / count if count else 0.0,
        "minimum": min(int(summary["min"]) for summary in summaries),
        "maximum": max(int(summary["max"]) for summary in summaries),
        "median_of_sequence_medians": median_int(int(summary["median"]) for summary in summaries),
        "maximum_sequence_p95": max(int(summary["p95"]) for summary in summaries),
        "note": "Cross-sequence median/P95 require raw samples; reported values summarize per-sequence distributions.",
    }


def normalized_power(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {
            "status": "not_measured",
            "measurement_scope": None,
            "tool": None,
            "sample_rate_hz": None,
            "idle_power_w": None,
            "active_power_w": None,
            "dynamic_power_w": None,
            "pl_power_w": None,
            "pl_power_is_directly_measured": False,
            "active_energy_j": None,
            "dynamic_energy_j": None,
            "active_energy_per_task_j": None,
            "dynamic_energy_per_task_j": None,
            "active_energy_per_patch_j": None,
            "dynamic_energy_per_patch_j": None,
            "limitation": "VCC_SOM telemetry is whole-SOM power; provide a power JSON or leave PL power null.",
        }
    value = load_json(path)
    for field in ("idle_power_w", "active_power_w"):
        measurement = value.get(field)
        if not isinstance(measurement, dict) or float(measurement.get("mean", 0.0)) <= 0.0:
            raise ValueError(f"{field}.mean must contain a positive measured value; placeholders are not accepted")
    direct = bool(value.get("pl_power_is_directly_measured", False))
    if not direct and value.get("pl_power_w") is not None:
        raise ValueError("pl_power_w must be null unless pl_power_is_directly_measured is true")
    value["source_file"] = str(path)
    value["source_sha256"] = sha256_file(path)
    value.setdefault("status", "ok")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path,
                        help="directory containing <sequence>.runtime.json files")
    parser.add_argument("--implementation", type=Path, default=default_implementation())
    parser.add_argument("--power-json", type=Path,
                        help="optional measured SOM/PL power summary; see README")
    parser.add_argument("--bitstream", type=Path, default=DEFAULT_BITSTREAM)
    parser.add_argument("--dtbo", type=Path, default=DEFAULT_DTBO)
    parser.add_argument("--shell", type=Path, default=DEFAULT_SHELL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--experiment-id", default="KV260-RUNTIME-EXACT-v2")
    parser.add_argument("--generation", default=ROOT.name,
                        help="generation label written to the result (default: deployment directory name)")
    arguments = parser.parse_args()

    manifest_path = arguments.manifest.resolve()
    manifest = load_json(manifest_path)
    if manifest.get("status") != "ok" or manifest.get("schema_version") != 1:
        raise ValueError("manifest is not a successful KV260 batch manifest")
    sequences = manifest.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("manifest contains no sequences")
    runtime_dir = (arguments.runtime_dir.resolve() if arguments.runtime_dir
                   else manifest_path.parent / "runtime")

    reports: List[Dict[str, Any]] = []
    per_sequence: List[Dict[str, Any]] = []
    full_model_times: List[int] = []
    for record in sequences:
        name = str(record["name"])
        stem = Path(name).stem
        path = runtime_dir / f"{stem}.runtime.json"
        if not path.is_file():
            path = runtime_dir / f"{stem}.pl.runtime.json"
        report = load_json(path)
        if report.get("status") != "ok":
            raise ValueError(f"runtime report is not successful: {path}")
        if int(report.get("patches", -1)) != int(record["patches"]):
            raise ValueError(f"runtime patch count differs from manifest: {name}")
        timing = report.get("timing")
        if not isinstance(timing, dict) or any(field not in timing for field in SUMMARY_FIELDS):
            raise ValueError(f"runtime report lacks segmented timing: {path}")
        reports.append(report)
        sequence_result: Dict[str, Any] = {
            "name": name,
            "role": record.get("role"),
            "patches": int(record["patches"]),
            "pl_runtime_file": str(path),
            "pl_runtime_sha256": sha256_file(path),
            "pl_timing": timing,
        }
        h2_path = runtime_dir / f"{stem}.h2.runtime.json"
        full_path = runtime_dir / f"{stem}.full.runtime.json"
        if h2_path.is_file():
            h2_report = load_json(h2_path)
            if h2_report.get("status") != "ok":
                raise ValueError(f"H2 runtime report is not successful: {h2_path}")
            sequence_result["h2_runtime_file"] = str(h2_path)
            sequence_result["h2_runtime_sha256"] = sha256_file(h2_path)
            sequence_result["h2_timing"] = h2_report.get("timing")
            sequence_result["h2_complexity"] = h2_report.get("B_model_complexity")
        if full_path.is_file():
            full_report = load_json(full_path)
            if full_report.get("status") != "ok" or "full_model_total_ns" not in full_report:
                raise ValueError(f"full-model runtime report is invalid: {full_path}")
            full_time = int(full_report["full_model_total_ns"])
            if full_time <= 0:
                raise ValueError(f"full-model runtime must be positive: {full_path}")
            full_model_times.append(full_time)
            sequence_result["full_runtime_file"] = str(full_path)
            sequence_result["full_runtime_sha256"] = sha256_file(full_path)
            sequence_result["full_model_total_ns"] = full_time
        per_sequence.append(sequence_result)

    implementation_path = arguments.implementation.resolve()
    implementation = load_json(implementation_path)
    if implementation.get("status") != "complete":
        raise ValueError("implementation result is not complete")
    primary = implementation.get("primary_metrics")
    if not isinstance(primary, dict):
        raise ValueError("implementation result has no primary_metrics")
    implementation_configuration = implementation.get("configuration", {})
    area = {
        "source": str(implementation_path),
        "source_sha256": sha256_file(implementation_path),
        "clock_mhz": primary.get("clock_mhz"),
        "clb_luts": primary.get("clb_luts"),
        "clb_lut_utilization_percent": primary.get("clb_lut_utilization_percent"),
        "clb_registers": primary.get("clb_registers"),
        "clb_register_utilization_percent": primary.get("clb_register_utilization_percent"),
        "block_ram_tiles": primary.get("block_ram_tiles"),
        "dsps": primary.get("dsps"),
        "wns_ns": primary.get("wns_ns"),
        "tns_ns": primary.get("tns_ns"),
        "whs_ns": primary.get("whs_ns"),
        "ths_ns": primary.get("ths_ns"),
        "vivado_vectorless_total_on_chip_power_w": primary.get("vectorless_total_on_chip_power_w"),
        "vivado_power_confidence": primary.get("vectorless_power_confidence"),
        "tool": implementation_configuration.get("tool"),
        "device": implementation_configuration.get("device"),
        "board_part": implementation_configuration.get("board_part"),
        "note": "Area/timing are imported from implementation; vectorless power is not board power.",
    }

    detected_power = arguments.power_json.resolve() if arguments.power_json else runtime_dir / "power_summary.json"
    power = normalized_power(detected_power if detected_power.is_file() else None)
    batch_total_ns = sum(int(report["timing"]["batch_total_ns"]) for report in reports)
    patches = sum(int(report["patches"]) for report in reports)
    aggregate = {field: aggregate_summary(reports, field) for field in SUMMARY_FIELDS}
    aggregate.update({
        "capture_flush_fsync_ns_total": sum(
            int(report["timing"]["capture_flush_fsync_ns"]) for report in reports
        ),
        "pl_to_storage_component_total_ns": sum(
            int(report["timing"]["pl_to_storage_component_total_ns"]) for report in reports
        ),
        "batch_total_ns": batch_total_ns,
        "throughput_patch_per_s": patches * 1.0e9 / batch_total_ns if batch_total_ns else 0.0,
        "full_model_total_ns": sum(full_model_times) if len(full_model_times) == len(reports) else None,
        "full_model_sequence_count": len(full_model_times),
        "full_model_mean_sequence_ns": (
            statistics.fmean(full_model_times) if full_model_times else None
        ),
        "full_model_median_sequence_ns": (
            statistics.median(full_model_times) if full_model_times else None
        ),
    })

    proposal_complexity = manifest.get("B_model_complexity", {})
    proposal_static = proposal_complexity.get("model_properties", {})
    proposal_dynamic = proposal_complexity.get("input_dependent", {})
    h2_complexities = [
        item.get("h2_complexity") for item in per_sequence
        if isinstance(item.get("h2_complexity"), dict)
    ]
    h2_parameter_counts = {int(item["A01_parameter_count"]) for item in h2_complexities}
    h2_nonzero_counts = {int(item["nonzero_parameter_count"]) for item in h2_complexities}
    h2_storage_counts = {int(item["A02_static_storage_bytes"]) for item in h2_complexities}
    h2_work_counts = {int(item["A03_theoretical_work_memory_bytes_per_call"]) for item in h2_complexities}
    h2_parameters = next(iter(h2_parameter_counts)) if len(h2_parameter_counts) == 1 else None
    h2_nonzero = next(iter(h2_nonzero_counts)) if len(h2_nonzero_counts) == 1 else None
    h2_storage = next(iter(h2_storage_counts)) if len(h2_storage_counts) == 1 else None
    h2_work = next(iter(h2_work_counts)) if len(h2_work_counts) == 1 else None
    proposal_parameters = proposal_static.get("A01_parameter_count")
    proposal_nonzero = proposal_static.get("nonzero_parameter_count")
    proposal_storage = proposal_static.get("A02_static_storage_bytes")
    proposal_state = proposal_static.get("A03_theoretical_persistent_state_bytes_per_patch")
    proposal_dense = proposal_dynamic.get("dense_synaptic_products")
    proposal_effective = proposal_dynamic.get("effective_synaptic_products")
    h2_dense = sum(int(item["A04_dense_MAC"]) for item in h2_complexities)
    h2_effective = sum(int(item["A05_effective_MAC"]) for item in h2_complexities)
    complexity_complete = (
        proposal_parameters is not None and proposal_dense is not None
        and len(h2_complexities) == len(reports) and h2_parameters is not None
    )
    model_complexity = {
        "schema": "5_model_evaluation/B_model_complexity",
        "status": "complete" if complexity_complete else "incomplete_missing_new_instrumentation",
        "A01_parameter_count": (
            int(proposal_parameters) + int(h2_parameters)
            if proposal_parameters is not None and h2_parameters is not None else None
        ),
        "nonzero_parameter_count": (
            int(proposal_nonzero) + int(h2_nonzero)
            if proposal_nonzero is not None and h2_nonzero is not None else None
        ),
        "A02_static_storage_bytes": (
            int(proposal_storage) + int(h2_storage)
            if proposal_storage is not None and h2_storage is not None else None
        ),
        "A03_work_and_state_memory": {
            "proposal_persistent_state_bytes_per_patch": proposal_state,
            "h2_theoretical_work_memory_bytes_per_call": h2_work,
            "h2_process_max_rss_kib_max": max(
                (int(item.get("process_max_rss_kib", -1)) for item in h2_complexities),
                default=None,
            ),
            "note": "process RSS is deployment memory and is not substituted for theoretical tensor/state memory",
        },
        "A04_dense_MAC_or_synaptic_products": (
            int(proposal_dense) + h2_dense if proposal_dense is not None else None
        ),
        "A04_breakdown": {"proposal_synaptic_products": proposal_dense, "refiner_MAC": h2_dense},
        "A05_effective_MAC_or_AC": (
            int(proposal_effective) + h2_effective if proposal_effective is not None else None
        ),
        "A05_breakdown": {"proposal_AC": proposal_effective, "refiner_effective_MAC": h2_effective},
        "A06_other_operations": {
            "proposal": proposal_dynamic.get("A06_other_operations"),
            "h2_activation_quantizations": sum(
                int(item.get("A06_activation_quantizations", 0)) for item in h2_complexities
            ),
            "h2_probability_evaluations": sum(
                int(item.get("A06_probability_evaluations", 0)) for item in h2_complexities
            ),
        },
        "operation_interpretation": (
            "A05 is exact input activity (nonzero source times nonzero weight), not proof that the dense CPU/RTL skipped work. "
            "Do not infer energy from A04/A05."
        ),
        "sources": {
            "proposal": "prepared input manifest",
            "refiner": "actual board H2 runtime reports",
        },
    }

    full_total_ns = aggregate["full_model_total_ns"]
    windows_per_sequence = int(manifest.get("primary_metrics", {}).get("windows_per_sequence", 80))
    input_duration_s = len(reports) * windows_per_sequence * 0.1
    full_total_s = full_total_ns / 1.0e9 if full_total_ns else None
    deployment_metrics = {
        "schema": "5_model_evaluation/C_deployment_efficiency_and_hardware_cost",
        "B01_model_kernel_latency": {
            "scope": "PS-observed PL start-to-done, including AXI polling overhead",
            "unit": "ns",
            **aggregate["pl_start_to_done_observed_ns"],
        },
        "B02_end_to_end_latency": {
            "scope": "input read + PS-to-PL + PL + PL-to-PS/storage + PS H2 + output fsync",
            "total_ns": full_total_ns,
            "mean_sequence_ns": aggregate["full_model_mean_sequence_ns"],
            "median_sequence_ns": aggregate["full_model_median_sequence_ns"],
        },
        "B03_throughput_and_realtime": {
            "patch_per_s": aggregate["throughput_patch_per_s"],
            "sequence_per_s": len(reports) / full_total_s if full_total_s else None,
            "input_duration_s": input_duration_s,
            "processing_to_input_realtime_factor": full_total_s / input_duration_s if full_total_s else None,
            "sustained_input_time_per_wall_time": input_duration_s / full_total_s if full_total_s else None,
            "condition": "serial full-batch run; no dropped/backlogged sequence counter in current PL",
        },
        "B04_runtime_power": power,
        "B05_energy_per_task": {
            "active_energy_j": power.get("active_energy_j"),
            "dynamic_energy_j": power.get("dynamic_energy_j"),
            "active_energy_per_sequence_j": power.get("active_energy_per_task_j"),
            "dynamic_energy_per_sequence_j": power.get("dynamic_energy_per_task_j"),
            "active_energy_per_patch_j": power.get("active_energy_per_patch_j"),
            "dynamic_energy_per_patch_j": power.get("dynamic_energy_per_patch_j"),
            "scope": power.get("measurement_scope"),
        },
        "B07_hardware_resources_and_area": area,
    }

    artifact_paths = {
        "bitstream": arguments.bitstream.resolve(),
        "dtbo": arguments.dtbo.resolve(),
        "shell": arguments.shell.resolve(),
    }
    for name, path in artifact_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{name} artifact is missing: {path}")
    current_firmware = implementation.get("deployment_artifact_verification", {}).get(
        "current_firmware", []
    )
    expected_hashes = {
        str(item.get("name")): str(item.get("sha256", "")).lower()
        for item in current_firmware if isinstance(item, dict)
    }
    artifact_names = {
        "bitstream": "wangdao_pl_flat.bit.bin",
        "dtbo": "wangdao_pl_flat.dtbo",
        "shell": "shell.json",
    }
    if expected_hashes:
        for kind, path in artifact_paths.items():
            expected = expected_hashes.get(artifact_names[kind])
            if not expected or sha256_file(path).lower() != expected:
                raise ValueError(f"{kind} does not match hardware_resource_manifest.json")
    else:
        implementation_hashes = {
            str(item.get("sha256", "")).lower()
            for item in implementation.get("artifacts", []) if isinstance(item, dict)
        }
        if sha256_file(artifact_paths["bitstream"]).lower() not in implementation_hashes:
            raise ValueError("bitstream hash is not present in the selected implementation evidence")

    if not complexity_complete:
        result_status = "incomplete_missing_complexity_instrumentation"
    elif power.get("status") != "ok":
        result_status = "timing_and_area_complete_power_missing"
    elif bool(power.get("pl_power_is_directly_measured", False)):
        result_status = "complete"
    else:
        result_status = "complete_with_som_power_proxy"
    result: Dict[str, Any] = {
        "schema_version": 1,
        "status": result_status,
        "experiment_id": arguments.experiment_id,
        "data_role": manifest.get("data_role"),
        "configuration": {
            "generation": arguments.generation,
            "pl_bitstream_changed": False,
            "transport": "AXI-Lite one patch at a time",
            "timing_clock": "CLOCK_MONOTONIC_RAW",
        },
        "primary_metrics": {
            "sequences": len(reports),
            "patches": patches,
            "batch_total_ns": batch_total_ns,
            "throughput_patch_per_s": aggregate["throughput_patch_per_s"],
            "full_model_total_ns": aggregate["full_model_total_ns"],
        },
        "timing": aggregate,
        "power": power,
        "area": area,
        "B_model_complexity": model_complexity,
        "C_deployment_efficiency_and_hardware_cost": deployment_metrics,
        "per_sequence": per_sequence,
        "artifacts": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            **{
                name: {"path": str(path), "sha256": sha256_file(path)}
                for name, path in artifact_paths.items()
            },
        },
        "limitations": [
            "Storage reads may be served by Linux page cache unless a controlled cold-cache protocol is used.",
            "PL start-to-done is PS-observed and includes polling; pure PL cycles require a new RTL counter/bitstream.",
            "AXI-Lite input cannot overlap safely with PL compute because the current PL has one shared input buffer.",
            "Power remains incomplete until a measured power JSON is supplied.",
        ],
    }
    atomic_json(result, arguments.output.resolve())
    print(json.dumps({
        "status": result["status"],
        "sequences": len(reports),
        "patches": patches,
        "output": str(arguments.output.resolve()),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
