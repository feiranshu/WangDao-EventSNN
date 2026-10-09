"""Prepare per-sequence binary Proposal-SNN transfers for KV260 board runs.

The generated input files contain one 512-byte record per patch.  Records are
ordered by selected window and then by patch index.  The matching golden file
contains sixteen little-endian signed int32 Q12 logits for every record.

This is intentionally a preparation step: it does not access a board and it
does not run the reserved Test split unless the caller explicitly requests it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))

from wangdao.data import load_events
from wangdao.fixed import FixedProposalSNN, load_fixed_proposal_snn, signed_round_divide
from wangdao.h2.constants import PATCHES_PER_WINDOW, PATCH_FEATURE_DIM, WINDOWS_PER_SEQUENCE
from wangdao.s0p import build_s0p_sequence


INDEX_PATH = ROOT / "dataset" / "index.json"
DEFAULT_WEIGHT = ROOT / "model_runtime" / "weights" / "S1P_INT8_ACC24_MEM24_Q12.npz"
DEFAULT_OUTPUT = ROOT / "experiments" / "prepared_packages" / "hw_calibration" / "s1_batch"
TIME_STEPS = 4
OUTPUT_DIM = 16
BYTES_PER_PATCH = TIME_STEPS * PATCH_FEATURE_DIM


def forward_with_complexity(
    model: FixedProposalSNN, features: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run the frozen SNN and count input-dependent synaptic work exactly.

    ``effective_synaptic_products`` follows the B-model-complexity definition:
    a product is counted only when both the source activation/spike and the
    corresponding int8 weight are non-zero.  This is an algorithmic activity
    count for the exact prepared input; it is not a claim that the current RTL
    skips those products or consumes proportional energy.
    """
    value = np.ascontiguousarray(features, dtype=np.uint8)
    batch = int(value.shape[1])
    hidden_membranes = [
        np.zeros((batch, model.weights[0].shape[0]), dtype=np.int64),
        np.zeros((batch, model.weights[1].shape[0]), dtype=np.int64),
    ]
    readout_membrane = np.zeros((batch, model.weights[2].shape[0]), dtype=np.int64)
    spike_membrane = np.zeros_like(readout_membrane)
    readout_sum = np.zeros_like(readout_membrane)
    nonzero_fanout = [np.count_nonzero(weight, axis=0).astype(np.int64) for weight in model.weights]
    effective = [0, 0, 0]
    spike_counts = [0, 0, 0]
    input_nonzero = 0

    for step in range(model.time_steps):
        source = value[step]
        input_nonzero += int(np.count_nonzero(source))
        effective[0] += int(np.count_nonzero(source, axis=0) @ nonzero_fanout[0])
        current, _ = model._current(source, 0, model.configuration.input_fraction_bits)
        spikes_1, hidden_membranes[0], _ = model._update(current, hidden_membranes[0], True, 0)
        spike_counts[0] += int(spikes_1.sum())

        effective[1] += int(np.count_nonzero(spikes_1, axis=0) @ nonzero_fanout[1])
        current, _ = model._current(spikes_1, 1, 0)
        spikes_2, hidden_membranes[1], _ = model._update(current, hidden_membranes[1], True, 1)
        spike_counts[1] += int(spikes_2.sum())

        effective[2] += int(np.count_nonzero(spikes_2, axis=0) @ nonzero_fanout[2])
        output_current, _ = model._current(spikes_2, 2, 0)
        _unused, readout_membrane, _ = model._update(output_current, readout_membrane, False, 2)
        readout_sum += readout_membrane
        output_spikes, spike_membrane, _ = model._update(output_current, spike_membrane, True)
        spike_counts[2] += int(output_spikes.sum())

    logits_q = signed_round_divide(readout_sum, model.time_steps)
    dense_per_patch = model.time_steps * sum(int(weight.size) for weight in model.weights)
    membrane_updates_per_patch = model.time_steps * (
        int(model.weights[0].shape[0]) + int(model.weights[1].shape[0])
        + 2 * int(model.weights[2].shape[0])
    )
    return logits_q, {
        "dense_synaptic_products": int(batch * dense_per_patch),
        "effective_synaptic_products": int(sum(effective)),
        "effective_synaptic_products_by_layer": [int(item) for item in effective],
        "input_nonzero_activations": input_nonzero,
        "spikes_by_layer": [int(item) for item in spike_counts],
        "membrane_updates": int(batch * membrane_updates_per_patch),
    }


def proposal_static_complexity(model: FixedProposalSNN) -> Dict[str, Any]:
    parameters = sum(int(weight.size) for weight in model.weights)
    nonzero = sum(int(np.count_nonzero(weight)) for weight in model.weights)
    channels = sum(int(weight.shape[0]) for weight in model.weights)
    state_values = (
        int(model.weights[0].shape[0]) + int(model.weights[1].shape[0])
        + 2 * int(model.weights[2].shape[0])
    )
    return {
        "A01_parameter_count": parameters,
        "nonzero_parameter_count": nonzero,
        "A02_static_storage_bytes": parameters + channels * 4,
        "weight_storage_bytes_int8": parameters,
        "per_output_fraction_metadata_bytes_int8": channels,
        "float32_scale_equivalent_bytes_for_cross_model_comparison": channels * 4,
        "actual_deployed_weight_and_fraction_metadata_bytes": parameters + channels,
        "A03_theoretical_persistent_state_bytes_per_patch": state_values * 3,
        "state_precision_bits": 24,
        "A04_dense_synaptic_products_per_patch": model.time_steps * parameters,
        "membrane_updates_per_patch": model.time_steps * state_values,
        "method": "derived from the frozen int8 weights and exact fixed SNN schedule",
    }


def merge_counts(target: Dict[str, int], source: Dict[str, Any]) -> None:
    for key in ("dense_synaptic_products", "effective_synaptic_products",
                "input_nonzero_activations", "membrane_updates"):
        target[key] = target.get(key, 0) + int(source[key])
    for key in ("effective_synaptic_products_by_layer", "spikes_by_layer"):
        values = [int(item) for item in source[key]]
        current = target.setdefault(key, [0] * len(values))
        for index, item in enumerate(values):
            current[index] += item


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_path(path: Path, root: Path = ROOT) -> str:
    """Return a repo-relative path when possible, otherwise an absolute path."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_index() -> Dict[str, Any]:
    with INDEX_PATH.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("status") != "ok":
        raise ValueError("dataset index is not ready")
    return value


def resolve_dataset_paths(index: Dict[str, Any], roles: Sequence[str], sequence: str | None,
                          max_sequences: int | None) -> List[Tuple[str, str, Path]]:
    location = index.get("location")
    records = index.get("files")
    role_map = index.get("roles")
    if not isinstance(location, str) or not isinstance(records, list) or not isinstance(role_map, dict):
        raise ValueError("dataset index is missing location, files or roles")
    location_path = Path(location)
    if location_path.parts and location_path.parts[0] == "data":
        location_path = Path("dataset", *location_path.parts[1:])
    dataset_root = ROOT / location_path
    by_name = {
        Path(str(record["path"])).name: dataset_root / str(record["path"])
        for record in records
        if isinstance(record, dict) and "path" in record
    }
    role_names: Dict[str, str] = {}
    for role in ("train_main", "train_dev", "hw_calibration", "validation", "test"):
        names = role_map.get(role)
        if not isinstance(names, list):
            raise ValueError(f"dataset index has no role list: {role}")
        for name in names:
            role_names[str(name)] = role

    if sequence is not None:
        candidate = Path(sequence)
        if not candidate.is_file():
            candidate = by_name.get(candidate.name, candidate)
        if not candidate.is_file():
            raise FileNotFoundError(f"dataset sequence not found: {sequence}")
        name = candidate.name
        return [(name, role_names.get(name, "custom"), candidate.resolve())]

    selected_roles = list(roles)
    if "all_non_test" in selected_roles:
        selected_roles = ["train_main", "train_dev", "hw_calibration", "validation"]
    if "all" in selected_roles:
        selected_roles = ["train_main", "train_dev", "hw_calibration", "validation", "test"]
    if not selected_roles:
        raise ValueError("at least one role or --sequence is required")
    result: List[Tuple[str, str, Path]] = []
    seen: set[str] = set()
    for role in selected_roles:
        names = role_map.get(role)
        if not isinstance(names, list):
            raise ValueError(f"unknown dataset role: {role}")
        for raw_name in names:
            name = str(raw_name)
            if name in seen:
                continue
            if name not in by_name:
                raise ValueError(f"role references missing sequence: {name}")
            result.append((name, role, by_name[name].resolve()))
            seen.add(name)
    if max_sequences is not None:
        if max_sequences < 1:
            raise ValueError("--max-sequences must be positive")
        result = result[:max_sequences]
    if not result:
        raise ValueError("selected dataset roles are empty")
    missing = [str(path) for _name, _role, path in result if not path.is_file()]
    if missing:
        raise FileNotFoundError("dataset files are absent: " + ", ".join(missing))
    return result


def selected_windows(start: int, count: int) -> List[int]:
    if start < 0 or start >= WINDOWS_PER_SEQUENCE:
        raise ValueError("--window-start is outside the sequence")
    if count < 1 or start + count > WINDOWS_PER_SEQUENCE:
        raise ValueError("selected windows exceed the 80-window sequence")
    return list(range(start, start + count))


def atomic_write_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent,
                                      delete=False) as handle:
        temporary = Path(handle.name)
        np.asarray(array).tofile(handle)
    temporary.replace(path)


def prepare_sequence(path: Path, model: FixedProposalSNN, windows: Sequence[int],
                     input_path: Path, golden_path: Path, output_dir: Path,
                     batch_size: int) -> Dict[str, Any]:
    events = load_events(path, require_labels=True)
    sequence = build_s0p_sequence(path, TIME_STEPS, windows=windows)
    temporal = sequence.temporal_features
    expected_shape = (len(windows), TIME_STEPS, PATCHES_PER_WINDOW, PATCH_FEATURE_DIM)
    if temporal.shape != expected_shape or temporal.dtype != np.uint8:
        raise ValueError(f"unexpected temporal feature shape/dtype for {path.name}: {temporal.shape} {temporal.dtype}")
    # Board order is window-major, patch-major, then four time steps and 128 features.
    patches = np.ascontiguousarray(temporal.transpose(0, 2, 1, 3).reshape(
        len(windows) * PATCHES_PER_WINDOW, TIME_STEPS, PATCH_FEATURE_DIM
    ))
    logits_parts: List[np.ndarray] = []
    complexity: Dict[str, Any] = {}
    for start in range(0, len(patches), batch_size):
        logits, batch_complexity = forward_with_complexity(
            model, patches[start:start + batch_size].transpose(1, 0, 2)
        )
        merge_counts(complexity, batch_complexity)
        logits = np.asarray(logits, dtype=np.int64)
        if logits.shape != (len(patches[start:start + batch_size]), OUTPUT_DIM):
            raise ValueError("frozen Proposal-SNN returned an unexpected shape")
        if np.any(logits < -(1 << 31)) or np.any(logits >= (1 << 31)):
            raise ValueError("Proposal logits do not fit the board int32 transport")
        logits_parts.append(logits.astype("<i4", copy=False))
    logits_q = np.concatenate(logits_parts, axis=0)
    atomic_write_array(input_path, patches.reshape(-1))
    atomic_write_array(golden_path, logits_q.reshape(-1))
    return {
        "name": path.name,
        "path": portable_path(path),
        "windows": [int(value) for value in windows],
        "patches": int(len(patches)),
        "events": int(len(events)),
        "target_events": int(events["label"].sum()),
        "input_file": input_path.relative_to(output_dir).as_posix(),
        "golden_file": golden_path.relative_to(output_dir).as_posix(),
        "input_bytes": int(input_path.stat().st_size),
        "golden_bytes": int(golden_path.stat().st_size),
        "input_sha256": sha256_file(input_path),
        "golden_sha256": sha256_file(golden_path),
        "B_model_complexity_input_dependent": {
            **complexity,
            "status": "derived_exact_input_activity",
            "scope": "Proposal-SNN over all selected time steps and patches",
            "effective_operation_note": (
                "Counts nonzero source activation/spike times nonzero outgoing weight; "
                "the current dense RTL is not claimed to skip these operations."
            ),
        },
    }


def atomic_json(value: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".tmp", dir=path.parent,
                                      mode="w", encoding="utf-8", newline="\n", delete=False) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", action="append", default=None,
                        choices=["train_main", "train_dev", "hw_calibration", "validation",
                                 "test", "all_non_test", "all"])
    parser.add_argument("--sequence", help="one dataset filename/path; overrides --role")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--window-count", type=int, default=WINDOWS_PER_SEQUENCE)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--s1-weight", type=Path, default=DEFAULT_WEIGHT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    weight_path = arguments.s1_weight.resolve()
    if not weight_path.is_file():
        raise FileNotFoundError(weight_path)
    index = load_index()
    roles = arguments.role if arguments.role is not None else ["hw_calibration"]
    paths = resolve_dataset_paths(index, roles, arguments.sequence, arguments.max_sequences)
    windows = selected_windows(arguments.window_start, arguments.window_count)
    model, configuration, metadata = load_fixed_proposal_snn(weight_path)
    output_dir = arguments.output_dir.resolve()
    sequence_dir = output_dir / "sequences"
    sequence_records: List[Dict[str, Any]] = []
    started = time.perf_counter()
    role_label = "custom" if arguments.sequence else "+".join(roles)
    for number, (name, role, path) in enumerate(paths, start=1):
        stem = Path(name).stem
        input_path = sequence_dir / f"{stem}.inputs.bin"
        golden_path = sequence_dir / f"{stem}.golden.bin"
        record = prepare_sequence(path, model, windows, input_path, golden_path,
                                  output_dir, arguments.batch_size)
        record["role"] = role
        sequence_records.append(record)
        print(f"[{number:03d}/{len(paths):03d}] {name}: {record['patches']} patches", flush=True)

    total_patches = sum(int(record["patches"]) for record in sequence_records)
    static_complexity = proposal_static_complexity(model)
    total_complexity: Dict[str, Any] = {}
    for record in sequence_records:
        merge_counts(total_complexity, record["B_model_complexity_input_dependent"])
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "experiment_id": "KV260-BATCH-VECTORS-v1",
        "stage": "P5-kv260-batch-preparation",
        "data_role": role_label,
        "primary_metrics": {
            "sequences": len(sequence_records),
            "windows_per_sequence": len(windows),
            "patches": total_patches,
            "input_bytes": sum(int(record["input_bytes"]) for record in sequence_records),
            "golden_logit_records": total_patches * OUTPUT_DIM,
        },
        "decision": "ready_for_board_batch",
        "conclusion": "Per-sequence exact Proposal-SNN board transfers and golden logits are ready.",
        "configuration": {
            "base_address": "0xA0000000",
            "time_steps": TIME_STEPS,
            "feature_dim": PATCH_FEATURE_DIM,
            "bytes_per_patch": BYTES_PER_PATCH,
            "patches_per_window": PATCHES_PER_WINDOW,
            "output_dim": OUTPUT_DIM,
            "output_format": "little-endian signed int32 Q12",
            "bitstream_mode": "S1 exact fixed schedule",
            "board_transport": "AXI-Lite one patch at a time",
        },
        "source": {
            "dataset_index": portable_path(INDEX_PATH),
            "dataset_index_sha256": sha256_file(INDEX_PATH),
            "s1_weight": portable_path(weight_path),
            "s1_weight_sha256": sha256_file(weight_path),
            "s1_configuration": configuration,
            "s1_metadata": metadata,
            "script_sha256": sha256_file(Path(__file__).resolve()),
            "test_used": any(record["role"] == "test" for record in sequence_records),
        },
        "B_model_complexity": {
            "schema": "5_model_evaluation/B_model_complexity",
            "status": "complete_for_proposal_preparation",
            "model_properties": static_complexity,
            "input_dependent": {
                **total_complexity,
                "A05_effective_MAC_or_AC": int(total_complexity["effective_synaptic_products"]),
                "A06_other_operations": {
                    "input_activation_quantizations_or_reads": total_patches * TIME_STEPS * PATCH_FEATURE_DIM,
                    "threshold_comparisons": total_patches * TIME_STEPS * (
                        int(model.weights[0].shape[0]) + int(model.weights[1].shape[0])
                        + int(model.weights[2].shape[0])
                    ),
                    "readout_accumulations": total_patches * TIME_STEPS * OUTPUT_DIM,
                },
            },
            "provenance": "computed while generating the exact board input and golden logits",
        },
        "sequences": sequence_records,
        "elapsed_s": time.perf_counter() - started,
    }
    atomic_json(manifest, output_dir / "manifest.json")
    print(json.dumps({"status": "ok", "manifest": str((output_dir / 'manifest.json').resolve()),
                      "sequences": len(sequence_records), "patches": total_patches}, ensure_ascii=False))


if __name__ == "__main__":
    main()
