"""Prepare self-contained PS-side H2 inputs and exact golden predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
from typing import Any, Dict, List

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))

from wangdao.data import load_events
from wangdao.fixed import FixedH2, FixedH2Config, load_fixed_proposal_snn
from wangdao.h2.features import extract_event_features
from wangdao.h2.numpy_model import _refiner_probability, load_export
from wangdao.h2.representation import (
    build_refiner_features,
    candidate_indices,
    patch_scores_to_heatmaps,
    window_indices,
)


EVENT_MAGIC = 0x32445645
H2_MAGIC = 0x32444857
FILE_VERSION = 1
EVENT_RECORD_BYTES = 36
EVENT_FEATURE_DIM = 28
WINDOWS = 80
PATCHES_PER_WINDOW = 336
OUTPUT_DIM = 16
Q12_SCALE = 4096.0


def portable_path(path: Path, root: Path = ROOT) -> str:
    """Use a relative provenance path only when both paths share a root."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()
H2_WEIGHT = ROOT / "model_runtime" / "weights" / "H2_D1_seed42.npz"

EVENT_DTYPE = np.dtype(
    [
        ("x", "<u2"),
        ("y", "<u2"),
        ("window", "u1"),
        ("within_q8", "u1"),
        ("label", "u1"),
        ("name", "u1"),
        ("features", "u1", (EVENT_FEATURE_DIM,)),
    ],
    align=False,
)


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


def signed_q12_to_probability(logits_q: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits_q, dtype=np.float64) / Q12_SCALE
    clipped = np.clip(logits, -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def build_event_records(events: np.ndarray) -> np.ndarray:
    base, _ = extract_event_features(events)
    assigned = window_indices(events).astype(np.uint8, copy=False)
    origin = float(events["t"][0])
    within = np.mod(events["t"].astype(np.float64) - origin, 100.0) / 100.0
    within_q8 = np.rint(255.0 * np.clip(within, 0.0, 1.0)).astype(np.uint8)
    records = np.empty(len(events), dtype=EVENT_DTYPE)
    records["x"] = events["x"].astype(np.uint16, copy=False)
    records["y"] = events["y"].astype(np.uint16, copy=False)
    records["window"] = assigned
    records["within_q8"] = within_q8
    records["label"] = events["label"].astype(np.uint8, copy=False)
    records["name"] = events["name"].astype(np.uint8, copy=False)
    records["features"] = base
    if records.dtype.itemsize != EVENT_RECORD_BYTES:
        raise AssertionError("event record format changed")
    return np.ascontiguousarray(records)


def write_event_file(path: Path, records: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack(
        "<8I", EVENT_MAGIC, FILE_VERSION, len(records), EVENT_RECORD_BYTES,
        WINDOWS, 346, 260, EVENT_FEATURE_DIM,
    )
    with path.open("wb") as handle:
        handle.write(header)
        records.tofile(handle)


def write_h2_weights(path: Path, h2_path: Path) -> Dict[str, Any]:
    export = load_export(h2_path)
    if len(export.refiner_layers) != 3:
        raise ValueError("H2 refiner must contain three layers")
    header = struct.pack(
        "<9I2f", H2_MAGIC, FILE_VERSION, 3, 24, 20, 16, 2, 32, 4,
        np.float32(export.proposal_threshold).item(),
        np.float32(export.event_threshold).item(),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(header)
        for weight, scale in export.refiner_layers:
            weight = np.ascontiguousarray(weight, dtype=np.int8)
            multipliers = np.rint(scale.astype(np.float64) * (1 << 20)).astype("<i4")
            handle.write(struct.pack("<2I", weight.shape[0], weight.shape[1]))
            handle.write(weight.tobytes(order="C"))
            handle.write(multipliers.tobytes(order="C"))
    parameter_count = sum(int(weight.size) for weight, _scale in export.refiner_layers)
    nonzero_parameter_count = sum(
        int(np.count_nonzero(weight)) for weight, _scale in export.refiner_layers
    )
    scale_count = sum(int(scale.size) for _weight, scale in export.refiner_layers)
    return {
        "format": "wangdao_ps_h2_weights_v1",
        "sha256": sha256_file(path),
        "proposal_threshold": float(export.proposal_threshold),
        "event_threshold": float(export.event_threshold),
        "dimensions": [list(weight.shape) for weight, _scale in export.refiner_layers],
        "parameter_count": parameter_count,
        "nonzero_parameter_count": nonzero_parameter_count,
        "weight_storage_bytes_int8": parameter_count,
        "scale_storage_bytes_float32": scale_count * 4,
        "dense_macs_per_refiner_call": parameter_count,
        "accumulator_bits": 24,
        "scale_fraction_bits": 20,
        "logit_fraction_bits": 16,
    }


def write_prediction_golden(path: Path, events: np.ndarray, logits_q: np.ndarray,
                            windows: List[int], refiner: FixedH2,
                            proposal_threshold: float, event_threshold: float) -> Dict[str, int]:
    scores = signed_q12_to_probability(logits_q).reshape(
        len(windows), PATCHES_PER_WINDOW, OUTPUT_DIM
    )
    heatmaps = patch_scores_to_heatmaps(scores)
    candidates = candidate_indices(events, windows, heatmaps, proposal_threshold)
    features = build_refiner_features(
        events, candidates, windows, heatmaps, proposal_threshold
    )
    probability = np.zeros(len(events), dtype=np.float32)
    if len(features):
        raw = refiner.refiner(features).logits
        probability[candidates] = _refiner_probability(raw, refiner.source.prototypes)
    output = np.zeros((len(events), 2), dtype=np.uint8)
    output[candidates, 0] = 1
    output[:, 1] = (probability >= event_threshold).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.reshape(-1).tofile(path)
    return {
        "events": int(len(events)),
        "candidates": int(len(candidates)),
        "positives": int(np.count_nonzero(output[:, 1])),
        "target_events": int(np.count_nonzero(events["label"])),
        "candidate_targets": int(np.count_nonzero(events["label"][candidates])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Proposal-SNN batch manifest")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="directory for events, H2 weights and golden predictions")
    parser.add_argument("--h2-weight", type=Path, default=H2_WEIGHT)
    parser.add_argument("--max-sequences", type=int)
    arguments = parser.parse_args()

    manifest_path = arguments.manifest.resolve()
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("status") != "ok" or manifest.get("schema_version") != 1:
        raise ValueError("manifest is not a successful Proposal-SNN batch manifest")
    sequences = list(manifest.get("sequences", []))
    if arguments.max_sequences is not None:
        if arguments.max_sequences < 1:
            raise ValueError("--max-sequences must be positive")
        sequences = sequences[:arguments.max_sequences]
    if not sequences:
        raise ValueError("manifest contains no sequences")

    output_dir = arguments.output_dir.resolve()
    event_dir = output_dir / "events"
    golden_dir = output_dir / "h2_golden"
    h2_path = arguments.h2_weight.resolve()
    if not h2_path.is_file():
        raise FileNotFoundError(h2_path)
    h2_info = write_h2_weights(output_dir / "h2_weights.bin", h2_path)

    s1_path = ROOT / str(manifest["source"]["s1_weight"])
    s1_model, _s1_configuration, s1_metadata = load_fixed_proposal_snn(s1_path)
    del s1_model
    h2_configuration = s1_metadata["h2_fixed_configuration"]
    refiner = FixedH2(load_export(h2_path), FixedH2Config(
        accumulator_bits=int(h2_configuration["accumulator_bits"]),
        scale_fraction_bits=int(h2_configuration["scale_fraction_bits"]),
        logit_fraction_bits=int(h2_configuration["logit_fraction_bits"]),
    ))
    proposal_threshold = float(s1_metadata["proposal_threshold"])
    event_threshold = float(s1_metadata["event_threshold"])
    package_root = manifest_path.parent
    output_records: List[Dict[str, Any]] = []
    for number, source_record in enumerate(sequences, start=1):
        name = str(source_record["name"])
        windows = [int(value) for value in source_record["windows"]]
        if windows != list(range(WINDOWS)):
            raise ValueError(f"PS H2 package requires all 80 windows: {name}")
        event_path = (ROOT / str(source_record["path"])).resolve()
        events = load_events(event_path, require_labels=True)
        records = build_event_records(events)
        event_file = event_dir / f"{Path(name).stem}.events.bin"
        golden_file = golden_dir / f"{Path(name).stem}.pred.bin"
        write_event_file(event_file, records)
        golden_path = package_root / str(source_record["golden_file"])
        logits = np.fromfile(golden_path, dtype="<i4").reshape(
            len(windows) * PATCHES_PER_WINDOW, OUTPUT_DIM
        )
        metrics = write_prediction_golden(
            golden_file, events, logits, windows, refiner,
            proposal_threshold, event_threshold,
        )
        output_records.append({
            "name": name,
            "role": source_record.get("role"),
            "events": int(len(events)),
            "event_file": event_file.relative_to(output_dir).as_posix(),
            "event_bytes": int(event_file.stat().st_size),
            "event_sha256": sha256_file(event_file),
            "golden_file": golden_file.relative_to(output_dir).as_posix(),
            "golden_bytes": int(golden_file.stat().st_size),
            "golden_sha256": sha256_file(golden_file),
            "metrics": metrics,
        })
        print(f"[{number:03d}/{len(sequences):03d}] {name}: events={len(events)} candidates={metrics['candidates']}", flush=True)

    output_manifest: Dict[str, Any] = {
        "schema_version": 1,
        "status": "ok",
        "experiment_id": "KV260-PS-PL-FULL-v1",
        "stage": "P5-kv260-ps-pl-h2-preparation",
        "data_role": manifest.get("data_role"),
        "primary_metrics": {
            "sequences": len(output_records),
            "events": sum(item["events"] for item in output_records),
            "candidates": sum(item["metrics"]["candidates"] for item in output_records),
            "positives": sum(item["metrics"]["positives"] for item in output_records),
        },
        "decision": "ready_for_ps_pl_full_run",
        "conclusion": "Board captures can be routed and refined on the PS with the frozen H2 model.",
        "configuration": {
            "event_format": "header plus 36-byte little-endian records",
            "prediction_format": "two raw bytes per event: candidate, prediction",
            "h2_weights": "h2_weights.bin",
            "proposal_threshold": proposal_threshold,
            "event_threshold": event_threshold,
            "h2_fixed_configuration": h2_configuration,
        },
        "source": {
            "s1_manifest": portable_path(manifest_path),
            "s1_manifest_sha256": sha256_file(manifest_path),
            "h2_weight": portable_path(h2_path),
            "h2_weight_sha256": sha256_file(h2_path),
            "h2_weights_bin_sha256": h2_info["sha256"],
            "test_used": bool(manifest.get("source", {}).get("test_used", False)),
        },
        "B_model_complexity": {
            "schema": "5_model_evaluation/B_model_complexity",
            "status": "static_and_scheduled_complete_effective_h2_from_board_runtime",
            "model_properties": {
                "proposal": manifest.get("B_model_complexity", {}).get("model_properties"),
                "refiner": {
                    "A01_parameter_count": h2_info["parameter_count"],
                    "nonzero_parameter_count": h2_info["nonzero_parameter_count"],
                    "A02_static_storage_bytes": (
                        h2_info["weight_storage_bytes_int8"]
                        + h2_info["scale_storage_bytes_float32"]
                    ),
                    "weight_storage_bytes_int8": h2_info["weight_storage_bytes_int8"],
                    "scale_storage_bytes_float32": h2_info["scale_storage_bytes_float32"],
                    "A03_theoretical_work_memory_bytes_per_call": 176,
                    "work_memory_components": {
                        "features_u8": 32,
                        "current_u8": 64,
                        "next_u8": 64,
                        "logits_i32": 16,
                    },
                    "A04_dense_MAC_per_call": h2_info["dense_macs_per_refiner_call"],
                },
            },
            "input_dependent_expected_from_exact_golden_routing": {
                "proposal": manifest.get("B_model_complexity", {}).get("input_dependent"),
                "refiner_calls": sum(item["metrics"]["candidates"] for item in output_records),
                "refiner_dense_MAC": sum(item["metrics"]["candidates"] for item in output_records)
                * h2_info["dense_macs_per_refiner_call"],
                "A05_effective_H2_MAC": None,
                "A05_status": "emitted by wangdao_ps_h2.c from actual board-side activations",
                "A06_other_operations": {
                    "proposal_sigmoid_evaluations": len(output_records) * WINDOWS
                    * PATCHES_PER_WINDOW * OUTPUT_DIM,
                    "refiner_probability_evaluations": sum(
                        item["metrics"]["candidates"] for item in output_records
                    ),
                },
            },
        },
        "sequences": output_records,
    }
    atomic_json(output_manifest, output_dir / "ps_manifest.json")
    print(json.dumps({
        "status": "ok",
        "manifest": str((output_dir / "ps_manifest.json").resolve()),
        "sequences": len(output_records),
        "events": output_manifest["primary_metrics"]["events"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
