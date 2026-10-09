"""Frozen third-layer and full Proposal-SNN RTL vectors, without dataset evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))
from scripts.generate_rtl_vectors import (DEFAULT_OUTPUT, S1_WEIGHT_PATH, S3_WEIGHT_PATH,
    atomic_text, decode_signed, engine_trace, frozen_first_layer, hardware_calibration_path,
    second_layer_trace, sha256_file, write_weight_file)
from scripts.run_rtl_simulation import validate_manifest
from wangdao.fixed import FixedProposalSNN, loa_add_signed, signed_round_shift
from wangdao.s0p import build_s0p_sequence


def readout_trace(spikes: np.ndarray, model: FixedProposalSNN, low_bits: int) -> tuple[np.ndarray, np.ndarray]:
    """Full 24-leaf tree, four unreset membranes, then exact mean (no diagnostic spikes)."""
    if spikes.ndim != 3 or spikes.shape[1:] != (4, 24) or not np.isin(spikes, (0, 1)).all():
        raise ValueError("Readout input must be binary with shape (patches, 4, 24)")
    if model.readout != "mean_membrane" or model.time_steps != 4:
        raise ValueError("Frozen readout requires four-step mean_membrane")
    weights = model.weights[2].astype(np.int64)
    fractions = model.weight_fraction_bits[2].astype(np.int64)
    if weights.shape != (16, 24) or set(fractions.tolist()) != {5, 6}:
        raise ValueError("Unexpected third-layer frozen weights/scales")
    # Each k=2 LOA node deviates from exact addition by at most three.
    # This bound covers all binary patterns and every possible internal node.
    bound = (np.abs(weights).sum(-1) + 23 * 3) << (12 - fractions)
    if np.any(bound >= (1 << 23)):
        raise ValueError("Readout current bound exceeds the 24-bit interface")
    membrane = np.zeros((len(spikes), 16), dtype=np.int64)
    total = np.zeros_like(membrane)
    trace = np.zeros((len(spikes), 4, 16, 4), dtype=np.int64)
    for step in range(4):
        values = spikes[:, step, None, :].astype(np.int64) * weights[None, :, :]
        tree_sats = np.zeros_like(membrane)
        while values.shape[-1] > 1:
            pairs = values.shape[-1] // 2
            raw, _ = loa_add_signed(values[..., :pairs * 2:2], values[..., 1:pairs * 2:2], 25, low_bits)
            tree_sats += ((raw < -(1 << 23)) | (raw >= (1 << 23))).sum(-1)
            reduced = np.clip(raw, -(1 << 23), (1 << 23) - 1)
            values = np.concatenate((reduced, values[..., -1:]), -1) if values.shape[-1] % 2 else reduced
        current = values[..., 0] << (12 - fractions)
        leaked = signed_round_shift(membrane * 7, 3)
        raw, _ = loa_add_signed(leaked, current, 25, low_bits)
        mem_sats = (raw < -(1 << 23)) | (raw >= (1 << 23))
        updated = np.clip(raw, -(1 << 23), (1 << 23) - 1)
        reference_current, reference_tree_sats = model._current(spikes[:, step], 2, 0)
        unused_spikes, reference_membrane, reference_mem_sats = model._update(reference_current, membrane, False, 2)
        if (not np.array_equal(reference_current, current) or not np.array_equal(reference_membrane, updated)
                or unused_spikes.any() or reference_tree_sats != int(tree_sats.sum())
                or reference_mem_sats != int(mem_sats.sum())):
            raise AssertionError("Readout trace differs from frozen model")
        membrane = updated
        total += membrane
        if np.any(total < -(1 << 25)) or np.any(total >= (1 << 25)):
            raise AssertionError("Readout sum exceeds 26 bits")
        trace[:, step, :, 0] = membrane & 0xFFFFFF
        trace[:, step, :, 1] = total & 0x3FFFFFF
        trace[:, step, :, 2] = tree_sats
        trace[:, step, :, 3] = mem_sats
    final = np.stack((signed_round_shift(total, 2) & 0xFFFFFFFF,
                      trace[..., 2].sum(1), trace[..., 3].sum(1)), axis=-1)
    return trace, final


def write_records(path: Path, values: np.ndarray, bases: list[int]) -> dict[str, Any]:
    rows = values.reshape(-1, len(bases))
    atomic_text(path, (" ".join(f"{int(v):08X}" if base == 16 else str(int(v))
                                for v, base in zip(row, bases)) + "\n" for row in rows))
    restored = np.asarray([[int(v, base) for v, base in zip(line.split(), bases)]
                          for line in path.read_text(encoding="ascii").splitlines()], dtype=np.int64)
    if not np.array_equal(restored, rows):
        raise AssertionError("Serialized readout vectors differ from golden arrays")
    return {"path": path.name, "records": len(rows), "sha256": sha256_file(path)}


def generate(output_directory: Path, real_patches: int, seed: int) -> dict[str, Any]:
    manifest_path = output_directory / "proposal_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Refusing to overwrite existing Proposal vectors")
    if not 0 <= real_patches <= 336:
        raise ValueError("real-patches must be between 0 and 336")
    exact, approximate = frozen_first_layer()
    for layer in (1, 2):
        if (not np.array_equal(exact.weights[layer], approximate.weights[layer])
                or not np.array_equal(exact.weight_fraction_bits[layer], approximate.weight_fraction_bits[layer])):
            raise ValueError("S1/S3 frozen hidden/readout weights or scaling differ")
    original, _ = validate_manifest(DEFAULT_OUTPUT, "two_layers")
    source = hardware_calibration_path()
    if (original["source"]["frozen_weight_sha256"] != sha256_file(S3_WEIGHT_PATH)
            or original["source"]["exact_weight_sha256"] != sha256_file(S1_WEIGHT_PATH)
            or original["source"]["dataset_sequence_sha256"] != sha256_file(source)
            or original["configuration"]["real_patches"] != 336):
        raise ValueError("Saved fixtures do not match frozen provenance")
    features = np.asarray([[int(v, 16) for v in line.split()] for line in
        (DEFAULT_OUTPUT / "two_layer_inputs.txt").read_text(encoding="ascii").splitlines()],
        dtype=np.uint8).reshape(-1, 4, 128)[:10 + real_patches]
    if real_patches:
        sequence = build_s0p_sequence(source, 4, windows=(0,))
        if not np.array_equal(features[10:], sequence.temporal_features[0].transpose(1, 0, 2)[:real_patches]):
            raise ValueError("Saved features differ from the HW-calibration window")
    cases = [np.zeros((4, 24), dtype=np.uint8), np.ones((4, 24), dtype=np.uint8)]
    for step in range(4):
        for index in range(24):
            case = np.zeros((4, 24), dtype=np.uint8)
            case[step, index] = 1
            cases.append(case)
    for positive in (True, False):
        for weights in exact.weights[2]:
            cases.append(np.tile(weights > 0 if positive else weights < 0, (4, 1)).astype(np.uint8))
    cases.extend(np.random.default_rng(seed).integers(0, 2, (4, 4, 24), dtype=np.uint8))
    binary = np.stack(cases)
    first, second, third, final, standalone_trace, standalone_final = [], [], [], [], [], []
    for model, k in ((exact, 0), (approximate, 2)):
        first.append(engine_trace(features, model, k))
        second.append(second_layer_trace((first[-1][..., 0] >> 24) & 1, model, k))
        trace, logits = readout_trace((second[-1][..., 0] >> 24) & 1, model, k)
        third.append(trace)
        final.append(logits)
        # Public forward is authoritative for final logits and inference saturation totals.
        reference = model.forward(features.transpose(1, 0, 2))
        if (not np.array_equal(logits[..., 0], reference.logits_q & 0xFFFFFFFF)
                or tuple(int(t[..., i].sum()) for t, i in ((first[-1], 1), (second[-1], 1), (trace, 2)))
                    != reference.accumulator_saturations
                or tuple(int(t[..., i].sum()) for t, i in ((first[-1], 2), (second[-1], 2), (trace, 3)))
                    != reference.membrane_saturations[:3]):
            raise AssertionError("Full Proposal forward/logits/saturation mismatch")
        trace, logits = readout_trace(binary, model, k)
        standalone_trace.append(trace)
        standalone_final.append(logits)
    output_directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, arrays, bases in (
        ("readout_inputs", [binary.reshape(len(binary), -1)], [16] * 96),
        ("proposal_inputs", [features.reshape(len(features), -1)], [16] * 512),
        ("readout_expected", standalone_final, [16, 10, 10] * 2),
        ("proposal_expected", final, [16, 10, 10] * 2),
        ("readout_trace", standalone_trace, [16, 16, 10, 10] * 2),
        ("proposal_trace", third, [16, 16, 10, 10] * 2),
        ("proposal_layer1", first, [16, 10, 10] * 2),
        ("proposal_layer2", second, [16, 10, 10] * 2),
    ):
        files[name] = write_records(output_directory / (name + ".txt"), np.concatenate(arrays, -1), bases)
    rounding = np.asarray([-(1 << 25), -(1 << 25) + 1, -15, -14, -10, -7, -6, -5, -3, -2, -1,
                           0, 1, 2, 3, 5, 6, 7, 10, 14, 15, (1 << 25) - 5, (1 << 25) - 4], dtype=np.int64)
    files["readout_rounding"] = write_records(output_directory / "readout_rounding.txt",
        np.stack((rounding & 0x3FFFFFF, signed_round_shift(rounding, 2) & 0xFFFFFFFF), -1), [16, 16])
    weight_files = dict(original["weight_files"])
    for entry in weight_files.values():
        src, dst = DEFAULT_OUTPUT / entry["path"], output_directory / entry["path"]
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    for layer in (0, 1, 2):
        for bank in range(8):
            path = output_directory / f"layer{layer + 1}_bank{bank}.txt"
            values = exact.weights[layer][:, bank::8].reshape(-1)
            if layer == 2:
                write_weight_file(path, values)
                weight_files[path.stem] = {"path": path.name, "values": len(values), "sha256": sha256_file(path)}
            if [decode_signed(v, 8) for v in path.read_text(encoding="ascii").split()] != values.tolist():
                raise ValueError("ROM differs from frozen weights")
        path = output_directory / f"layer{layer + 1}_shifts.txt"
        shifts = exact.weight_fraction_bits[layer] - 4 if layer == 0 else 12 - exact.weight_fraction_bits[layer]
        if layer == 2:
            atomic_text(path, (f"{v:X}\n" for v in shifts))
            weight_files[path.stem] = {"path": path.name, "values": len(shifts), "sha256": sha256_file(path)}
        if [int(v, 16) for v in path.read_text(encoding="ascii").split()] != shifts.tolist():
            raise ValueError("ROM scaling differs from frozen model")
    result = {
        "schema_version": 1, "status": "ok", "experiment_id": "RTL-PROPOSAL-VECTORS-v1",
        "stage": "P4-rtl-proposal-vectors", "data_role": "synthetic_and_hw_calibration_golden_vectors",
        "primary_metrics": {"standalone_patches": len(binary), "cascade_patches": len(features),
            "logit_records": (len(binary) + len(features)) * 16, "readout_step_records": (len(binary) + len(features)) * 64,
            "full_forward_checked_logits_per_mode": len(features) * 16},
        "decision": "ready_for_rtl_simulation", "conclusion": "Readout traces and final logits match frozen S1/S3, including public forward.",
        "configuration": {"standalone_patches": len(binary), "patches": len(features), "real_patches": real_patches,
            "complete_window": real_patches == 336, "window": 0, "synthetic_cascade_patches": 10,
            "binary_cases": {"zero": 1, "ones": 1, "single_step_input": 96, "channel_sign_extremes": 32, "random": 4},
            "seed": seed, "time_steps": 4, "channels": [48, 24, 16], "lanes": 8,
            "layer3_left_shifts": (12 - exact.weight_fraction_bits[2]).tolist(), "readout": "mean_membrane",
            "sum_bits": 26, "sum_addition": "exact_without_saturation", "rounding": "nearest_ties_away_from_zero",
            "output": "16 sign-extended int32 Q12 logits per patch in channel order",
            "diagnostic_output_spikes_in_rtl": False, "reset_between_patches": True},
        "files": files, "weight_files": weight_files,
        "source": {**original["source"], "two_layer_manifest_sha256": sha256_file(DEFAULT_OUTPUT / "two_layer_manifest.json")},
        "test_used": False, "validation_used": False,
    }
    atomic_text(manifest_path, [json.dumps(result, indent=2) + "\n"])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--real-patches", type=int, default=336)
    parser.add_argument("--seed", type=int, default=20260905)
    args = parser.parse_args()
    result = generate(args.output_dir.resolve(), args.real_patches, args.seed)
    print(json.dumps(result["primary_metrics"]))


if __name__ == "__main__":
    main()
