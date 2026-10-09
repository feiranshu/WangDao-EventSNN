"""Export one real EV-UAV Proposal patch for the frozen KV260 SNN."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_MODEL,
    add_software_path,
    all_time_to_lane_orders,
    assert_known_patch_baseline,
    load_model,
    order_name,
    pack_temporal_features,
    parse_time_to_lane,
    sha256_file,
    signed_to_u32,
    write_hex_words,
    write_json,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_DATA_ROOT / "val" / "val_000.npz",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--window-index", type=int, default=0)
    parser.add_argument(
        "--patch-index",
        default="auto",
        help="0..335, or auto to select an active positive patch",
    )
    parser.add_argument("--sample-id", default="sample_000")
    parser.add_argument(
        "--time-to-lane",
        help="confirmed permutation such as 0123; omit to emit all 24 candidates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "real_samples" / "sample_000",
    )
    return parser.parse_args()


def choose_patch(
    temporal: np.ndarray, targets: np.ndarray, requested: str
) -> tuple[int, str]:
    if requested != "auto":
        index = int(requested)
        if not 0 <= index < temporal.shape[1]:
            raise ValueError(f"patch index must be in [0, {temporal.shape[1] - 1}]")
        return index, "explicit"

    activity = temporal.astype(np.uint64).sum(axis=(0, 2))
    positives = np.flatnonzero(targets.sum(axis=1) > 0)
    if len(positives):
        index = int(positives[np.argmax(activity[positives])])
        return index, "maximum_activity_among_positive_targets"
    index = int(np.argmax(activity))
    return index, "maximum_activity_no_positive_target_available"


def main() -> int:
    arguments = parse_arguments()
    events_path = arguments.events.resolve()
    model_path = arguments.model.resolve()
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    add_software_path()
    from wangdao.data import load_events
    from wangdao.h2.representation import (
        build_patch_targets,
        build_temporal_patch_features,
    )
    from wangdao.h2.constants import (
        FEATURE_CELL,
        OUTPUT_CELL,
        PADDED_HEIGHT,
        PADDED_WIDTH,
        PATCH_COLUMNS,
        PATCH_ROWS,
        PATCH_SIZE,
        PATCH_STRIDE,
    )

    model, configuration, model_metadata = load_model(model_path)
    assert_known_patch_baseline(model)
    if model.time_steps != 4:
        raise RuntimeError(f"frozen board protocol expects 4 steps, model has {model.time_steps}")

    events = load_events(events_path, require_labels=True)
    windows = np.asarray([arguments.window_index], dtype=np.int16)
    temporal_all = build_temporal_patch_features(events, windows, model.time_steps)[0]
    targets_all = build_patch_targets(events, windows)[0]
    patch_index, selection_rule = choose_patch(
        temporal_all, targets_all, arguments.patch_index
    )
    temporal = np.ascontiguousarray(temporal_all[:, patch_index, :], dtype=np.uint8)
    target_cells = np.ascontiguousarray(targets_all[patch_index], dtype=np.uint8)
    output_values = model.forward(temporal[:, None, :])
    logits_q = np.ascontiguousarray(output_values.logits_q[0], dtype=np.int64)
    logits_words = signed_to_u32(logits_q)

    temporal_path = output / f"{arguments.sample_id}_temporal_uint8.npy"
    target_path = output / f"{arguments.sample_id}_target_cells.npy"
    expected_path = output / f"{arguments.sample_id}_expected_logits.txt"
    np.save(temporal_path, temporal, allow_pickle=False)
    np.save(target_path, target_cells, allow_pickle=False)
    write_hex_words(expected_path, logits_words)

    candidates: dict[str, dict[str, str]] = {}
    confirmed_order: str | None = None
    canonical_input: Path | None = None
    if arguments.time_to_lane:
        order = parse_time_to_lane(arguments.time_to_lane)
        confirmed_order = order_name(order)
        canonical_input = output / f"{arguments.sample_id}_input_words.txt"
        write_hex_words(canonical_input, pack_temporal_features(temporal, order))
    else:
        candidate_dir = output / "input_candidates"
        for order in all_time_to_lane_orders():
            name = order_name(order)
            path = candidate_dir / f"{arguments.sample_id}_input_words_order_{name}.txt"
            write_hex_words(path, pack_temporal_features(temporal, order))
            candidates[name] = {
                "file": path.relative_to(output).as_posix(),
                "sha256": sha256_file(path),
            }

    target_indices = np.flatnonzero(target_cells).astype(int).tolist()
    patch_row = patch_index // PATCH_COLUMNS
    patch_column = patch_index % PATCH_COLUMNS
    meta = {
        "format_version": 1,
        "sample_id": arguments.sample_id,
        "source_sequence": str(events_path),
        "source_sequence_sha256": sha256_file(events_path),
        "window_index": int(arguments.window_index),
        "window_duration_ms": 100,
        "time_steps": 4,
        "time_step_duration_ms": 25,
        "patch_index": patch_index,
        "patch_selection_rule": selection_rule,
        "feature_shape": [4, 128],
        "feature_dtype": "uint8",
        "feature_format": "unsigned uint8 compressed event counts used as Q0.8 input",
        "feature_encoding": {
            "sensor_size_xy": [346, 260],
            "padded_size_xy": [PADDED_WIDTH, PADDED_HEIGHT],
            "patch_size_xy": [PATCH_SIZE, PATCH_SIZE],
            "patch_stride_xy": [PATCH_STRIDE, PATCH_STRIDE],
            "patch_grid_rows_columns": [PATCH_ROWS, PATCH_COLUMNS],
            "patch_row_column": [patch_row, patch_column],
            "patch_origin_xy": [
                patch_column * PATCH_STRIDE,
                patch_row * PATCH_STRIDE,
            ],
            "feature_cell_size_xy": [FEATURE_CELL, FEATURE_CELL],
            "feature_grid_per_patch": [
                PATCH_SIZE // FEATURE_CELL,
                PATCH_SIZE // FEATURE_CELL,
            ],
            "polarity_count": 2,
            "flatten_order": "polarity-major, then cell_y, then cell_x",
            "raw_value": "event count in one polarity/cell/25ms time step",
            "compression": "clip(round(32 * ln(1 + raw_count)), 0, 255)",
        },
        "feature_nonzero": int(np.count_nonzero(temporal)),
        "feature_sum": int(temporal.astype(np.uint64).sum()),
        "target_cells_4x4_flat": target_cells.astype(int).tolist(),
        "positive_target_indices": target_indices,
        "word_count": 128,
        "time_to_lane": confirmed_order,
        "time_to_lane_definition": (
            "digit at index t is the physical byte lane containing software time t; "
            "lane0=bits[7:0], lane3=bits[31:24]"
        ),
        "input_words": canonical_input.name if canonical_input else None,
        "input_words_sha256": sha256_file(canonical_input)
        if canonical_input
        else None,
        "input_candidates": candidates,
        "golden_model": str(model_path),
        "golden_model_sha256": sha256_file(model_path),
        "golden_model_configuration": configuration,
        "golden_model_metadata": model_metadata,
        "logit_format": "signed int32 Q12 represented as uint32 hexadecimal",
        "logit_shape": [4, 4],
        "logit_flatten_order": "output_cell_y-major, then output_cell_x",
        "output_cell_size_xy": [OUTPUT_CELL, OUTPUT_CELL],
        "logits_q_signed": logits_q.astype(int).tolist(),
        "logits_float": output_values.logits[0].astype(float).tolist(),
        "expected_logits": expected_path.name,
        "expected_logits_sha256": sha256_file(expected_path),
        "temporal_npy": temporal_path.name,
        "temporal_npy_sha256": sha256_file(temporal_path),
        "target_cells_npy": target_path.name,
        "target_cells_npy_sha256": sha256_file(target_path),
        "board_ready": confirmed_order is not None,
    }
    meta_path = output / f"{arguments.sample_id}_meta.json"
    write_json(meta_path, meta)

    print(f"sample: {arguments.sample_id}")
    print(f"events: {events_path}")
    print(f"window: {arguments.window_index}")
    print(f"patch: {patch_index} ({selection_rule})")
    print(f"feature nonzero: {meta['feature_nonzero']}")
    print(f"positive target indices: {target_indices}")
    print(f"expected logits: {expected_path}")
    if canonical_input:
        print(f"input words: {canonical_input}")
        print(f"time_to_lane: {confirmed_order}")
        print("BOARD_READY: yes")
    else:
        print(f"input candidates: {len(candidates)}")
        print("BOARD_READY: no (run lane probe and rerun with --time-to-lane)")
    print(f"metadata: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
