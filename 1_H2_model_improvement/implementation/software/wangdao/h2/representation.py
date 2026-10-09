"""H2 window patches, routing masks and event-refiner context."""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .constants import (
    EVENT_FEATURE_DIM,
    FEATURE_CELL,
    OUTPUT_CELL,
    PADDED_HEIGHT,
    PADDED_WIDTH,
    PATCH_COLUMNS,
    PATCH_FEATURE_DIM,
    PATCH_OUTPUT_DIM,
    PATCH_ROWS,
    PATCH_SIZE,
    PATCH_STRIDE,
    PATCHES_PER_WINDOW,
    REFINER_FEATURE_DIM,
    WINDOW_MS,
    WINDOWS_PER_SEQUENCE,
)
from .features import compress_counts, extract_event_features


def window_indices(events: np.ndarray) -> np.ndarray:
    if len(events) == 0:
        return np.empty((0,), dtype=np.int16)
    origin = float(events["t"][0])
    values = np.floor((events["t"].astype(np.float64) - origin) / WINDOW_MS)
    return np.clip(values, 0, WINDOWS_PER_SEQUENCE - 1).astype(np.int16)


def selected_windows(start_ms: float, duration_ms: float) -> np.ndarray:
    if duration_ms <= 0.0 or not np.isclose(duration_ms % WINDOW_MS, 0.0):
        raise ValueError("duration must be a positive multiple of 100 ms")
    if start_ms < 0.0 or not np.isclose(start_ms % WINDOW_MS, 0.0):
        raise ValueError("start must be a non-negative multiple of 100 ms")
    first = int(round(start_ms / WINDOW_MS))
    count = int(round(duration_ms / WINDOW_MS))
    if first + count > WINDOWS_PER_SEQUENCE:
        raise ValueError("requested range exceeds the 8-second sequence")
    return np.arange(first, first + count, dtype=np.int16)


def _local_window_indices(
    assigned: np.ndarray, chosen: np.ndarray
) -> Optional[np.ndarray]:
    """Map global windows to output rows when the selection has no duplicates."""
    if len(np.unique(chosen)) != len(chosen):
        return None
    lookup = np.full(WINDOWS_PER_SEQUENCE, -1, dtype=np.int16)
    valid = (chosen >= 0) & (chosen < WINDOWS_PER_SEQUENCE)
    lookup[chosen[valid]] = np.flatnonzero(valid).astype(np.int16)
    return lookup[assigned]


def _fine_counts_to_patch_features(fine: np.ndarray) -> np.ndarray:
    """Convert `[..., polarity, y, x]` cell counts to packed patch features."""
    cell_patch = PATCH_SIZE // FEATURE_CELL
    cell_stride = PATCH_STRIDE // FEATURE_CELL
    windows = sliding_window_view(
        fine, (cell_patch, cell_patch), axis=(-2, -1)
    )
    windows = windows[..., ::cell_stride, ::cell_stride, :, :]
    leading_dimensions = windows.ndim - 5
    order = (
        *range(leading_dimensions),
        leading_dimensions + 1,
        leading_dimensions + 2,
        leading_dimensions,
        leading_dimensions + 3,
        leading_dimensions + 4,
    )
    packed = windows.transpose(order).reshape(
        *fine.shape[:-3], PATCHES_PER_WINDOW, PATCH_FEATURE_DIM
    )
    return np.ascontiguousarray(compress_counts(packed))


def _accumulate_patch_counts(
    events: np.ndarray,
    chosen: np.ndarray,
    assigned: np.ndarray,
) -> np.ndarray:
    height = PADDED_HEIGHT // FEATURE_CELL
    width = PADDED_WIDTH // FEATURE_CELL
    fine = np.zeros((len(chosen), 2, height, width), dtype=np.int32)
    x_cell = events["x"].astype(np.int64, copy=False) // FEATURE_CELL
    y_cell = events["y"].astype(np.int64, copy=False) // FEATURE_CELL
    polarity = events["p"].astype(np.int64, copy=False)
    local = _local_window_indices(assigned, chosen)
    if local is not None:
        member = local >= 0
        np.add.at(
            fine,
            (local[member], polarity[member], y_cell[member], x_cell[member]),
            1,
        )
        return fine

    # Duplicate window requests are unusual but remain supported exactly.
    for output_window, window in enumerate(chosen):
        window_member = assigned == int(window)
        np.add.at(
            fine[output_window],
            (
                polarity[window_member],
                y_cell[window_member],
                x_cell[window_member],
            ),
            1,
        )
    return fine


def _validate_patch_events(events: np.ndarray) -> None:
    required = ("x", "y", "t", "p")
    if events.dtype.names is None or any(name not in events.dtype.names for name in required):
        raise ValueError("events must contain x/y/t/p fields")


def build_patch_features(events: np.ndarray, windows: Sequence[int]) -> np.ndarray:
    """Build `[window, 336, 128]` uint8 proposal features."""
    _validate_patch_events(events)
    chosen = np.asarray(windows, dtype=np.int16)
    assigned = window_indices(events)
    fine = _accumulate_patch_counts(events, chosen, assigned)
    return _fine_counts_to_patch_features(fine)


def build_patch_targets(events: np.ndarray, windows: Sequence[int]) -> np.ndarray:
    """Build binary `[window, 336, 16]` targets for proposal training."""
    if events.dtype.names is None or "label" not in events.dtype.names:
        raise ValueError("proposal targets require the label field")
    chosen = np.asarray(windows, dtype=np.int16)
    assigned = window_indices(events)
    x = events["x"].astype(np.int64, copy=False)
    y = events["y"].astype(np.int64, copy=False)
    labels = events["label"].astype(bool, copy=False)
    coarse_shape = (PADDED_HEIGHT // OUTPUT_CELL, PADDED_WIDTH // OUTPUT_CELL)
    coarse = np.zeros((len(chosen), *coarse_shape), dtype=np.uint8)
    local = _local_window_indices(assigned, chosen)
    if local is not None:
        target = labels & (local >= 0)
        coarse[local[target], y[target] // OUTPUT_CELL, x[target] // OUTPUT_CELL] = 1
    else:
        for output_window, window in enumerate(chosen):
            target = (assigned == int(window)) & labels
            coarse[
                output_window, y[target] // OUTPUT_CELL, x[target] // OUTPUT_CELL
            ] = 1
    output_patch = PATCH_SIZE // OUTPUT_CELL
    output_stride = PATCH_STRIDE // OUTPUT_CELL
    patches = sliding_window_view(
        coarse, (output_patch, output_patch), axis=(-2, -1)
    )
    patches = patches[..., ::output_stride, ::output_stride, :, :]
    return np.ascontiguousarray(
        patches.reshape(len(chosen), PATCHES_PER_WINDOW, PATCH_OUTPUT_DIM)
    )


def patch_scores_to_heatmaps(patch_scores: np.ndarray) -> np.ndarray:
    if patch_scores.ndim != 3 or patch_scores.shape[1:] != (
        PATCHES_PER_WINDOW, PATCH_OUTPUT_DIM
    ):
        raise ValueError("proposal score shape mismatch")
    heatmaps = np.zeros(
        (len(patch_scores), PADDED_HEIGHT // OUTPUT_CELL, PADDED_WIDTH // OUTPUT_CELL),
        dtype=np.float32,
    )
    for window in range(len(patch_scores)):
        row = 0
        for patch_y in range(PATCH_ROWS):
            y0 = patch_y * PATCH_STRIDE // OUTPUT_CELL
            for patch_x in range(PATCH_COLUMNS):
                x0 = patch_x * PATCH_STRIDE // OUTPUT_CELL
                destination = heatmaps[window, y0:y0 + 4, x0:x0 + 4]
                np.maximum(destination, patch_scores[window, row].reshape(4, 4), out=destination)
                row += 1
    return heatmaps


def dilate_cells(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 3:
        raise ValueError("cell mask must have shape [window, height, width]")
    padded = np.pad(mask, ((0, 0), (1, 1), (1, 1)), mode="constant")
    result = np.zeros_like(mask)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            result |= padded[
                :, 1 + dy:1 + dy + mask.shape[1], 1 + dx:1 + dx + mask.shape[2]
            ]
    return result


def _component_sizes(mask: np.ndarray) -> np.ndarray:
    result = np.zeros(mask.shape, dtype=np.uint16)
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for start_y, start_x in np.argwhere(mask):
        sy, sx = int(start_y), int(start_x)
        if visited[sy, sx]:
            continue
        queue = deque([(sy, sx)])
        visited[sy, sx] = True
        component = []
        while queue:
            cy, cx = queue.popleft()
            component.append((cy, cx))
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
        size = min(len(component), 65535)
        for cy, cx in component:
            result[cy, cx] = size
    return result


def candidate_indices(
    events: np.ndarray,
    windows: Sequence[int],
    heatmaps: np.ndarray,
    threshold: float,
) -> np.ndarray:
    chosen = np.asarray(windows, dtype=np.int16)
    assigned = window_indices(events)
    in_range = np.isin(assigned, chosen)
    event_indices = np.flatnonzero(in_range).astype(np.int64)
    lookup = {int(window): index for index, window in enumerate(chosen)}
    local_window = np.asarray(
        [lookup[int(assigned[index])] for index in event_indices], dtype=np.int64
    )
    x_cell = events["x"][event_indices].astype(np.int64) // OUTPUT_CELL
    y_cell = events["y"][event_indices].astype(np.int64) // OUTPUT_CELL
    mask = dilate_cells(heatmaps >= float(threshold))
    return event_indices[mask[local_window, y_cell, x_cell]]


def build_refiner_features(
    events: np.ndarray,
    event_indices: np.ndarray,
    windows: Sequence[int],
    heatmaps: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Build H2's 28 base plus 4 proposal-context features."""
    selected = np.asarray(event_indices, dtype=np.int64)
    if len(selected) == 0:
        return np.empty((0, REFINER_FEATURE_DIM), dtype=np.uint8)
    chosen = np.asarray(windows, dtype=np.int16)
    base, _ = extract_event_features(events)
    assigned = window_indices(events).astype(np.int64)
    lookup = {int(window): index for index, window in enumerate(chosen)}
    local_window = np.asarray(
        [lookup[int(assigned[index])] for index in selected], dtype=np.int64
    )
    x_cell = events["x"][selected].astype(np.int64) // OUTPUT_CELL
    y_cell = events["y"][selected].astype(np.int64) // OUTPUT_CELL
    cell_mask = dilate_cells(heatmaps >= float(threshold))
    padded = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge")
    neighbor = np.zeros_like(heatmaps)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbor += padded[
                :, 1 + dy:1 + dy + heatmaps.shape[1],
                1 + dx:1 + dx + heatmaps.shape[2],
            ] / 9.0
    sizes = np.stack([_component_sizes(mask) for mask in cell_mask], axis=0)
    origin = float(events["t"][0])
    within = np.mod(
        events["t"][selected].astype(np.float64) - origin, WINDOW_MS
    ) / WINDOW_MS
    context = np.column_stack(
        (
            np.rint(255.0 * heatmaps[local_window, y_cell, x_cell]),
            np.rint(255.0 * neighbor[local_window, y_cell, x_cell]),
            np.rint(
                255.0
                * np.minimum(sizes[local_window, y_cell, x_cell].astype(np.float64) / 64.0, 1.0)
            ),
            np.rint(255.0 * np.clip(within, 0.0, 1.0)),
        )
    ).astype(np.uint8)
    result = np.concatenate((base[selected], context), axis=1)
    if result.shape[1] != EVENT_FEATURE_DIM + 4:
        raise AssertionError("H2 refiner feature dimension mismatch")
    return np.ascontiguousarray(result, dtype=np.uint8)
