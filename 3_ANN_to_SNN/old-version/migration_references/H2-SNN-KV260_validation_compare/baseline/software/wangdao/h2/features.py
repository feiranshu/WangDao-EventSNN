"""Label-free 28-dimensional event features used by the H2 refiner."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .constants import EVENT_FEATURE_DIM, EVENT_FEATURE_SPECS, SENSOR_HEIGHT, SENSOR_WIDTH


NEIGHBOR_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))


def _lookup_counts(
    unique_keys: np.ndarray,
    counts: np.ndarray,
    time_bin: np.ndarray,
    x_cell: np.ndarray,
    y_cell: np.ndarray,
    polarity: np.ndarray,
    nx: int,
    ny: int,
) -> np.ndarray:
    valid = (
        (time_bin >= 0)
        & (x_cell >= 0)
        & (x_cell < nx)
        & (y_cell >= 0)
        & (y_cell < ny)
        & (polarity >= 0)
        & (polarity <= 1)
    )
    query = (((time_bin * ny + y_cell) * nx + x_cell) * 2 + polarity).astype(np.int64)
    positions = np.searchsorted(unique_keys, query)
    safe = np.minimum(positions, max(len(unique_keys) - 1, 0))
    matched = valid & (positions < len(unique_keys))
    if len(unique_keys):
        matched &= unique_keys[safe] == query
    result = np.zeros(query.shape, dtype=np.int64)
    if len(unique_keys):
        result[matched] = counts[safe[matched]]
    return result


def compress_counts(values: np.ndarray) -> np.ndarray:
    compressed = np.rint(32.0 * np.log1p(values.astype(np.float64)))
    return np.clip(compressed, 0.0, 255.0).astype(np.uint8)


def extract_event_features(events: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return uint8 event features and the legacy candidate-density score."""
    required = ("x", "y", "t", "p")
    if events.dtype.names is None or any(name not in events.dtype.names for name in required):
        raise ValueError("events must contain x/y/t/p fields")
    count = len(events)
    if count == 0:
        return (
            np.empty((0, EVENT_FEATURE_DIM), dtype=np.uint8),
            np.empty((0,), dtype=np.uint16),
        )
    x = events["x"].astype(np.int64, copy=False)
    y = events["y"].astype(np.int64, copy=False)
    timestamp = events["t"].astype(np.float64, copy=False)
    polarity = events["p"].astype(np.int64, copy=False)
    origin = float(timestamp[0])
    duration = max(float(timestamp[-1] - origin), 1.0)
    features = np.empty((count, EVENT_FEATURE_DIM), dtype=np.uint8)
    features[:, 0] = np.rint(255.0 * x / float(SENSOR_WIDTH - 1)).astype(np.uint8)
    features[:, 1] = np.rint(255.0 * y / float(SENSOR_HEIGHT - 1)).astype(np.uint8)
    features[:, 2] = np.rint(255.0 * (timestamp - origin) / duration).astype(np.uint8)
    features[:, 3] = (polarity * 255).astype(np.uint8)
    candidate_score = np.zeros(count, dtype=np.int64)
    column = 4
    for spec_index, (spatial_cell, temporal_cell_ms) in enumerate(EVENT_FEATURE_SPECS):
        nx = (SENSOR_WIDTH + spatial_cell - 1) // spatial_cell
        ny = (SENSOR_HEIGHT + spatial_cell - 1) // spatial_cell
        x_cell = x // spatial_cell
        y_cell = y // spatial_cell
        time_bin = np.floor((timestamp - origin) / temporal_cell_ms).astype(np.int64)
        keys = (((time_bin * ny + y_cell) * nx + x_cell) * 2 + polarity).astype(np.int64)
        unique_keys, inverse, unique_counts = np.unique(
            keys, return_inverse=True, return_counts=True
        )
        unique_counts = unique_counts.astype(np.int64, copy=False)
        current_same = unique_counts[inverse]
        current_other = _lookup_counts(
            unique_keys, unique_counts, time_bin, x_cell, y_cell, 1 - polarity, nx, ny
        )
        lag_values = []
        for lag in (1, 2):
            same = np.zeros(count, dtype=np.int64)
            other = np.zeros(count, dtype=np.int64)
            for dx, dy in NEIGHBOR_OFFSETS:
                same += _lookup_counts(
                    unique_keys, unique_counts, time_bin - lag, x_cell + dx, y_cell + dy,
                    polarity, nx, ny,
                )
                other += _lookup_counts(
                    unique_keys, unique_counts, time_bin - lag, x_cell + dx, y_cell + dy,
                    1 - polarity, nx, ny,
                )
            lag_values.extend((same, other))
        raw_columns = (current_same, current_other, *lag_values)
        for values in raw_columns:
            features[:, column] = compress_counts(values)
            column += 1
        if spec_index == 1:
            candidate_score = sum(raw_columns)
    if column != EVENT_FEATURE_DIM:
        raise AssertionError("event feature dimension contract is inconsistent")
    return features, np.clip(candidate_score, 0, 65535).astype(np.uint16)
