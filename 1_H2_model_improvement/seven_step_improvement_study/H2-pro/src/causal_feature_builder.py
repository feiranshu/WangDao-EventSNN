"""Causal counterpart of the frozen 28 event columns, with an explicit origin."""
from __future__ import annotations

import numpy as np
from wangdao.h2.constants import EVENT_FEATURE_SPECS, SENSOR_HEIGHT, SENSOR_WIDTH
from wangdao.h2.features import NEIGHBOR_OFFSETS, _lookup_counts, compress_counts


def base_features(history_and_window: np.ndarray, window_start: int, t0: float) -> np.ndarray:
    """Calculate old columns 0,1,3:28 for a closed window using only past and current events.

    The caller retains the previous 100 ms of events. Maximum old lag is two
    50 ms bins, so no older event can affect the current window.
    """
    events = history_and_window
    n = len(events)
    out = np.zeros((n - window_start, 28), dtype=np.uint8)
    if not len(out):
        return out
    x = events['x'].astype(np.int64)
    y = events['y'].astype(np.int64)
    p = events['p'].astype(np.int64)
    t = events['t'].astype(np.float64)
    out[:, 0] = np.rint(255 * x[window_start:] / (SENSOR_WIDTH - 1)).astype(np.uint8)
    out[:, 1] = np.rint(255 * y[window_start:] / (SENSOR_HEIGHT - 1)).astype(np.uint8)
    out[:, 3] = (255 * p[window_start:]).astype(np.uint8)
    col = 4
    for cell, ms in EVENT_FEATURE_SPECS:
        nx = (SENSOR_WIDTH + cell - 1) // cell
        ny = (SENSOR_HEIGHT + cell - 1) // cell
        xb, yb = x // cell, y // cell
        tb = np.floor((t - t0) / ms).astype(np.int64)
        keys = (((tb * ny + yb) * nx + xb) * 2 + p).astype(np.int64)
        unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        raw = [counts[inverse][window_start:],
               _lookup_counts(unique, counts, tb, xb, yb, 1-p, nx, ny)[window_start:]]
        for lag in (1, 2):
            for polarity in (p, 1-p):
                value = np.zeros(n, dtype=np.int64)
                for dx, dy in NEIGHBOR_OFFSETS:
                    value += _lookup_counts(unique, counts, tb-lag, xb+dx, yb+dy,
                                            polarity, nx, ny)
                raw.append(value[window_start:])
        for value in raw:
            out[:, col] = compress_counts(value)
            col += 1
    return out


def neighbor_mean(heatmap: np.ndarray) -> np.ndarray:
    padded = np.pad(heatmap, ((1, 1), (1, 1)), mode='edge')
    mean = np.zeros_like(heatmap)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            mean += padded[1+dy:1+dy+34, 1+dx:1+dx+44] / 9.0
    return mean
