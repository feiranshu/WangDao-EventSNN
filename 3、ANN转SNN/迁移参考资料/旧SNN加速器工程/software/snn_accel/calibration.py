from __future__ import annotations

import numpy as np


def event_count_candidates(window_counts: np.ndarray) -> dict[str, int]:
    counts = np.asarray(window_counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0 or np.any(counts < 0):
        raise ValueError("window_counts must be a non-empty vector of non-negative counts")
    quantiles = np.percentile(counts, [25, 50, 75], method="nearest")
    return {name: max(1, int(value)) for name, value in zip(("p25", "p50", "p75"), quantiles)}


def choose_density_hysteresis(event_counts: np.ndarray) -> tuple[int, int]:
    """Return deterministic initial thresholds; final thresholds require validation metrics."""

    counts = np.asarray(event_counts, dtype=np.int64)
    if counts.ndim != 1 or counts.size == 0:
        raise ValueError("event_counts must be a non-empty vector")
    low, high = np.percentile(counts, [25, 75], method="nearest")
    if low == high:
        high = low + 1
    return int(low), int(high)

