"""Versioned engineering Pd/Fa; legacy evaluators remain frozen.

Use raw float64 millisecond timestamps and an explicit, closed-window exposure.
This module does not run a model or select a threshold.
"""
from __future__ import annotations

import math
import numpy as np

from .metrics import _component_count


PROTOCOL_ID = "evuav-localization-half-open-v2"
WINDOW_MS = 50.0
CORRECT_THRESHOLD = 0.0001
WIDTH, HEIGHT = 346, 260
COUNT_KEYS = ("targets", "correct_targets", "false_components", "frame_denominator",
              "scored_localization_events", "excluded_events")


def localization_counts(timestamps_ms, x, y, labels, identities, probabilities,
                        threshold, *, start_ms, end_ms):
    """Count [start_ms, end_ms), including empty windows in Fa exposure.

    The common 100 ms prefix must be supplied by the caller. Input outside it
    is counted as excluded, never silently used to extend the denominator.
    Pd retains the declared 1e-4 target-event fraction; it is a permissive
    legacy-style detection diagnostic, not a box-localization criterion.
    """
    arrays = [np.asarray(v) for v in (timestamps_ms, x, y, labels, identities, probabilities)]
    t, x, y, labels, identities, probabilities = arrays
    if any(a.ndim != 1 or a.shape != t.shape for a in arrays):
        raise ValueError("Expected equal one-dimensional localization arrays")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Localization input must be finite")
    if not math.isfinite(float(threshold)) or not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    if not math.isfinite(start_ms) or not math.isfinite(end_ms) or end_ms <= start_ms:
        raise ValueError("An explicit positive exposure is required")
    windows = (end_ms - start_ms) / WINDOW_MS
    if not math.isclose(windows, round(windows), abs_tol=1e-9, rel_tol=0):
        raise ValueError("Exposure must contain complete 50 ms windows")
    if (np.any((x < 0) | (x >= WIDTH) | (x != np.floor(x))) or
            np.any((y < 0) | (y >= HEIGHT) | (y != np.floor(y)))):
        raise ValueError("Invalid sensor coordinates")
    if (not np.isin(labels, (0, 1)).all() or
            np.any((identities < 0) | (identities != np.floor(identities))) or
            not np.array_equal(labels == 0, identities == 0)):
        raise ValueError("Binary labels and nonnegative target identities must agree")
    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be in [0, 1]")
    selected = (t >= start_ms) & (t < end_ms)
    totals = dict.fromkeys(COUNT_KEYS, 0)
    totals.update(frame_denominator=int(round(windows)),
                  scored_localization_events=int(selected.sum()),
                  excluded_events=int((~selected).sum()))
    if not np.any(selected):
        return totals
    edges = start_ms + np.arange(int(round(windows)) + 1, dtype=np.float64) * WINDOW_MS
    frame = np.searchsorted(edges, t[selected].astype(np.float64), side="right") - 1
    sx, sy, truth, ids, pred = (a[selected] for a in (x, y, labels, identities, probabilities))
    pred = pred >= threshold
    for frame_id in np.unique(frame):
        mask = frame == frame_id
        for identity in np.unique(ids[mask & (truth == 1)]):
            target = mask & (ids == identity)
            totals["targets"] += 1
            totals["correct_targets"] += int(np.count_nonzero(pred[target]) /
                                               np.count_nonzero(target) >= CORRECT_THRESHOLD)
        false = mask & (truth == 0) & pred
        occupancy = np.zeros((HEIGHT, WIDTH), dtype=bool)
        occupancy[sy[false].astype(np.int64), sx[false].astype(np.int64)] = True
        totals["false_components"] += _component_count(occupancy)
    return totals


def summarize_counts(rows):
    """Pool integer counts; undefined Pd/Fa is null, never a perfect score."""
    totals = dict.fromkeys(COUNT_KEYS, 0)
    for row in rows:
        for key in COUNT_KEYS:
            value = row[key]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
                raise ValueError("Localization counts must be nonnegative integers")
            totals[key] += int(value)
    if totals["correct_targets"] > totals["targets"]:
        raise ValueError("Detected targets cannot exceed target windows")
    return {**totals,
            "pd": totals["correct_targets"] / totals["targets"] if totals["targets"] else None,
            "fa": totals["false_components"] / (totals["frame_denominator"] * WIDTH * HEIGHT)
                  if totals["frame_denominator"] else None}
