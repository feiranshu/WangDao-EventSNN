"""One evaluator: full-event micro counts and pinned EV-UAV localization rules."""
from __future__ import annotations

import numpy as np

WIDTH, HEIGHT = 346, 260
WINDOW_MS = 50
CORRECT_THRESHOLD = 0.0001
PROTOCOL_ID = "evuav-33e2aff-official-compatible-v1"


def ratio(numerator, denominator):
    # Undefined ratios remain null, never silently become a perfect score.
    return float(numerator / denominator) if denominator else None


def event_counts(labels, probabilities, threshold):
    y = np.asarray(labels)
    p = np.asarray(probabilities)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Expected equal, nonempty one-dimensional event arrays")
    if not np.all(np.isin(y, (0, 1))) or not np.isfinite(p).all():
        raise ValueError("Invalid labels or nonfinite probabilities")
    if np.any((p < 0) | (p > 1)) or not 0 < threshold < 1:
        raise ValueError("Probabilities must be in [0,1]; threshold must be in (0,1)")
    truth, pred = y == 1, p >= threshold
    return {"tp": int(np.count_nonzero(truth & pred)),
            "fp": int(np.count_nonzero(~truth & pred)),
            "fn": int(np.count_nonzero(truth & ~pred)),
            "tn": int(np.count_nonzero(~truth & ~pred))}


def event_summary(counts):
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    return {**counts, "iou": ratio(tp, tp + fp + fn),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "official_acc": ratio(tp, tp + fn), "f1": ratio(2 * tp, 2 * tp + fp + fn),
            "overall_accuracy": ratio(tp + tn, tp + fp + fn + tn)}


def connected_components_8(mask):
    """Count nonzero 8-connected components by flood fill (no OpenCV dependency)."""
    points = set(map(tuple, np.argwhere(mask != 0).tolist()))
    total = 0
    while points:
        total += 1
        stack = [points.pop()]
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (y + dy, x + dx)
                    if neighbor in points:
                        points.remove(neighbor)
                        stack.append(neighbor)
    return total


def localization_counts(loc, labels, identities, probabilities, threshold):
    loc = np.asarray(loc)
    labels, identities, probabilities = map(np.asarray, (labels, identities, probabilities))
    event_counts(labels, probabilities, threshold)
    if loc.shape != (len(labels), 3) or identities.shape != labels.shape:
        raise ValueError("Localization array shape mismatch")
    if not np.isfinite(loc).all() or not np.isfinite(identities).all():
        raise ValueError("Nonfinite localization input")
    # Official collate casts all ev_loc coordinates to int64, then test.py to float32.
    xyz = loc.astype(np.int64).astype(np.float32)
    if np.any(xyz[:, 0] < 0) or np.any(xyz[:, 0] >= WIDTH) or np.any(xyz[:, 1] < 0) or np.any(xyz[:, 1] >= HEIGHT):
        raise ValueError("Sensor coordinate outside bounds")
    if not np.all((labels == 0) == (identities == 0)):
        raise ValueError("Identity 0 must correspond exactly to background")
    ts = xyz[:, 2]
    duration = float(ts.max() - ts.min())
    totals = {"targets": 0, "correct_targets": 0, "false_components": 0,
              "frame_denominator": int(duration / WINDOW_MS), "scored_localization_events": 0,
              "wrapped_false_pixels": 0}
    pred = probabilities >= threshold
    for i in range(int(duration / WINDOW_MS + 1)):
        # Intentionally retain the official open boundaries, origin and final partial frame.
        member = (ts > i * WINDOW_MS) & (ts < (i + 1) * WINDOW_MS)
        ids, gt, selected = identities[member], labels[member], pred[member]
        totals["scored_localization_events"] += int(np.count_nonzero(member))
        for target_id in np.unique(ids):
            if target_id == 0:
                continue
            target = ids == target_id
            totals["targets"] += 1
            if np.count_nonzero(selected[target] == gt[target]) / gt[target].sum() >= CORRECT_THRESHOLD:
                totals["correct_targets"] += 1
        false_xy = xyz[member, :2][(gt == 0) & selected].astype(np.int64)
        if len(false_xy):
            counts = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
            np.add.at(counts, (false_xy[:, 1], false_xy[:, 0]), 1)
            totals["wrapped_false_pixels"] += int(np.count_nonzero(counts >= 256))
            # Compatibility with the source's uint8 counter, not a silently corrected mask.
            mask = (counts % 256).astype(np.uint8)
            totals["false_components"] += connected_components_8(mask)
    return totals


def localization_summary(counts):
    fa = ratio(counts["false_components"], counts["frame_denominator"] * WIDTH * HEIGHT)
    return {**counts, "pd": ratio(counts["correct_targets"], counts["targets"]),
            "fa": fa, "fa_times_1e4": None if fa is None else fa * 1e4}


def sum_counts(rows):
    if not rows:
        raise ValueError("No sequence results")
    return {key: sum(row[key] for row in rows) for key in rows[0]}
