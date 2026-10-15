"""Full-event error attribution; ground truth is used here only after inference."""
from __future__ import annotations

import numpy as np


def fraction(numerator: int, denominator: int):
    return float(numerator / denominator) if denominator else None


def error_counts(labels, probabilities, candidate_mask, threshold=0.35) -> dict:
    labels, probabilities, candidate = np.asarray(labels), np.asarray(probabilities), np.asarray(candidate_mask)
    if labels.ndim != 1 or labels.shape != probabilities.shape or labels.shape != candidate.shape:
        raise ValueError("Labels, probabilities and candidate mask must have equal vector shapes")
    if not np.isin(labels, [0, 1]).all() or candidate.dtype != np.bool_:
        raise ValueError("Binary labels and a boolean candidate mask are required")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be finite and in [0,1]")
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Threshold must lie in (0,1)")
    if np.any(probabilities[~candidate] != 0):
        raise ValueError("Rejected events must remain background in the full-event metric")
    positive = labels == 1
    predicted = probabilities >= threshold
    return {"events": len(labels), "target_events": int(positive.sum()),
            "candidate_events": int(candidate.sum()), "candidate_targets": int((candidate & positive).sum()),
            "proposal_fn": int((~candidate & positive).sum()),
            "refiner_fn": int((candidate & positive & ~predicted).sum()),
            "tp": int((positive & predicted).sum()), "fp": int((~positive & predicted).sum()),
            "fn": int((positive & ~predicted).sum()), "tn": int((~positive & ~predicted).sum())}


def summarize_counts(counts: dict) -> dict:
    result = dict(counts)
    result.update(iou=fraction(counts["tp"], counts["tp"] + counts["fp"] + counts["fn"]),
                  precision=fraction(counts["tp"], counts["tp"] + counts["fp"]),
                  recall=fraction(counts["tp"], counts["target_events"]),
                  proposal_target_recall=fraction(counts["candidate_targets"], counts["target_events"]),
                  refiner_conditional_recall=fraction(counts["tp"], counts["candidate_targets"]),
                  candidate_fraction=fraction(counts["candidate_events"], counts["events"]))
    if counts["proposal_fn"] + counts["refiner_fn"] != counts["fn"]:
        raise ValueError("False-negative attribution does not partition the full-event count")
    return result


def aggregate_counts(rows: list) -> dict:
    keys = error_counts(np.array([], dtype=int), np.array([]), np.array([], dtype=bool)).keys()
    return summarize_counts({key: sum(int(row[key]) for row in rows) for key in keys})


def feature_collisions(features: np.ndarray, labels: np.ndarray) -> dict:
    if features.ndim != 2 or len(features) != len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid feature/label arrays")
    if not len(features):
        return {"rows": 0, "unique_codes": 0, "conflicting_codes": 0, "events_in_conflicting_codes": 0}
    _, inverse = np.unique(features, axis=0, return_inverse=True)
    total = np.bincount(inverse)
    positives = np.bincount(inverse, weights=labels)
    conflict = (positives > 0) & (positives < total)
    return {"rows": len(features), "unique_codes": len(total), "conflicting_codes": int(conflict.sum()),
            "events_in_conflicting_codes": int(total[conflict].sum())}


def temporal_support(events, labels, t0: float, end_ms: float) -> dict:
    """Descriptive label statistics; never derive selection cutoffs from Train-dev."""
    result = {}
    for width in (10.0, 100.0):
        bins = int(np.floor((end_ms-t0) / width + 1e-9))
        times = np.asarray(events["t"])[np.asarray(labels) == 1]
        indices = np.floor((times-t0)/width).astype(np.int64)
        counts = np.bincount(indices[(indices >= 0) & (indices < bins)], minlength=bins)
        occupied = counts[counts > 0]
        result[str(int(width)) + "ms"] = {
            "complete_bins": bins, "target_occupied_bins": int(len(occupied)),
            "positive_events_per_occupied_bin_quantiles":
                np.quantile(occupied, [0, 0.25, 0.5, 0.75, 1]).tolist() if len(occupied) else None}
    return result
