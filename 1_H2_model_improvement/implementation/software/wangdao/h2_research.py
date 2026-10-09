"""H2-first diagnostics and accuracy experiments for plan steps 1-3.

The frozen H2 Proposal remains unchanged.  Labels are consumed only by the
training and scoring callers; every feature in this module is label-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch

from .data import validate_events
from .h2.constants import OUTPUT_CELL, PATCHES_PER_WINDOW, WINDOWS_PER_SEQUENCE
from .h2.features import extract_event_features
from .h2.numpy_model import (
    H2Export,
    _refiner_probability,
    _sigmoid,
    load_export,
    quantized_mlp,
)
from .h2.representation import (
    build_patch_features,
    candidate_indices,
    patch_scores_to_heatmaps,
    window_indices,
)
from .h2.torch_model import QuantizedMLP
from .h2_refiner import refiner_features_shared


H2_RESEARCH_PROTOCOL = "h2-first-accuracy-v2"
HISTORY_FEATURES = 4


@dataclass
class H2Prepared:
    indices: np.ndarray
    features: np.ndarray
    history: np.ndarray
    heatmaps: np.ndarray
    base_features: np.ndarray
    windows: Tuple[int, ...]
    timings: Dict[str, float]
    all_indices: Optional[np.ndarray] = None
    all_features: Optional[np.ndarray] = None


def previous_window_features(
    events: np.ndarray,
    indices: np.ndarray,
    windows: Sequence[int],
    heatmaps: np.ndarray,
    null_history: bool = False,
) -> np.ndarray:
    """Return four uint8 features from the two preceding 100 ms windows.

    Columns are previous-cell score, previous 3x3 mean, maximum same-cell
    score over the previous two windows, and maximum 3x3 mean over them.
    Missing history is zero.  Current/future windows are never read.
    """
    selected = np.asarray(indices, dtype=np.int64)
    result = np.zeros((len(selected), HISTORY_FEATURES), dtype=np.uint8)
    if null_history or not len(selected):
        return result
    chosen = np.asarray(windows, dtype=np.int64)
    if heatmaps.shape != (len(chosen), 34, 44):
        raise ValueError("heatmaps must match the selected legacy windows")
    if len(np.unique(chosen)) != len(chosen):
        raise ValueError("windows must be unique")
    assigned = window_indices(events).astype(np.int64)
    if np.any(selected < 0) or np.any(selected >= len(events)):
        raise ValueError("event index is outside the sequence")
    lookup = {int(window): position for position, window in enumerate(chosen)}
    if any(int(assigned[index]) not in lookup for index in selected):
        raise ValueError("selected event belongs to an unselected window")
    padded = np.pad(heatmaps, ((0, 0), (1, 1), (1, 1)), mode="edge")
    neighbor = np.zeros_like(heatmaps)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neighbor += padded[:, 1 + dy:1 + dy + 34, 1 + dx:1 + dx + 44] / 9.0
    x = events["x"][selected].astype(np.int64) // OUTPUT_CELL
    y = events["y"][selected].astype(np.int64) // OUTPUT_CELL
    values = np.zeros((len(selected), HISTORY_FEATURES), dtype=np.float32)
    for row, (event_index, cy, cx) in enumerate(zip(selected, y, x)):
        current_window = int(assigned[event_index])
        prior = [lookup.get(current_window - lag) for lag in (1, 2)]
        cell = [heatmaps[p, cy, cx] if p is not None else 0.0 for p in prior]
        local = [neighbor[p, cy, cx] if p is not None else 0.0 for p in prior]
        values[row] = (cell[0], local[0], max(cell), max(local))
    return np.rint(255.0 * np.clip(values, 0.0, 1.0)).astype(np.uint8)


def augment_refiner_features(features: np.ndarray, history: np.ndarray) -> np.ndarray:
    if features.ndim != 2 or features.shape[1] != 32:
        raise ValueError("H2 Refiner features must have shape [events, 32]")
    if history.shape != (len(features), HISTORY_FEATURES):
        raise ValueError("history feature shape mismatch")
    return np.ascontiguousarray(np.concatenate((features, history), axis=1), dtype=np.uint8)


class FrozenH2Pipeline:
    """Exact frozen H2 Proposal/Refiner with explicit diagnostic outputs."""

    def __init__(self, h2_path: Path, batch_size: int = 4096) -> None:
        if batch_size <= 0:
            raise ValueError("batch size must be positive")
        self.h2_path = Path(h2_path)
        self.h2 = load_export(self.h2_path)
        self.batch_size = int(batch_size)

    def _probability(self, features: np.ndarray) -> np.ndarray:
        if not len(features):
            return np.empty((0,), dtype=np.float32)
        pieces = []
        for first in range(0, len(features), self.batch_size):
            raw = quantized_mlp(features[first:first + self.batch_size], self.h2.refiner_layers)
            pieces.append(_refiner_probability(raw, self.h2.prototypes))
        return np.concatenate(pieces)

    def prepare(
        self,
        events: np.ndarray,
        windows: Optional[Sequence[int]] = None,
        include_all: bool = False,
    ) -> H2Prepared:
        validate_events(events)
        chosen = tuple(range(WINDOWS_PER_SEQUENCE)) if windows is None else tuple(int(v) for v in windows)
        if not chosen or len(set(chosen)) != len(chosen) or min(chosen) < 0 or max(chosen) >= WINDOWS_PER_SEQUENCE:
            raise ValueError("windows must be unique valid legacy window indices")
        timings: Dict[str, float] = {}
        started = time.perf_counter()
        patch = build_patch_features(events, chosen)
        base = extract_event_features(events)[0]
        timings["representation_s"] = time.perf_counter() - started
        started = time.perf_counter()
        raw = quantized_mlp(patch.reshape(-1, 128), self.h2.proposal_layers)
        scores = _sigmoid(raw).reshape(len(chosen), PATCHES_PER_WINDOW, 16)
        heatmaps = patch_scores_to_heatmaps(scores)
        timings["proposal_s"] = time.perf_counter() - started
        started = time.perf_counter()
        indices = candidate_indices(events, chosen, heatmaps, self.h2.proposal_threshold)
        timings["routing_s"] = time.perf_counter() - started
        started = time.perf_counter()
        features = refiner_features_shared(
            events, indices, chosen, heatmaps, self.h2.proposal_threshold, base
        )
        history = previous_window_features(events, indices, chosen, heatmaps)
        all_indices = None
        all_features = None
        if include_all:
            assigned = window_indices(events)
            all_indices = np.flatnonzero(np.isin(assigned, np.asarray(chosen))).astype(np.int64)
            all_features = refiner_features_shared(
                events, all_indices, chosen, heatmaps, self.h2.proposal_threshold, base
            )
        timings["refiner_features_s"] = time.perf_counter() - started
        return H2Prepared(
            indices=indices,
            features=features,
            history=history,
            heatmaps=heatmaps,
            base_features=base,
            windows=chosen,
            timings=timings,
            all_indices=all_indices,
            all_features=all_features,
        )

    def infer_prepared(self, prepared: H2Prepared, event_count: int) -> Dict[str, Any]:
        started = time.perf_counter()
        candidate_probability = self._probability(prepared.features)
        refiner_s = time.perf_counter() - started
        probability = np.zeros(event_count, dtype=np.float32)
        probability[prepared.indices] = candidate_probability
        return {
            "probabilities": probability,
            "candidate_probabilities": candidate_probability,
            "timings": dict(prepared.timings, refiner_s=refiner_s),
        }

    def ungated_probability(self, prepared: H2Prepared, event_count: int) -> np.ndarray:
        if prepared.all_indices is None or prepared.all_features is None:
            raise ValueError("prepare(..., include_all=True) is required")
        probability = np.zeros(event_count, dtype=np.float32)
        probability[prepared.all_indices] = self._probability(prepared.all_features)
        return probability


def build_h2_refiner(variant: str, exported: H2Export, seed: int = 42) -> QuantizedMLP:
    if variant not in ("A1", "A2", "A3", "A3-null"):
        raise ValueError("variant must be A1, A2, A3, or A3-null")
    torch.manual_seed(seed)
    input_width = 36 if variant in ("A3", "A3-null") else 32
    model = QuantizedMLP((input_width, 64, 32, 4), quantized=False)
    with torch.no_grad():
        for index, (layer, (weight, scale)) in enumerate(zip(model.layers, exported.refiner_layers)):
            restored = torch.from_numpy((weight.astype(np.float32) * scale[None, :]).T.copy())
            if index == 0 and input_width > 32:
                layer.weight.zero_()
                layer.weight[:, :32].copy_(restored)
            else:
                layer.weight.copy_(restored)
    return model


def hard_random_sample(
    labels: np.ndarray,
    baseline_probability: np.ndarray,
    seed: int,
    negative_ratio: int = 3,
) -> np.ndarray:
    """Keep all positives and mix equal hard/random negative quotas."""
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(baseline_probability, dtype=np.float32)
    if y.ndim != 1 or p.shape != y.shape or not np.all(np.isin(y, (0, 1))):
        raise ValueError("labels/probabilities must be matching binary vectors")
    if negative_ratio < 1:
        raise ValueError("negative_ratio must be positive")
    positive = np.flatnonzero(y == 1)
    negative = np.flatnonzero(y == 0)
    if not len(negative):
        return positive
    requested = len(positive) * negative_ratio if len(positive) else min(4096, len(negative))
    quota = min(len(negative), requested)
    hard_quota = quota // 2
    ranked = negative[np.argsort(-p[negative], kind="stable")]
    hard = ranked[:hard_quota]
    remaining = np.setdiff1d(negative, hard, assume_unique=False)
    rng = np.random.default_rng(seed)
    random_count = min(quota - len(hard), len(remaining))
    random_negative = rng.choice(remaining, size=random_count, replace=False) if random_count else np.empty(0, dtype=np.int64)
    selected = np.concatenate((positive, hard, random_negative)).astype(np.int64)
    return selected[rng.permutation(len(selected))]


def select_fair_working_point(
    curve: Dict[str, Dict[str, Any]], anchor_pd: float, anchor_fa: float
) -> Dict[str, Any]:
    """Apply the same Pd/Fa constraint and deterministic tie-break to any model."""
    if not np.isfinite(anchor_pd) or not np.isfinite(anchor_fa) or not 0 <= anchor_pd <= 1 or anchor_fa < 0:
        raise ValueError("invalid anchor Pd/Fa")
    limit = 0.0 if anchor_fa == 0.0 else anchor_fa * 1.10
    feasible = []
    for key, metrics in curve.items():
        threshold = float(key)
        event, localization = metrics["event"], metrics["localization"]
        iou, recall = event["iou"], event["recall"]
        pd, fa = localization["pd"], localization["fa"]
        if any(value is None or not np.isfinite(value) for value in (iou, recall, pd, fa)):
            continue
        if pd >= anchor_pd - 0.005 and fa <= limit:
            feasible.append((float(iou), -float(fa), float(recall), -threshold, threshold, metrics))
    if not feasible:
        return {"eligible": False, "threshold": None, "metrics": None,
                "reason": "no point satisfies common Pd/Fa constraints"}
    selected = max(feasible)
    return {"eligible": True, "threshold": selected[4], "metrics": selected[5],
            "reason": "max IoU, then lower Fa, higher Recall, lower threshold"}
