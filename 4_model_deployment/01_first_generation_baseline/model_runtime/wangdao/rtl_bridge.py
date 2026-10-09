"""Strict XSim transfer capture and single-window CPU routing of Q12 logits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np

from .fixed import FixedH2, FixedMLPOutput
from .h2.constants import PATCHES_PER_WINDOW, WINDOWS_PER_SEQUENCE
from .h2.numpy_model import _refiner_probability
from .h2.representation import (
    build_refiner_features, candidate_indices, patch_scores_to_heatmaps, window_indices,
)


@dataclass(frozen=True)
class ProposalCapture:
    logits_q: np.ndarray  # [exact/approximate, patch, channel], signed Q12
    tree_saturations: np.ndarray
    membrane_saturations: np.ndarray
    cycles: np.ndarray


def read_proposal_capture(path: Path, patches: int) -> ProposalCapture:
    """Reject incomplete runs, missing/duplicate transfers, X/Z and invalid words."""
    if not 2 <= patches <= 2048:
        raise ValueError("Invalid capture patch count")
    lines = path.read_text(encoding="ascii").splitlines()
    count = patches * 16
    if (len(lines) != count + 2
            or lines[0] != f"WD_CAPTURE version=1 patches={patches} channels=16"
            or lines[-1] != f"WD_CAPTURE_END records={count}"):
        raise ValueError("Incomplete capture/header/footer/count mismatch")
    logits = np.empty((2, count), dtype=np.int64)
    tree = np.empty_like(logits)
    membrane = np.empty_like(logits)
    cycles = np.empty(count, dtype=np.int64)
    for index, line in enumerate(lines[1:-1]):
        fields = line.split()
        if len(fields) != 9 or any(
            not re.fullmatch(r"[0-9a-fA-F]{8}" if column in (3, 6) else r"[0-9]+", value)
            for column, value in enumerate(fields)
        ):
            raise ValueError(f"Malformed capture row {index}")
        values = [int(value, 16 if column in (3, 6) else 10) for column, value in enumerate(fields)]
        if values[:2] != [index // 16, index % 16]:
            raise ValueError(f"Missing/duplicate/out-of-order transfer at {index}")
        if not 0 < values[2] <= 2**31 - 1 or (index and values[2] <= cycles[index - 1]):
            raise ValueError("Invalid transfer cycle order")
        cycles[index] = values[2]
        for mode, offset in enumerate((3, 6)):
            word, tree_count, membrane_count = values[offset:offset + 3]
            signed = word - (1 << 32) if word & (1 << 31) else word
            if not -(1 << 23) <= signed < (1 << 23):
                raise ValueError("Q12 readout must be sign-extended from 24 bits")
            if tree_count > 92 or membrane_count > 4:
                raise ValueError("Impossible readout saturation count")
            logits[mode, index] = signed
            tree[mode, index] = tree_count
            membrane[mode, index] = membrane_count
    return ProposalCapture(logits.reshape(2, patches, 16), tree.reshape(2, patches, 16),
                           membrane.reshape(2, patches, 16), cycles.reshape(patches, 16))


def route_window(events: np.ndarray, window: int, logits_q: np.ndarray,
                 refiner: FixedH2, proposal_threshold: float, event_threshold: float,
                 batch_size: int = 4096) -> dict[str, np.ndarray | tuple[int, ...]]:
    """Keep original sequence indices/history; excluded events retain probability zero."""
    value = np.asarray(logits_q)
    if (value.shape != (PATCHES_PER_WINDOW, 16) or value.dtype.kind not in "iu"
            or np.any(value < -(1 << 23)) or np.any(value >= (1 << 23))):
        raise ValueError("Expected 336 x 16 signed Q12 logits")
    if not 0 <= window < WINDOWS_PER_SEQUENCE or batch_size < 1:
        raise ValueError("Invalid window/batch size")
    if not (0 < proposal_threshold < 1 and 0 < event_threshold < 1):
        raise ValueError("Invalid frozen thresholds")
    logits = (value.astype(np.float64) / 4096.0).astype(np.float32)
    scores = (1.0 / (1.0 + np.exp(-np.clip(logits.astype(np.float64), -60, 60)))).astype(np.float32)
    heatmaps = patch_scores_to_heatmaps(scores[None, ...])
    candidates = candidate_indices(events, [window], heatmaps, proposal_threshold)
    features = build_refiner_features(events, candidates, [window], heatmaps, proposal_threshold)
    pieces: list[FixedMLPOutput] = [refiner.refiner(features[start:start + batch_size])
                                  for start in range(0, len(features), batch_size)]
    refiner_q = np.concatenate([p.logits_q for p in pieces]) if pieces else np.empty((0, 4), dtype=np.int64)
    raw = np.concatenate([p.logits for p in pieces]) if pieces else np.empty((0, 4), dtype=np.float32)
    saturation = tuple(sum(p.accumulator_saturations[i] for p in pieces) for i in range(3))
    candidate_probability = (_refiner_probability(raw, refiner.source.prototypes)
                             if pieces else np.empty(0, dtype=np.float32))
    members = np.flatnonzero(window_indices(events) == window)
    local = np.searchsorted(members, candidates)
    if not np.array_equal(members[local], candidates):
        raise ValueError("Candidate event indices escaped the selected window")
    probability = np.zeros(len(members), dtype=np.float32)
    probability[local] = candidate_probability
    return {"logits_q": value, "heatmaps": heatmaps, "candidates": candidates,
            "features": features, "refiner_logits_q": refiner_q,
            "refiner_saturations": saturation, "event_indices": members,
            "probabilities": probability, "predictions": probability >= event_threshold}


def assert_pipeline_equal(actual: dict, expected: dict) -> None:
    for key in expected:
        if key not in actual or not np.array_equal(actual[key], expected[key]):
            raise ValueError(f"RTL/ Python pipeline mismatch: {key}")
