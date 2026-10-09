"""S0-R 的 H2 Proposal 数据入口与事件流训练目标。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from .data import load_events
from .h2.constants import PROPOSAL_DIMENSIONS, WINDOWS_PER_SEQUENCE
from .h2.losses import refiner_loss
from .h2.numpy_model import H2Export, _refiner_probability, quantized_mlp
from .h2.representation import (
    build_patch_features,
    build_refiner_features,
    candidate_indices,
    patch_scores_to_heatmaps,
    window_indices,
)


@dataclass(frozen=True)
class S0RWindow:
    path: Path
    window: int
    event_indices: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    teacher_probability: np.ndarray


def flatten_s0r_windows(
    windows: Sequence[S0RWindow],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把窗口缓存连接为互相独立的候选事件样本。"""
    if not windows:
        raise ValueError("S0-R data must contain at least one window")
    return (
        np.ascontiguousarray(np.concatenate([item.features for item in windows])),
        np.ascontiguousarray(np.concatenate([item.labels for item in windows])),
        np.ascontiguousarray(
            np.concatenate([item.teacher_probability for item in windows])
        ),
    )


def build_s0r_candidates(
    events: np.ndarray,
    h2: H2Export,
    *,
    heatmaps: np.ndarray | None = None,
    proposal_threshold: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """从同一前级热图构建候选、特征和教师目标；默认保留 H2 路径。"""
    windows = list(range(WINDOWS_PER_SEQUENCE))
    if heatmaps is None:
        if proposal_threshold is not None:
            raise ValueError("external threshold requires external heatmaps")
        patch_features = build_patch_features(events, windows)
        patch_logits = quantized_mlp(
            patch_features.reshape(-1, PROPOSAL_DIMENSIONS[0]), h2.proposal_layers
        )
        clipped = np.clip(patch_logits.astype(np.float64), -60.0, 60.0)
        patch_scores = (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)
        heatmaps = patch_scores_to_heatmaps(
            patch_scores.reshape(WINDOWS_PER_SEQUENCE, -1, PROPOSAL_DIMENSIONS[-1])
        )
        proposal_threshold = h2.proposal_threshold
    elif proposal_threshold is None or not 0.0 < proposal_threshold < 1.0:
        raise ValueError("external heatmaps require an explicit threshold in (0, 1)")
    if not np.isfinite(heatmaps).all() or np.any((heatmaps < 0) | (heatmaps > 1)):
        raise ValueError("invalid proposal probabilities")
    candidates = candidate_indices(events, windows, heatmaps, proposal_threshold)
    if len(candidates) == 0:
        return (candidates, np.empty((0, 32), dtype=np.uint8),
                np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32))
    features = build_refiner_features(
        events, candidates, windows, heatmaps, proposal_threshold
    )
    teacher_raw = quantized_mlp(features, h2.refiner_layers)
    teacher_probability = _refiner_probability(teacher_raw, h2.prototypes)
    return candidates, features, events["label"][candidates].astype(np.float32), teacher_probability


def build_s0r_windows(path: Path, h2: H2Export) -> List[S0RWindow]:
    """使用冻结 H2 Proposal 构造按 100 ms 分组的候选事件流。"""
    events = load_events(path, require_labels=True)
    candidates, features, labels, teacher_probability = build_s0r_candidates(events, h2)
    assigned = window_indices(events)[candidates]
    result = []
    for window in range(WINDOWS_PER_SEQUENCE):
        member = assigned == window
        if not np.any(member):
            continue
        selected = candidates[member]
        result.append(
            S0RWindow(
                path=path,
                window=window,
                event_indices=np.ascontiguousarray(selected),
                features=np.ascontiguousarray(features[member]),
                labels=np.ascontiguousarray(labels[member]),
                teacher_probability=np.ascontiguousarray(
                    teacher_probability[member]
                ),
            )
        )
    return result


def s0r_loss(
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_probability: torch.Tensor,
    spike_rates: torch.Tensor,
    distillation_weight: float = 0.20,
    spike_rate_weight: float = 0.01,
    minimum_rate: float = 0.01,
    maximum_rate: float = 0.25,
) -> Dict[str, torch.Tensor]:
    """组合 H2 Refiner 目标、蒸馏和非退化发放率约束。"""
    if raw_logits.ndim != 2 or raw_logits.shape[-1] != 4:
        raise ValueError("S0-R raw logits must have shape [event, 4]")
    expected = raw_logits.shape[:1]
    if labels.shape != expected or teacher_probability.shape != expected:
        raise ValueError("S0-R labels and teacher probability must match event/batch")
    if spike_rates.shape != (3,):
        raise ValueError("S0-R spike rates must contain three layers")
    base = refiner_loss(
        raw_logits,
        labels.to(raw_logits.device),
        teacher_probability.to(raw_logits.device),
        torch.ones_like(labels, device=raw_logits.device),
        distillation_weight=distillation_weight,
    )
    too_low = torch.relu(float(minimum_rate) - spike_rates)
    too_high = torch.relu(spike_rates - float(maximum_rate))
    rate_penalty = torch.mean(torch.square(too_low) + torch.square(too_high))
    total = base["total"] + float(spike_rate_weight) * rate_penalty
    return {**base, "rate_penalty": rate_penalty, "total": total}
