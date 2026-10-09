"""训练专用的 Proposal 时空相关性工具。"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as functional

from .h2.constants import (
    OUTPUT_CELL,
    PADDED_HEIGHT,
    PADDED_WIDTH,
    PATCH_COLUMNS,
    PATCH_OUTPUT_DIM,
    PATCH_ROWS,
    PATCH_STRIDE,
    PATCHES_PER_WINDOW,
)


STC_KERNELS: Tuple[Tuple[int, int], ...] = ((1, 3), (3, 3), (3, 5))
_HEATMAP_HEIGHT = PADDED_HEIGHT // OUTPUT_CELL
_HEATMAP_WIDTH = PADDED_WIDTH // OUTPUT_CELL


def _patch_cell_indices() -> Tuple[int, ...]:
    indices = []
    for patch_y in range(PATCH_ROWS):
        y0 = patch_y * PATCH_STRIDE // OUTPUT_CELL
        for patch_x in range(PATCH_COLUMNS):
            x0 = patch_x * PATCH_STRIDE // OUTPUT_CELL
            for local_y in range(4):
                for local_x in range(4):
                    indices.append(
                        (y0 + local_y) * _HEATMAP_WIDTH + x0 + local_x
                    )
    if len(indices) != PATCHES_PER_WINDOW * PATCH_OUTPUT_DIM:
        raise AssertionError("Proposal patch 到热图的映射数量错误")
    return tuple(indices)


_PATCH_CELL_INDICES = _patch_cell_indices()


def stc_block_ranges(
    window_count: int, core_windows: int, halo_windows: int = 1
) -> list[tuple[int, int, int, int]]:
    """Return load and local-core ranges without duplicating core windows."""
    if window_count <= 0 or core_windows <= 0 or halo_windows < 0:
        raise ValueError("invalid STC block geometry")
    ranges = []
    for core_start in range(0, window_count, core_windows):
        core_stop = min(core_start + core_windows, window_count)
        load_start = max(0, core_start - halo_windows)
        load_stop = min(window_count, core_stop + halo_windows)
        ranges.append(
            (
                load_start,
                load_stop,
                core_start - load_start,
                core_stop - load_start,
            )
        )
    return ranges


def validate_stc_kernel(spatial_kernel: int, temporal_kernel: int) -> None:
    if (int(spatial_kernel), int(temporal_kernel)) not in STC_KERNELS:
        raise ValueError(
            "STC kernel must be one of "
            + ", ".join(f"{spatial}x{spatial}x{temporal}" for spatial, temporal in STC_KERNELS)
        )


def patch_probabilities_to_heatmaps(probabilities: torch.Tensor) -> torch.Tensor:
    """Fuse ``[window, time, 336, 16]`` non-negative values by maximum."""
    if probabilities.ndim != 4 or probabilities.shape[2:] != (
        PATCHES_PER_WINDOW,
        PATCH_OUTPUT_DIM,
    ):
        raise ValueError("patch probabilities must have shape [window, time, 336, 16]")
    if not probabilities.is_floating_point():
        probabilities = probabilities.to(torch.float32)
    window_count, time_steps = probabilities.shape[:2]
    rows = probabilities.reshape(
        window_count * time_steps, PATCHES_PER_WINDOW * PATCH_OUTPUT_DIM
    )
    index = torch.as_tensor(
        _PATCH_CELL_INDICES, dtype=torch.int64, device=probabilities.device
    ).unsqueeze(0).expand(rows.shape[0], -1)
    heatmaps = torch.zeros(
        (rows.shape[0], _HEATMAP_HEIGHT * _HEATMAP_WIDTH),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    heatmaps = heatmaps.scatter_reduce(
        1, index, rows, reduce="amax", include_self=True
    )
    return heatmaps.reshape(
        window_count, time_steps, _HEATMAP_HEIGHT, _HEATMAP_WIDTH
    )


def stc_support(
    probabilities: torch.Tensor, spatial_kernel: int, temporal_kernel: int
) -> torch.Tensor:
    """Sum neighboring probabilities across consecutive micro time steps."""
    validate_stc_kernel(spatial_kernel, temporal_kernel)
    if probabilities.ndim != 4 or probabilities.shape[2:] != (
        _HEATMAP_HEIGHT,
        _HEATMAP_WIDTH,
    ):
        raise ValueError("probability heatmaps must have shape [window, time, 34, 44]")
    window_count, time_steps = probabilities.shape[:2]
    sequence = probabilities.reshape(
        1, 1, window_count * time_steps, _HEATMAP_HEIGHT, _HEATMAP_WIDTH
    )
    kernel = torch.ones(
        (1, 1, temporal_kernel, spatial_kernel, spatial_kernel),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    support = functional.conv3d(
        sequence,
        kernel,
        padding=(temporal_kernel // 2, spatial_kernel // 2, spatial_kernel // 2),
    )
    return support.reshape_as(probabilities)


def stc_weights(
    probabilities: torch.Tensor,
    spatial_kernel: int,
    temporal_kernel: int,
    variant: str = "released",
) -> torch.Tensor:
    """Return detached paper- or release-style correlation weights."""
    support = stc_support(
        probabilities.detach(), spatial_kernel, temporal_kernel
    )
    if variant == "released":
        weights = torch.sigmoid(support - support.mean())
    elif variant == "paper":
        weights = torch.sigmoid(support)
    else:
        raise ValueError("STC variant must be released or paper")
    return weights.detach()


def proposal_stc_loss(
    step_currents: torch.Tensor,
    temporal_targets: torch.Tensor,
    spatial_kernel: int,
    temporal_kernel: int,
    core_start: int = 0,
    core_stop: int | None = None,
    eps: float = 1.0e-5,
) -> Dict[str, torch.Tensor]:
    """Compute the released-code-inspired loss on Proposal micro time steps."""
    if step_currents.ndim != 3 or step_currents.shape[2] != PATCH_OUTPUT_DIM:
        raise ValueError("step currents must have shape [time, window*336, 16]")
    if temporal_targets.ndim != 4 or temporal_targets.shape[2:] != (
        PATCHES_PER_WINDOW,
        PATCH_OUTPUT_DIM,
    ):
        raise ValueError("temporal targets must have shape [window, time, 336, 16]")
    window_count, time_steps = temporal_targets.shape[:2]
    if step_currents.shape[:2] != (
        time_steps,
        window_count * PATCHES_PER_WINDOW,
    ):
        raise ValueError("step currents and temporal targets are not aligned")
    if core_stop is None:
        core_stop = window_count
    if not 0 <= core_start < core_stop <= window_count:
        raise ValueError("invalid STC core window range")
    probability_patches = torch.sigmoid(step_currents).reshape(
        time_steps, window_count, PATCHES_PER_WINDOW, PATCH_OUTPUT_DIM
    ).permute(1, 0, 2, 3)
    probabilities = patch_probabilities_to_heatmaps(probability_patches)
    targets = patch_probabilities_to_heatmaps(
        temporal_targets.to(device=step_currents.device, dtype=step_currents.dtype)
    )
    weights = stc_weights(
        probabilities, spatial_kernel, temporal_kernel, variant="released"
    )
    clipped = probabilities.clamp(eps, 1.0 - eps)
    values = -(
        targets * weights * torch.log(clipped)
        + (1.0 - targets) * (1.0 - weights) * torch.log(1.0 - clipped)
    )
    selected = values[core_start:core_stop]
    selected_weights = weights[core_start:core_stop]
    selected_targets = targets[core_start:core_stop]
    return {
        "loss": selected.mean(),
        "mean_weight": selected_weights.mean(),
        "positive_weight": (
            selected_weights[selected_targets > 0.5].mean()
            if bool(torch.any(selected_targets > 0.5))
            else selected_weights.new_zeros(())
        ),
        "negative_weight": (
            selected_weights[selected_targets <= 0.5].mean()
            if bool(torch.any(selected_targets <= 0.5))
            else selected_weights.new_zeros(())
        ),
    }


def binary_rank_auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Compute tie-aware binary AUROC without an additional dependency."""
    value = np.asarray(scores, dtype=np.float64).reshape(-1)
    truth = np.asarray(labels, dtype=bool).reshape(-1)
    if len(value) != len(truth):
        raise ValueError("scores and labels must have identical lengths")
    positive_count = int(truth.sum())
    negative_count = len(truth) - positive_count
    if positive_count == 0 or negative_count == 0:
        return None
    order = np.argsort(value, kind="mergesort")
    sorted_value = value[order]
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        stop = start + 1
        while stop < len(value) and sorted_value[stop] == sorted_value[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive_rank_sum = float(ranks[truth].sum())
    return (
        positive_rank_sum - positive_count * (positive_count + 1) / 2.0
    ) / (positive_count * negative_count)


def _quantiles(values: torch.Tensor) -> Dict[str, float]:
    flattened = values.detach().to(torch.float64).cpu().reshape(-1)
    levels = torch.tensor(
        [0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0], dtype=torch.float64
    )
    result = torch.quantile(flattened, levels).tolist()
    return {
        name: float(value)
        for name, value in zip(
            ("min", "p01", "p05", "median", "p95", "p99", "max"), result
        )
    }


def diagnose_stc_heatmaps(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
    spatial_kernel: int,
    temporal_kernel: int,
    prediction_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Measure whether local correlation separates target and false cells."""
    if probabilities.shape != targets.shape:
        raise ValueError("diagnostic probabilities and targets must be aligned")
    if probabilities.ndim == 4:
        probability_sequences = [probabilities]
        target_sequences = [targets]
    elif probabilities.ndim == 5:
        probability_sequences = list(probabilities.unbind(dim=0))
        target_sequences = list(targets.unbind(dim=0))
    else:
        raise ValueError(
            "diagnostic heatmaps must have shape [window, time, 34, 44] "
            "or [sequence, window, time, 34, 44]"
        )
    probability_flat = torch.cat(
        [value.reshape(-1) for value in probability_sequences]
    )
    target_flat = torch.cat([value.reshape(-1) for value in target_sequences])
    released = torch.cat(
        [
            stc_weights(
                value, spatial_kernel, temporal_kernel, variant="released"
            ).reshape(-1)
            for value in probability_sequences
        ]
    )
    paper = torch.cat(
        [
            stc_weights(value, spatial_kernel, temporal_kernel, variant="paper").reshape(-1)
            for value in probability_sequences
        ]
    )
    truth = target_flat > 0.5
    predicted = probability_flat >= float(prediction_threshold)
    union = truth | predicted
    union_labels = truth[union].detach().cpu().numpy()
    union_scores = released[union].detach().cpu().numpy()
    target_weights = released[truth]
    false_weights = released[predicted & ~truth]
    target_count = int(truth.sum().item())
    baseline_positive = int(predicted.sum().item())
    baseline_recall = (
        float((predicted & truth).sum().item()) / target_count
        if target_count
        else 1.0
    )
    reranked = torch.clamp(probability_flat * (0.5 + released), 0.0, 1.0)
    reranked_predicted = reranked >= float(prediction_threshold)
    reranked_recall = (
        float((reranked_predicted & truth).sum().item()) / target_count
        if target_count
        else 1.0
    )
    total_cells = int(probability_flat.numel())
    baseline_fraction = baseline_positive / total_cells if total_cells else 0.0
    reranked_fraction = (
        float(reranked_predicted.sum().item()) / total_cells if total_cells else 0.0
    )
    released_saturation = float(
        ((released < 0.01) | (released > 0.99)).to(torch.float32).mean().item()
    )
    paper_saturation = float(
        ((paper < 0.01) | (paper > 0.99)).to(torch.float32).mean().item()
    )
    auc = binary_rank_auc(union_scores, union_labels)
    target_median = (
        float(torch.median(target_weights).item()) if target_weights.numel() else None
    )
    false_median = (
        float(torch.median(false_weights).item()) if false_weights.numel() else None
    )
    gates = {
        "auc_at_least_0_60": auc is not None and auc >= 0.60,
        "target_median_above_false": (
            target_median is not None
            and false_median is not None
            and target_median > false_median
        ),
        "released_weights_not_saturated": released_saturation < 0.95,
        "probe_recall_improves": reranked_recall > baseline_recall,
        "probe_candidate_growth_within_5_points": (
            reranked_fraction - baseline_fraction <= 0.05
        ),
    }
    return {
        "kernel": {
            "spatial": int(spatial_kernel),
            "temporal_micro_steps": int(temporal_kernel),
        },
        "prediction_threshold": float(prediction_threshold),
        "cells": total_cells,
        "target_cells": target_count,
        "false_positive_cells": int((predicted & ~truth).sum().item()),
        "released": {
            "auroc_on_prediction_target_union": auc,
            "target_weight_median": target_median,
            "false_weight_median": false_median,
            "saturated_fraction": released_saturation,
            "weight_quantiles": _quantiles(released),
        },
        "paper_equation": {
            "gamma": 2.0,
            "saturated_fraction_before_gamma": paper_saturation,
            "weight_quantiles_before_gamma": _quantiles(paper),
        },
        "probe": {
            "rule": "clip(probability * (0.5 + released_weight), 0, 1)",
            "baseline_recall": baseline_recall,
            "reranked_recall": reranked_recall,
            "baseline_positive_fraction": baseline_fraction,
            "reranked_positive_fraction": reranked_fraction,
        },
        "gates": gates,
        "passes": all(gates.values()),
    }


def choose_stc_diagnostic(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any] | None:
    """Choose the highest-AUROC passing kernel, preferring smaller volume."""
    passing = [candidate for candidate in candidates if candidate.get("passes")]
    if not passing:
        return None
    return max(
        passing,
        key=lambda candidate: (
            float(candidate["released"]["auroc_on_prediction_target_union"]),
            -int(candidate["kernel"]["spatial"]) ** 2
            * int(candidate["kernel"]["temporal_micro_steps"]),
        ),
    )
