"""Event-level and EV-UAV localization metrics."""

from __future__ import annotations

from collections import deque
from typing import Dict

import numpy as np

from .data import SENSOR_HEIGHT, SENSOR_WIDTH


OFFICIAL_PD_WINDOW_MS = 50.0
OFFICIAL_CORRECT_THRESHOLD = 0.0001


def event_metrics(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> Dict[str, float]:
    if len(labels) != len(probabilities):
        raise ValueError("labels and probabilities must have identical lengths")
    truth = labels.astype(bool, copy=False)
    predicted = probabilities >= float(threshold)
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    tn = int(np.count_nonzero(~truth & ~predicted))
    union = tp + fp + fn
    predicted_positive = tp + fp
    target_total = tp + fn
    return {
        "threshold": float(threshold),
        "true_positive": float(tp),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_negative": float(tn),
        "iou": float(tp / union) if union else 1.0,
        "precision": float(tp / predicted_positive) if predicted_positive else 1.0,
        "recall": float(tp / target_total) if target_total else 1.0,
        "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0,
        "accuracy": float((tp + tn) / len(labels)) if len(labels) else 1.0,
    }


def _component_count(mask: np.ndarray) -> int:
    visited = np.zeros(mask.shape, dtype=bool)
    components = 0
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        yi, xi = int(y), int(x)
        if visited[yi, xi]:
            continue
        components += 1
        visited[yi, xi] = True
        queue = deque([(yi, xi)])
        while queue:
            cy, cx = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = cy + dy, cx + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not visited[ny, nx]
                    ):
                        visited[ny, nx] = True
                        queue.append((ny, nx))
    return components


def localization_metrics(
    timestamps_ms: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    identities: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    window_ms: float = OFFICIAL_PD_WINDOW_MS,
    correct_threshold: float = OFFICIAL_CORRECT_THRESHOLD,
) -> Dict[str, float]:
    lengths = {len(timestamps_ms), len(x), len(y), len(labels), len(identities), len(probabilities)}
    if len(lengths) != 1:
        raise ValueError("localization arrays must have identical lengths")
    if len(labels) == 0:
        return {
            "pd": 1.0,
            "fa": 0.0,
            "targets": 0.0,
            "correct_targets": 0.0,
            "false_components": 0.0,
            "frame_denominator": 0.0,
        }
    relative = timestamps_ms.astype(np.float64) - float(timestamps_ms[0])
    duration = float(relative.max() - relative.min())
    frame_denominator = int(duration / window_ms)
    predicted = probabilities >= float(threshold)
    correct_targets = targets = false_components = 0
    for frame_index in range(int(duration / window_ms + 1)):
        member = (relative > frame_index * window_ms) & (relative < (frame_index + 1) * window_ms)
        if not np.any(member):
            continue
        frame_identities = identities[member]
        frame_labels = labels[member]
        frame_predictions = predicted[member]
        for identity in np.unique(frame_identities):
            if int(identity) == 0:
                continue
            target = frame_identities == identity
            targets += 1
            denominator = int(frame_labels[target].sum())
            if denominator and np.count_nonzero(frame_predictions[target] == frame_labels[target]) / denominator >= correct_threshold:
                correct_targets += 1
        false_event = (frame_labels == 0) & frame_predictions
        false_mask = np.zeros((SENSOR_HEIGHT, SENSOR_WIDTH), dtype=bool)
        false_mask[y[member][false_event].astype(np.int64), x[member][false_event].astype(np.int64)] = True
        false_components += _component_count(false_mask)
    denominator_pixels = frame_denominator * SENSOR_WIDTH * SENSOR_HEIGHT
    return {
        "pd": float(correct_targets / targets) if targets else 1.0,
        "fa": float(false_components / denominator_pixels) if denominator_pixels else 0.0,
        "targets": float(targets),
        "correct_targets": float(correct_targets),
        "false_components": float(false_components),
        "frame_denominator": float(frame_denominator),
    }
