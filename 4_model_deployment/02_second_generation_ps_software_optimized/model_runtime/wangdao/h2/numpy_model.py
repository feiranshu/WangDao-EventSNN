"""Pure NumPy loader and end-to-end inference for the selected H2 model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, NamedTuple, Tuple

import numpy as np

from .constants import (
    EVENT_THRESHOLD,
    EXPORT_FORMAT_VERSION,
    MAX_BATCH_ROWS,
    PROPOSAL_DIMENSIONS,
    PROPOSAL_THRESHOLD,
    PROTOTYPES,
    REFINER_DIMENSIONS,
    SEQUENCE_DURATION_MS,
)
from .representation import (
    build_patch_features,
    build_refiner_features,
    candidate_indices,
    patch_scores_to_heatmaps,
    selected_windows,
)
from ..data import validate_events


Layer = Tuple[np.ndarray, np.ndarray]


class QuantizedMLPActivity(NamedTuple):
    logits: np.ndarray
    activation_nonzero: Tuple[int, ...]
    activation_values: Tuple[int, ...]


@dataclass(frozen=True)
class H2Export:
    proposal_layers: Tuple[Layer, ...]
    refiner_layers: Tuple[Layer, ...]
    proposal_threshold: float = PROPOSAL_THRESHOLD
    event_threshold: float = EVENT_THRESHOLD
    prototypes: int = PROTOTYPES
    recipe: str = "D1"
    seed: int = 42


def _read_layers(arrays: Dict[str, np.ndarray], prefix: str) -> Tuple[Layer, ...]:
    count = int(arrays[prefix + "_layer_count"].item())
    result = []
    for index in range(count):
        weight = arrays[f"{prefix}_layer{index}_weight"]
        scale = arrays[f"{prefix}_layer{index}_scale"]
        if weight.dtype != np.int8 or weight.ndim != 2:
            raise ValueError(f"{prefix} layer {index} weight must be 2D int8")
        if scale.dtype != np.float32 or scale.shape != (weight.shape[1],):
            raise ValueError(f"{prefix} layer {index} scale shape mismatch")
        result.append((np.ascontiguousarray(weight), np.ascontiguousarray(scale)))
    return tuple(result)


def _layer_dimensions(layers: Tuple[Layer, ...]) -> Tuple[int, ...]:
    if not layers:
        return ()
    dimensions = [int(layers[0][0].shape[0])]
    for index, (weight, _scale) in enumerate(layers):
        if index and weight.shape[0] != dimensions[-1]:
            raise ValueError("exported layers do not connect")
        dimensions.append(int(weight.shape[1]))
    return tuple(dimensions)


def load_export(path: Path) -> H2Export:
    """Load and strictly validate an H2 exported-weight archive."""
    with np.load(str(path), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "format_version", "variant", "recipe", "seed", "proposal_threshold",
        "event_threshold", "prototypes", "proposal_layer_count", "refiner_layer_count",
    }
    missing = sorted(required.difference(arrays))
    if missing:
        raise ValueError("weight archive is missing: " + ", ".join(missing))
    if int(arrays["format_version"].item()) != EXPORT_FORMAT_VERSION:
        raise ValueError("unsupported H2 export format")
    if str(arrays["variant"].item()) != "H2":
        raise ValueError("this package accepts only the selected H2 architecture")
    proposal = _read_layers(arrays, "proposal")
    refiner = _read_layers(arrays, "refiner")
    if _layer_dimensions(proposal) != PROPOSAL_DIMENSIONS:
        raise ValueError("proposal dimensions do not match H2")
    if _layer_dimensions(refiner) != REFINER_DIMENSIONS:
        raise ValueError("refiner dimensions do not match H2")
    prototypes = int(arrays["prototypes"].item())
    if prototypes != PROTOTYPES:
        raise ValueError("prototype count does not match H2")
    return H2Export(
        proposal_layers=proposal,
        refiner_layers=refiner,
        proposal_threshold=float(arrays["proposal_threshold"].item()),
        event_threshold=float(arrays["event_threshold"].item()),
        prototypes=prototypes,
        recipe=str(arrays["recipe"].item()),
        seed=int(arrays["seed"].item()),
    )


def quantized_mlp(features: np.ndarray, layers: Tuple[Layer, ...]) -> np.ndarray:
    """Execute uint8 activations and int8 weights with exact integer products."""
    value = np.ascontiguousarray(features, dtype=np.uint8)
    output = np.empty((len(value), layers[-1][0].shape[1]), dtype=np.float32)
    for index, (weight, scale) in enumerate(layers):
        pieces = []
        for start in range(0, len(value), MAX_BATCH_ROWS):
            activation = value[start:start + MAX_BATCH_ROWS]
            pieces.append(
                np.matmul(activation.astype(np.int32), weight.astype(np.int32))
            )
        raw = (
            np.concatenate(pieces, axis=0)
            if pieces
            else np.empty((0, weight.shape[1]), dtype=np.int32)
        )
        floating = raw.astype(np.float64) * scale[None, :].astype(np.float64) / 255.0
        if index + 1 < len(layers):
            value = np.clip(
                np.rint(np.clip(floating, 0.0, 1.0) * 255.0), 0.0, 255.0
            ).astype(np.uint8)
        else:
            output = floating.astype(np.float32)
    return output


def quantized_mlp_with_activity(
    features: np.ndarray, layers: Tuple[Layer, ...]
) -> QuantizedMLPActivity:
    """执行与 :func:`quantized_mlp` 相同的推理并记录各层激活稀疏性。"""
    value = np.ascontiguousarray(features, dtype=np.uint8)
    nonzero = [int(np.count_nonzero(value))]
    sizes = [int(value.size)]
    output = np.empty((len(value), layers[-1][0].shape[1]), dtype=np.float32)
    for index, (weight, scale) in enumerate(layers):
        pieces = []
        for start in range(0, len(value), MAX_BATCH_ROWS):
            activation = value[start:start + MAX_BATCH_ROWS]
            pieces.append(
                np.matmul(activation.astype(np.int32), weight.astype(np.int32))
            )
        raw = (
            np.concatenate(pieces, axis=0)
            if pieces
            else np.empty((0, weight.shape[1]), dtype=np.int32)
        )
        floating = raw.astype(np.float64) * scale[None, :].astype(np.float64) / 255.0
        if index + 1 < len(layers):
            value = np.clip(
                np.rint(np.clip(floating, 0.0, 1.0) * 255.0), 0.0, 255.0
            ).astype(np.uint8)
            nonzero.append(int(np.count_nonzero(value)))
            sizes.append(int(value.size))
        else:
            output = floating.astype(np.float32)
    return QuantizedMLPActivity(
        logits=output,
        activation_nonzero=tuple(nonzero),
        activation_values=tuple(sizes),
    )


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(logits.astype(np.float64), -60.0, 60.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def _refiner_probability(raw: np.ndarray, prototypes: int) -> np.ndarray:
    logits = raw.reshape(-1, 2, prototypes).max(axis=2)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponential = np.exp(shifted.astype(np.float64))
    return (exponential[:, 1] / exponential.sum(axis=1)).astype(np.float32)


def run_pipeline(
    events: np.ndarray,
    model: H2Export,
    start_ms: float = 0.0,
    duration_ms: float = SEQUENCE_DURATION_MS,
) -> Dict[str, np.ndarray]:
    """Run proposal routing and conditional event refinement on CPU."""
    validate_events(events)
    windows = selected_windows(start_ms, duration_ms)
    patch_features = build_patch_features(events, windows.tolist())
    flat = patch_features.reshape(-1, PROPOSAL_DIMENSIONS[0])
    patch_logits = quantized_mlp(flat, model.proposal_layers)
    patch_scores = _sigmoid(patch_logits).reshape(len(windows), -1, PROPOSAL_DIMENSIONS[-1])
    heatmaps = patch_scores_to_heatmaps(patch_scores)
    candidates = candidate_indices(
        events, windows.tolist(), heatmaps, model.proposal_threshold
    )
    probabilities = np.zeros(len(events), dtype=np.float32)
    if len(candidates):
        refiner_features = build_refiner_features(
            events, candidates, windows.tolist(), heatmaps, model.proposal_threshold
        )
        raw = quantized_mlp(refiner_features, model.refiner_layers)
        probabilities[candidates] = _refiner_probability(raw, model.prototypes)
    return {
        "probabilities": probabilities,
        "predictions": probabilities >= model.event_threshold,
        "candidate_indices": candidates,
        "heatmaps": heatmaps,
        "selected_windows": windows,
    }
