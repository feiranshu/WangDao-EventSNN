"""Export modified H2 PyTorch QAT models to the compact NPZ format."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from .constants import EXPORT_FORMAT_VERSION, FEATURE_VERSION, PROTOTYPES
from .numpy_model import load_export
from .torch_model import QuantizedMLP


def quantize_weight(weight: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
    value = weight.detach().cpu().to(torch.float32)
    scale = value.abs().amax(dim=1).clamp_min(1.0e-8) / 127.0
    quantized = torch.clamp(torch.round(value / scale[:, None]), -127.0, 127.0).to(torch.int8)
    return quantized.t().contiguous().numpy(), scale.numpy().astype(np.float32)


def _append_component(arrays: Dict[str, np.ndarray], prefix: str, model: QuantizedMLP) -> None:
    arrays[prefix + "_layer_count"] = np.asarray(len(model.layers), dtype=np.int64)
    for index, layer in enumerate(model.layers):
        if not isinstance(layer, torch.nn.Linear):
            raise TypeError("H2 export supports only Linear layers")
        weight, scale = quantize_weight(layer.weight)
        arrays[f"{prefix}_layer{index}_weight"] = weight
        arrays[f"{prefix}_layer{index}_scale"] = scale


def export_h2(
    proposal: QuantizedMLP,
    refiner: QuantizedMLP,
    output: Path,
    proposal_threshold: float,
    event_threshold: float,
    recipe: str = "D1",
    seed: int = 42,
) -> None:
    arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(EXPORT_FORMAT_VERSION, dtype=np.int64),
        "feature_version": np.asarray(FEATURE_VERSION, dtype=np.int64),
        "variant": np.asarray("H2"),
        "recipe": np.asarray(recipe),
        "seed": np.asarray(seed, dtype=np.int64),
        "proposal_threshold": np.asarray(proposal_threshold, dtype=np.float32),
        "event_threshold": np.asarray(event_threshold, dtype=np.float32),
        "prototypes": np.asarray(PROTOTYPES, dtype=np.int64),
    }
    _append_component(arrays, "proposal", proposal)
    _append_component(arrays, "refiner", refiner)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(output), **arrays)
    load_export(output)
