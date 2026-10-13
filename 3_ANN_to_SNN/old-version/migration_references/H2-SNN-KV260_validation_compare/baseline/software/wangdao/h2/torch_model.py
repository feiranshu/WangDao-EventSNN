"""Bias-free PyTorch QAT networks matching the H2 exported model."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from .constants import PROPOSAL_DIMENSIONS, PROTOTYPES, REFINER_DIMENSIONS
from .numpy_model import load_export


UINT8_MAX = 255.0
INT8_MAX = 127.0


def fake_uint8(value: torch.Tensor) -> torch.Tensor:
    clipped = torch.clamp(value, 0.0, 1.0)
    scaled = clipped * UINT8_MAX
    return (scaled + (torch.round(scaled) - scaled).detach()) / UINT8_MAX


def fake_int8_weight(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.detach().abs().amax(dim=1).clamp_min(1.0e-8) / INT8_MAX
    scaled = weight / scale[:, None]
    quantized = scaled + (torch.round(scaled) - scaled).detach()
    return torch.clamp(quantized, -127.0, 127.0) * scale[:, None]


class QuantizedMLP(nn.Module):
    def __init__(self, dimensions: Sequence[int], quantized: bool = False) -> None:
        super().__init__()
        self.dimensions = tuple(int(value) for value in dimensions)
        if len(self.dimensions) < 2 or any(value <= 0 for value in self.dimensions):
            raise ValueError("dimensions must contain at least two positive values")
        self.layers = nn.ModuleList(
            nn.Linear(self.dimensions[index], self.dimensions[index + 1], bias=False)
            for index in range(len(self.dimensions) - 1)
        )
        self.quantized = bool(quantized)

    def set_quantized(self, enabled: bool) -> None:
        self.quantized = bool(enabled)

    def forward(self, features_uint8: torch.Tensor) -> torch.Tensor:
        if features_uint8.ndim != 2 or features_uint8.shape[1] != self.dimensions[0]:
            raise ValueError("MLP input shape does not match declared dimensions")
        value = features_uint8.to(torch.float32) / UINT8_MAX
        if self.quantized:
            value = fake_uint8(value)
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, nn.Linear):
                raise TypeError("QuantizedMLP supports only Linear layers")
            weight = fake_int8_weight(layer.weight) if self.quantized else layer.weight
            value = functional.linear(value, weight)
            if index + 1 < len(self.layers):
                value = functional.hardtanh(value, 0.0, 1.0)
                if self.quantized:
                    value = fake_uint8(value)
        return value


def build_proposal(quantized: bool = False) -> QuantizedMLP:
    return QuantizedMLP(PROPOSAL_DIMENSIONS, quantized=quantized)


def build_refiner(quantized: bool = False) -> QuantizedMLP:
    return QuantizedMLP(REFINER_DIMENSIONS, quantized=quantized)


def prototype_logits(raw: torch.Tensor, prototypes: int = PROTOTYPES) -> torch.Tensor:
    if raw.ndim != 2 or raw.shape[1] != 2 * prototypes:
        raise ValueError("refiner output does not match the prototype contract")
    return raw.reshape(-1, 2, prototypes).amax(dim=2)


def load_exported_models(path: Path) -> tuple[QuantizedMLP, QuantizedMLP]:
    """Reconstruct QAT-compatible PyTorch models from the locked NPZ."""
    exported = load_export(path)
    proposal = build_proposal(quantized=True)
    refiner = build_refiner(quantized=True)
    with torch.no_grad():
        for model, layers in (
            (proposal, exported.proposal_layers),
            (refiner, exported.refiner_layers),
        ):
            for torch_layer, (weight, scale) in zip(model.layers, layers):
                if not isinstance(torch_layer, nn.Linear):
                    raise TypeError("QuantizedMLP supports only Linear layers")
                restored = weight.astype(np.float32) * scale[None, :]
                torch_layer.weight.copy_(torch.from_numpy(restored.T.copy()))
    return proposal, refiner
