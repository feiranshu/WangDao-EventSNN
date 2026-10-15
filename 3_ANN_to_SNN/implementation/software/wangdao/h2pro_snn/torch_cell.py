"""Differentiable dense reference of the transport operator, not a sparse runtime."""
from __future__ import annotations

import torch
from torch import nn
from typing import cast

from .transport import MODES, TransportConfig


class _Burst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage, threshold):
        ctx.save_for_backward(voltage)
        ctx.threshold = threshold
        return torch.floor(torch.clamp_min(voltage / threshold, 0.0))

    @staticmethod
    def backward(ctx, grad_output):
        voltage, = ctx.saved_tensors
        distance = voltage / ctx.threshold - 1.0
        return grad_output / (ctx.threshold * (1.0 + distance.abs()).square()), None


def burst(voltage: torch.Tensor, threshold: float) -> torch.Tensor:
    return cast(torch.Tensor, _Burst.apply(voltage, threshold))


class TransportLIF(nn.Module):
    """Three matched forward operators with soft reset and a surrogate derivative.

    Forward burst counts are exact; the backward derivative is an approximation.
    Dense tensor execution here must not be reported as an AC-only implementation.
    """
    def __init__(self, config: TransportConfig, mode: str = "fixed"):
        super().__init__()
        if mode not in MODES:
            raise ValueError("Unknown transport mode")
        self.config, self.mode = config, mode
        self.register_buffer("tau_ms", torch.tensor(config.tau, dtype=torch.float64))

    def forward(self, current, state, elapsed_ms, destinations=None):
        shape = (self.config.cells, self.config.channels)
        if (current.shape != shape or state.shape != shape or current.dtype != state.dtype or
                current.device != state.device):
            raise ValueError("Current/state shape or dtype mismatch")
        if not current.is_floating_point() or not torch.isfinite(current).all() or not torch.isfinite(state).all():
            raise ValueError("Current and state must be finite floating tensors")
        elapsed = torch.as_tensor(elapsed_ms, dtype=current.dtype, device=current.device)
        if elapsed.ndim == 0:
            elapsed = elapsed.expand(self.config.cells, 1)
        elif elapsed.shape == (self.config.cells,):
            elapsed = elapsed[:, None]
        if elapsed.shape != (self.config.cells, 1) or not torch.isfinite(elapsed).all() or (elapsed < 0).any():
            raise ValueError("Elapsed milliseconds must be nonnegative, scalar or per-cell")
        if destinations is None:
            destinations = torch.arange(self.config.cells, device=current.device)
        if destinations.dtype != torch.int64 or destinations.shape != (self.config.cells,):
            raise ValueError("Supply exactly one int64 destination per source")
        if destinations.device != current.device:
            raise ValueError("Route indices and state must use the same device")
        self.config.validate_routes(dict(enumerate(destinations.detach().cpu().tolist())))
        local = self.config.local_channels

        def transport(value):
            moved = torch.zeros_like(value[:, local:]).index_add(0, destinations, value[:, local:])
            return torch.cat((value[:, :local], moved), dim=1)

        decayed = state * torch.exp(-elapsed / self.tau_ms.to(current))
        voltage = (transport(decayed) if self.mode == "membrane_route" else decayed) + current
        if not torch.isfinite(voltage).all() or (voltage / self.config.threshold >= 2**63).any():
            raise ValueError("Nonfinite state or unrepresentable spike burst")
        spikes = burst(voltage, self.config.threshold)
        residual = voltage - self.config.threshold * spikes
        return (transport(spikes) if self.mode == "spike_route" else spikes), residual
