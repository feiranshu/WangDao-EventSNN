"""Exact multi-spike forward, positive-rate STE, and detached soft reset.

The derivative is an explicit surrogate, not the derivative of floor. Every
positive burst interval has the same 1/theta slope, including counts above one.
The transmitted charge theta*n therefore has surrogate gain one, independent
of theta. Reset is detached so weak evidence retains its temporal gradient.
No spike cap or membrane clamp silently discards accumulated evidence.
"""
from typing import cast

import torch


class _RateBurst(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage, threshold):
        ctx.save_for_backward(voltage)
        ctx.threshold = threshold
        return torch.floor(torch.clamp_min(voltage / threshold, 0.0))

    @staticmethod
    def backward(ctx, grad_output):
        voltage, = ctx.saved_tensors
        return grad_output * (voltage > 0).to(voltage.dtype) / ctx.threshold, None


def burst(voltage: torch.Tensor, threshold: float) -> torch.Tensor:
    return cast(torch.Tensor, _RateBurst.apply(voltage, threshold))


def lif(voltage: torch.Tensor, threshold: float):
    count = burst(voltage, threshold)
    return count, voltage - threshold * count.detach()
