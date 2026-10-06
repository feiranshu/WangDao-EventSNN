from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PotCodes:
    sign: np.ndarray
    exponent: np.ndarray
    is_zero: np.ndarray
    scale_pow2: int

    def packed_bytes(self) -> bytes:
        """Encode each weight as zero|sign|4-bit biased exponent in one byte.

        bit 7 marks zero, bit 6 is the sign, and bits 3:0 hold exponent+8.
        Bits 5:4 are reserved and zero.
        """

        exponent = np.asarray(self.exponent, dtype=np.int16)
        if np.any((exponent < -8) | (exponent > 7)):
            raise ValueError("packed hardware exponent must be in [-8, 7]")
        packed = (exponent + 8).astype(np.uint8)
        packed |= (np.asarray(self.sign) < 0).astype(np.uint8) << 6
        packed |= np.asarray(self.is_zero, dtype=np.uint8) << 7
        return packed.tobytes(order="C")


def quantize_signed_pot(
    weights: np.ndarray,
    *,
    exponent_min: int = -8,
    exponent_max: int = 7,
    scale_pow2: int = 0,
    zero_threshold: float | None = None,
) -> PotCodes:
    if exponent_min < -8 or exponent_max > 7 or exponent_min > exponent_max:
        raise ValueError("hardware exponent range must be within [-8, 7]")
    values = np.asarray(weights, dtype=np.float64)
    scaled = values / float(2.0**scale_pow2)
    threshold = float(2.0 ** (exponent_min - 1)) if zero_threshold is None else float(zero_threshold)
    is_zero = np.abs(scaled) < threshold
    safe = np.where(is_zero, 1.0, np.abs(scaled))
    exponent = np.rint(np.log2(safe)).astype(np.int8)
    exponent = np.clip(exponent, exponent_min, exponent_max).astype(np.int8)
    sign = np.where(values < 0, -1, 1).astype(np.int8)
    sign = np.where(is_zero, 1, sign).astype(np.int8)
    return PotCodes(sign=sign, exponent=exponent, is_zero=is_zero, scale_pow2=scale_pow2)


def decode_signed_pot(codes: PotCodes) -> np.ndarray:
    magnitude = np.exp2(np.asarray(codes.exponent, dtype=np.float64) + codes.scale_pow2)
    return np.where(codes.is_zero, 0.0, np.asarray(codes.sign) * magnitude)


try:
    import torch

    class _PowerOfTwoSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, weights, exponent_min=-8, exponent_max=7, scale_pow2=0):
            scale = float(2.0**scale_pow2)
            normalized = weights / scale
            exponent = torch.round(torch.log2(torch.clamp(normalized.abs(), min=2.0 ** (exponent_min - 1))))
            exponent = exponent.clamp(exponent_min, exponent_max)
            quantized = normalized.sign() * torch.pow(2.0, exponent) * scale
            return torch.where(normalized.abs() < 2.0 ** (exponent_min - 1), torch.zeros_like(quantized), quantized)

        @staticmethod
        def backward(ctx, gradient):
            return gradient, None, None, None


    def power_of_two_ste(weights, exponent_min: int = -8, exponent_max: int = 7, scale_pow2: int = 0):
        return _PowerOfTwoSTE.apply(weights, exponent_min, exponent_max, scale_pow2)

except ImportError:  # pragma: no cover - training is an optional dependency
    def power_of_two_ste(*args, **kwargs):
        raise RuntimeError("PyTorch is required for QAT")

