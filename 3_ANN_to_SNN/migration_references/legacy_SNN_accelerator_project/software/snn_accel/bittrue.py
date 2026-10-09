from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def signed_bounds(width: int) -> tuple[int, int]:
    if width < 2 or width > 63:
        raise ValueError("width must be in [2, 63]")
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def saturate(value: np.ndarray | int, width: int) -> np.ndarray:
    low, high = signed_bounds(width)
    return np.clip(np.asarray(value, dtype=np.int64), low, high).astype(np.int64)


def exact_add(a: np.ndarray | int, b: np.ndarray | int, width: int) -> np.ndarray:
    return saturate(np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64), width)


def approximate_add(
    a: np.ndarray | int,
    b: np.ndarray | int,
    width: int,
    approx_lsb_bits: int,
) -> np.ndarray:
    """LOA-style signed saturating add used by both software and RTL.

    Low bits are ORed. High signed fields are added without a carry from the
    approximate field, then the composed result is saturated to ``width``.
    """

    if not 1 <= approx_lsb_bits < width:
        raise ValueError("approx_lsb_bits must be in [1, width-1]")
    left = np.asarray(a, dtype=np.int64)
    right = np.asarray(b, dtype=np.int64)
    mask = (1 << approx_lsb_bits) - 1
    low = (left & mask) | (right & mask)
    high = ((left >> approx_lsb_bits) + (right >> approx_lsb_bits)) << approx_lsb_bits
    return saturate(high | low, width)


@dataclass(frozen=True)
class LifResult:
    membrane: np.ndarray
    spike: np.ndarray
    saturated: np.ndarray


def lif_step(
    membrane: np.ndarray | int,
    synaptic_input: np.ndarray | int,
    *,
    threshold: int,
    width: int,
    leak_shift: int = 0,
    approximate: bool = False,
    approx_lsb_bits: int = 4,
    reset_potential: int = 0,
) -> LifResult:
    if threshold <= reset_potential:
        raise ValueError("threshold must be greater than reset_potential")
    if leak_shift < 0:
        raise ValueError("leak_shift cannot be negative")
    old = np.asarray(membrane, dtype=np.int64)
    leak = np.zeros_like(old) if leak_shift == 0 else old >> leak_shift
    decayed = old - leak
    raw_sum = decayed + np.asarray(synaptic_input, dtype=np.int64)
    low, high = signed_bounds(width)
    was_saturated = (raw_sum < low) | (raw_sum > high)
    if approximate:
        updated = approximate_add(decayed, synaptic_input, width, approx_lsb_bits)
    else:
        updated = exact_add(decayed, synaptic_input, width)
    spike = updated >= threshold
    next_membrane = np.where(spike, reset_potential, updated).astype(np.int64)
    return LifResult(membrane=next_membrane, spike=spike.astype(np.bool_), saturated=was_saturated)


def required_accumulator_bits(max_abs_value: int, guard_bits: int = 2) -> int:
    if max_abs_value < 0 or guard_bits < 0:
        raise ValueError("max_abs_value and guard_bits must be non-negative")
    magnitude_bits = max(1, int(max_abs_value).bit_length())
    return min(24, max(16, 1 + magnitude_bits + guard_bits))

