"""Bit-true reference for the standalone LIF RTL milestone."""

from __future__ import annotations

from dataclasses import dataclass


def signed_bounds(width: int) -> tuple[int, int]:
    if width < 2:
        raise ValueError("width must be at least two")
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def saturate(value: int, width: int) -> tuple[int, bool]:
    minimum, maximum = signed_bounds(width)
    if value < minimum:
        return minimum, True
    if value > maximum:
        return maximum, True
    return int(value), False


def exact_add(a: int, b: int, width: int) -> tuple[int, bool]:
    return saturate(a + b, width)


def approximate_add(a: int, b: int, width: int, approx_lsb_bits: int) -> tuple[int, bool]:
    if not 1 <= approx_lsb_bits < width:
        raise ValueError("approx_lsb_bits must be in [1, width-1]")
    mask = (1 << approx_lsb_bits) - 1
    low = (a & mask) | (b & mask)
    high = ((a >> approx_lsb_bits) + (b >> approx_lsb_bits)) << approx_lsb_bits
    return saturate(high | low, width)


def int8_mac(accumulator: int, activation: int, weight: int, width: int = 20) -> tuple[int, bool]:
    if not -128 <= activation <= 127 or not -128 <= weight <= 127:
        raise ValueError("activation and weight must be signed INT8")
    return exact_add(accumulator, activation * weight, width)


def pot_contribution(
    *,
    spike: bool,
    weight_zero: bool,
    weight_negative: bool,
    exponent: int,
    layer_scale: int,
    width: int = 20,
    fraction_bits: int = 8,
) -> tuple[int, bool]:
    if not -8 <= exponent <= 7:
        raise ValueError("four-bit hardware exponent must be in [-8, 7]")
    if not spike or weight_zero:
        return 0, False

    bit_index = fraction_bits + exponent + layer_scale
    if bit_index < 0:
        return 0, True

    _, maximum = signed_bounds(width)
    magnitude = 1 << bit_index
    value = -magnitude if weight_negative else magnitude
    saturated_value, saturated_flag = saturate(value, width)
    # Positive 2^(width-1) and any larger magnitude are not representable.
    if not weight_negative and magnitude > maximum:
        saturated_flag = True
    return saturated_value, saturated_flag


def pot_accumulate(
    accumulator: int,
    *,
    spike: bool,
    weight_zero: bool,
    weight_negative: bool,
    exponent: int,
    layer_scale: int,
    width: int = 20,
    fraction_bits: int = 8,
    approximate: bool = False,
    approx_lsb_bits: int = 4,
) -> tuple[int, bool]:
    contribution, contribution_saturated = pot_contribution(
        spike=spike,
        weight_zero=weight_zero,
        weight_negative=weight_negative,
        exponent=exponent,
        layer_scale=layer_scale,
        width=width,
        fraction_bits=fraction_bits,
    )
    if approximate:
        result, add_saturated = approximate_add(accumulator, contribution, width, approx_lsb_bits)
    else:
        result, add_saturated = exact_add(accumulator, contribution, width)
    return result, contribution_saturated or add_saturated


@dataclass(frozen=True)
class LifResult:
    membrane: int
    spike: bool
    saturated: bool


def lif_step(
    membrane: int,
    synaptic_input: int,
    *,
    threshold: int,
    width: int = 20,
    leak_shift: int = 3,
    approximate: bool = False,
    approx_lsb_bits: int = 4,
    reset_potential: int = 0,
    subtractive_reset: bool = False,
    soft_reset: bool = False,
) -> LifResult:
    if threshold <= reset_potential:
        raise ValueError("threshold must exceed reset_potential")
    if leak_shift < 0:
        raise ValueError("leak_shift cannot be negative")
    if soft_reset:
        return LifResult(reset_potential, False, False)

    leak = 0 if leak_shift == 0 else membrane >> leak_shift
    decayed = membrane - leak
    if approximate:
        updated, add_saturated = approximate_add(decayed, synaptic_input, width, approx_lsb_bits)
    else:
        updated, add_saturated = exact_add(decayed, synaptic_input, width)
    spike = updated >= threshold
    if not spike:
        return LifResult(updated, False, add_saturated)
    if subtractive_reset:
        reset_value, reset_saturated = saturate(updated - threshold, width)
        return LifResult(reset_value, True, add_saturated or reset_saturated)
    return LifResult(reset_potential, True, add_saturated)

