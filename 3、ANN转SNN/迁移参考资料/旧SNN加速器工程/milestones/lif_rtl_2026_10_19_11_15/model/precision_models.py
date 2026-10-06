"""Bit-oriented FP32/FP16/INT8 LIF comparison models.

The floating-point path follows the RTL contract:
  candidate = round(round(alpha * membrane) + current)
  spike = candidate >= threshold
  membrane_next = 0 if spike else candidate

The approximate floating-point path replaces only multiplication with the
Mitchell log-domain approximation.  Binary16/binary32 subnormals are flushed
to zero, matching the hardware module's documented FTZ policy.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

from bittrue import approximate_add, exact_add


FORMATS = {
    "fp16": {"code": "e", "width": 16, "exp": 5, "frac": 10},
    "fp32": {"code": "f", "width": 32, "exp": 8, "frac": 23},
}


def _format(fmt: str) -> dict[str, int | str]:
    try:
        return FORMATS[fmt]
    except KeyError as exc:
        raise ValueError(f"unsupported format: {fmt}") from exc


def float_to_bits(value: float, fmt: str) -> int:
    spec = _format(fmt)
    try:
        packed = struct.pack(">" + str(spec["code"]), float(value))
    except OverflowError:
        packed = struct.pack(">" + str(spec["code"]), math.copysign(math.inf, value))
    return int.from_bytes(packed, "big")


def bits_to_float(bits: int, fmt: str) -> float:
    spec = _format(fmt)
    size = int(spec["width"]) // 8
    packed = int(bits).to_bytes(size, "big", signed=False)
    return float(struct.unpack(">" + str(spec["code"]), packed)[0])


def flush_subnormal_bits(bits: int, fmt: str) -> int:
    spec = _format(fmt)
    frac_width = int(spec["frac"])
    exp_width = int(spec["exp"])
    sign = bits >> (frac_width + exp_width)
    exponent = (bits >> frac_width) & ((1 << exp_width) - 1)
    return sign << (frac_width + exp_width) if exponent == 0 else bits


def quantize_ftz(value: float, fmt: str) -> float:
    return bits_to_float(flush_subnormal_bits(float_to_bits(value, fmt), fmt), fmt)


def mitchell_multiply_bits(a_bits: int, b_bits: int, fmt: str) -> int:
    spec = _format(fmt)
    exp_width = int(spec["exp"])
    frac_width = int(spec["frac"])
    exp_max = (1 << exp_width) - 1
    bias = (1 << (exp_width - 1)) - 1
    frac_mask = (1 << frac_width) - 1
    sign_shift = exp_width + frac_width

    sign = ((a_bits >> sign_shift) ^ (b_bits >> sign_shift)) & 1
    exp_a = (a_bits >> frac_width) & exp_max
    exp_b = (b_bits >> frac_width) & exp_max
    frac_a = a_bits & frac_mask
    frac_b = b_bits & frac_mask

    a_zero = exp_a == 0
    b_zero = exp_b == 0
    a_inf = exp_a == exp_max and frac_a == 0
    b_inf = exp_b == exp_max and frac_b == 0
    a_nan = exp_a == exp_max and frac_a != 0
    b_nan = exp_b == exp_max and frac_b != 0

    if a_nan or b_nan or ((a_inf or b_inf) and (a_zero or b_zero)):
        return (exp_max << frac_width) | (1 << (frac_width - 1))
    if a_inf or b_inf:
        return (sign << sign_shift) | (exp_max << frac_width)
    if a_zero or b_zero:
        return sign << sign_shift

    fraction_sum = frac_a + frac_b
    carry = fraction_sum >> frac_width
    exponent = exp_a + exp_b - bias + carry
    if exponent >= exp_max:
        return (sign << sign_shift) | (exp_max << frac_width)
    if exponent <= 0:
        return sign << sign_shift
    return (sign << sign_shift) | (exponent << frac_width) | (fraction_sum & frac_mask)


def mitchell_multiply(a: float, b: float, fmt: str) -> float:
    a_bits = flush_subnormal_bits(float_to_bits(a, fmt), fmt)
    b_bits = flush_subnormal_bits(float_to_bits(b, fmt), fmt)
    return bits_to_float(mitchell_multiply_bits(a_bits, b_bits, fmt), fmt)


@dataclass(frozen=True)
class FloatLifResult:
    candidate: float
    membrane: float
    spike: bool


def float_lif_step(
    membrane: float,
    current: float,
    *,
    alpha: float,
    threshold: float,
    fmt: str,
    approximate: bool = False,
) -> FloatLifResult:
    membrane_q = quantize_ftz(membrane, fmt)
    current_q = quantize_ftz(current, fmt)
    alpha_q = quantize_ftz(alpha, fmt)
    threshold_q = quantize_ftz(threshold, fmt)
    product = (
        mitchell_multiply(membrane_q, alpha_q, fmt)
        if approximate
        else quantize_ftz(membrane_q * alpha_q, fmt)
    )
    candidate = quantize_ftz(product + current_q, fmt)
    spike = not math.isnan(candidate) and candidate >= threshold_q
    return FloatLifResult(candidate, 0.0 if spike else candidate, spike)


@dataclass(frozen=True)
class Int8LifResult:
    candidate: int
    membrane: int
    spike: bool
    saturated: bool


def int8_lif_step(
    membrane: int,
    current: int,
    *,
    threshold: int = 96,
    leak_shift: int = 2,
    approximate: bool = False,
    approx_lsb_bits: int = 2,
) -> Int8LifResult:
    if not -128 <= membrane <= 127 or not -128 <= current <= 127:
        raise ValueError("INT8 operands must be in [-128, 127]")
    leak = 0 if leak_shift == 0 else membrane >> leak_shift
    decayed = membrane - leak
    if approximate:
        candidate, saturated = approximate_add(decayed, current, 8, approx_lsb_bits)
    else:
        candidate, saturated = exact_add(decayed, current, 8)
    spike = candidate >= threshold
    return Int8LifResult(candidate, 0 if spike else candidate, spike, saturated)
