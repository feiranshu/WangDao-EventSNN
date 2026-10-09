"""面向 RTL 对照的 H2 与 Proposal-SNN 位精确整数模型。"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Sequence, Tuple

import numpy as np
import torch

from .h2.numpy_model import H2Export, Layer
from .snn import ProposalSNN


FIXED_S0P_EXPORT_VERSION = 1
POT_S0P_EXPORT_VERSION = 1
APPROXIMATE_S0P_EXPORT_VERSION = 1
POT_ZERO_EXPONENT = 127


def signed_round_divide(value: np.ndarray, divisor: int) -> np.ndarray:
    """有符号整数除法，绝对值正中间时远离零舍入。"""
    if divisor <= 0:
        raise ValueError("除数必须为正数")
    source = np.asarray(value, dtype=np.int64)
    magnitude = (np.abs(source) + divisor // 2) // divisor
    return np.where(source < 0, -magnitude, magnitude).astype(np.int64)


def signed_round_shift(value: np.ndarray, bits: int) -> np.ndarray:
    """有符号移位；右移使用最近舍入，左移保持精确。"""
    source = np.asarray(value, dtype=np.int64)
    if bits == 0:
        return source
    if bits > 0:
        return signed_round_divide(source, 1 << bits)
    return np.left_shift(source, -bits)


def saturate_signed(value: np.ndarray, bits: int) -> Tuple[np.ndarray, int]:
    if bits < 2 or bits > 62:
        raise ValueError("有符号位宽必须在 2 到 62 之间")
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    source = np.asarray(value, dtype=np.int64)
    count = int(np.count_nonzero((source < minimum) | (source > maximum)))
    return np.clip(source, minimum, maximum).astype(np.int64), count


class FixedMLPOutput(NamedTuple):
    logits_q: np.ndarray
    logits: np.ndarray
    accumulator_saturations: Tuple[int, ...]


@dataclass(frozen=True)
class FixedH2Config:
    accumulator_bits: int = 24
    scale_fraction_bits: int = 20
    logit_fraction_bits: int = 16

    def __post_init__(self) -> None:
        if not 8 <= self.accumulator_bits <= 32:
            raise ValueError("H2 累加器位宽必须在 8 到 32 之间")
        if not 8 <= self.scale_fraction_bits <= 30:
            raise ValueError("H2 缩放系数小数位必须在 8 到 30 之间")
        if not 8 <= self.logit_fraction_bits <= 24:
            raise ValueError("H2 输出小数位必须在 8 到 24 之间")


class FixedH2:
    """保持 H2 int8 权重，以整数缩放替代运行时浮点缩放。"""

    def __init__(self, source: H2Export, configuration: FixedH2Config) -> None:
        self.source = source
        self.configuration = configuration
        factor = 1 << configuration.scale_fraction_bits
        self.proposal_multipliers = tuple(
            np.rint(scale.astype(np.float64) * factor).astype(np.int64)
            for _weight, scale in source.proposal_layers
        )
        self.refiner_multipliers = tuple(
            np.rint(scale.astype(np.float64) * factor).astype(np.int64)
            for _weight, scale in source.refiner_layers
        )

    def _forward(
        self,
        features: np.ndarray,
        layers: Tuple[Layer, ...],
        multipliers: Tuple[np.ndarray, ...],
    ) -> FixedMLPOutput:
        value = np.ascontiguousarray(features, dtype=np.uint8)
        saturation_counts = []
        logits_q = np.empty((len(value), layers[-1][0].shape[1]), dtype=np.int64)
        scale_denominator = 1 << self.configuration.scale_fraction_bits
        for index, ((weight, _scale), multiplier) in enumerate(
            zip(layers, multipliers)
        ):
            raw = np.matmul(value.astype(np.int64), weight.astype(np.int64))
            raw, saturation_count = saturate_signed(
                raw, self.configuration.accumulator_bits
            )
            saturation_counts.append(saturation_count)
            scaled = raw * multiplier[None, :]
            if index + 1 < len(layers):
                value = np.clip(
                    signed_round_divide(scaled, scale_denominator), 0, 255
                ).astype(np.uint8)
            else:
                numerator = scaled * (1 << self.configuration.logit_fraction_bits)
                logits_q = signed_round_divide(
                    numerator, scale_denominator * 255
                )
        logits = (
            logits_q.astype(np.float64)
            / float(1 << self.configuration.logit_fraction_bits)
        ).astype(np.float32)
        return FixedMLPOutput(
            logits_q=logits_q,
            logits=logits,
            accumulator_saturations=tuple(saturation_counts),
        )

    def proposal(self, features: np.ndarray) -> FixedMLPOutput:
        return self._forward(
            features, self.source.proposal_layers, self.proposal_multipliers
        )

    def refiner(self, features: np.ndarray) -> FixedMLPOutput:
        return self._forward(
            features, self.source.refiner_layers, self.refiner_multipliers
        )


@dataclass(frozen=True)
class FixedSNNConfig:
    accumulator_bits: int = 24
    membrane_bits: int = 24
    membrane_fraction_bits: int = 12
    input_fraction_bits: int = 8
    beta_numerator: int = 7
    beta_shift: int = 3

    def __post_init__(self) -> None:
        if not 8 <= self.accumulator_bits <= 32:
            raise ValueError("SNN 累加器位宽必须在 8 到 32 之间")
        if not 8 <= self.membrane_bits <= 32:
            raise ValueError("膜电位位宽必须在 8 到 32 之间")
        if not 4 <= self.membrane_fraction_bits < self.membrane_bits:
            raise ValueError("膜电位小数位必须小于膜电位位宽")
        if self.input_fraction_bits != 8:
            raise ValueError("当前 S1 输入固定使用 uint8/Q0.8")
        if self.beta_numerator <= 0 or self.beta_shift <= 0:
            raise ValueError("泄漏系数必须是正的整数/二次幂")


class FixedSNNOutput(NamedTuple):
    logits_q: np.ndarray
    logits: np.ndarray
    spike_counts: Tuple[int, int, int]
    accumulator_saturations: Tuple[int, int, int]
    membrane_saturations: Tuple[int, int, int, int]


class ApproximateSNNOutput(NamedTuple):
    logits_q: np.ndarray
    logits: np.ndarray
    spike_counts: Tuple[int, int, int]
    accumulator_saturations: Tuple[int, int, int]
    membrane_saturations: Tuple[int, int, int, int]
    approximate_additions: Tuple[int, int, int, int, int, int]


def loa_add_signed(
    first: np.ndarray | int,
    second: np.ndarray | int,
    bits: int,
    low_bits: int,
) -> Tuple[np.ndarray, int]:
    """按有符号二进制补码执行 LOA，并在输出端饱和。"""
    if bits < 2 or bits > 62:
        raise ValueError("LOA 位宽必须在 2 到 62 之间")
    if low_bits < 0 or low_bits >= bits:
        raise ValueError("LOA 近似低位数必须在 0 到位宽减一之间")
    left, right = np.broadcast_arrays(
        np.asarray(first, dtype=np.int64), np.asarray(second, dtype=np.int64)
    )
    if low_bits == 0:
        return saturate_signed(left + right, bits)

    low_mask = (1 << low_bits) - 1
    low = np.bitwise_and(np.bitwise_or(left, right), low_mask)
    carry = np.bitwise_and(
        np.bitwise_and(np.right_shift(left, low_bits - 1), 1),
        np.bitwise_and(np.right_shift(right, low_bits - 1), 1),
    )
    high = (
        np.right_shift(left, low_bits)
        + np.right_shift(right, low_bits)
        + carry
    )
    combined = np.left_shift(high, low_bits) + low
    return saturate_signed(combined, bits)


def fixed_balanced_add_tree(
    values: np.ndarray,
    bits: int,
    low_bits: int,
) -> Tuple[np.ndarray, int, int]:
    """沿末维按固定索引顺序执行平衡加法树。"""
    source = np.asarray(values, dtype=np.int64)
    if source.ndim == 0 or source.shape[-1] == 0:
        raise ValueError("固定加法树至少需要一个输入")
    current, saturation_count = saturate_signed(source, bits)
    addition_count = 0
    while current.shape[-1] > 1:
        pair_count = current.shape[-1] // 2
        paired, count = loa_add_signed(
            current[..., : pair_count * 2 : 2],
            current[..., 1 : pair_count * 2 : 2],
            bits,
            low_bits,
        )
        saturation_count += count
        addition_count += int(paired.size)
        if current.shape[-1] % 2:
            current = np.concatenate((paired, current[..., -1:]), axis=-1)
        else:
            current = paired
    return current[..., 0], saturation_count, addition_count


@dataclass(frozen=True)
class LOAPlacement:
    low_bits: int
    synaptic_layers: Tuple[bool, bool, bool] = (False, False, False)
    membrane_updates: Tuple[bool, bool, bool] = (False, False, False)

    def __post_init__(self) -> None:
        if self.low_bits < 0:
            raise ValueError("LOA 近似低位数不能为负数")
        if len(self.synaptic_layers) != 3:
            raise ValueError("必须为三个突触层分别指定是否近似")
        if len(self.membrane_updates) != 3:
            raise ValueError("必须为三个推理膜更新分别指定是否近似")


def _quantize_weight_per_output(weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    source = np.asarray(weight, dtype=np.float64)
    if source.ndim != 2:
        raise ValueError("权重必须是二维矩阵")
    maximum = np.max(np.abs(source), axis=1)
    fraction_bits = np.zeros(len(maximum), dtype=np.int8)
    positive = maximum > 0.0
    fraction_bits[positive] = np.floor(
        np.log2(127.0 / maximum[positive])
    ).astype(np.int8)
    fraction_bits = np.clip(fraction_bits, -8, 20).astype(np.int8)
    scaled = source * np.exp2(fraction_bits.astype(np.float64))[:, None]
    quantized = np.clip(np.rint(scaled), -127, 127).astype(np.int8)
    return np.ascontiguousarray(quantized), fraction_bits


def _shift_channels(
    raw: np.ndarray, shifts: np.ndarray
) -> np.ndarray:
    result = np.empty(raw.shape, dtype=np.int64)
    for channel, shift in enumerate(shifts.tolist()):
        result[:, channel] = signed_round_shift(raw[:, channel], -int(shift))
    return result


class FixedProposalSNN:
    """S0-P 的整数权重、整数膜状态和确定性软复位模型。"""

    def __init__(
        self,
        weights: Tuple[np.ndarray, np.ndarray, np.ndarray],
        weight_fraction_bits: Tuple[np.ndarray, np.ndarray, np.ndarray],
        threshold: float,
        time_steps: int,
        readout: str,
        configuration: FixedSNNConfig,
    ) -> None:
        if len(weights) != 3 or len(weight_fraction_bits) != 3:
            raise ValueError("S1 Proposal 必须包含三层权重")
        if readout != "mean_membrane":
            raise ValueError("当前 S1 只冻结 mean_membrane 读出")
        if time_steps <= 0 or (time_steps & (time_steps - 1)):
            raise ValueError("时间步必须是正的二次幂")
        self.weights = tuple(np.ascontiguousarray(value, dtype=np.int8) for value in weights)
        self.weight_fraction_bits = tuple(
            np.ascontiguousarray(value, dtype=np.int8)
            for value in weight_fraction_bits
        )
        self.threshold = float(threshold)
        self.threshold_q = int(
            round(threshold * (1 << configuration.membrane_fraction_bits))
        )
        self.time_steps = int(time_steps)
        self.readout = readout
        self.configuration = configuration

    @classmethod
    def from_float(
        cls,
        model: ProposalSNN,
        time_steps: int,
        configuration: FixedSNNConfig,
    ) -> "FixedProposalSNN":
        weights = []
        fractions = []
        for layer in model.layers:
            weight = getattr(layer, "weight", None)
            if weight is None:
                raise TypeError("S0-P 必须由线性层组成")
            quantized, fraction_bits = _quantize_weight_per_output(
                weight.detach().cpu().numpy()
            )
            weights.append(quantized)
            fractions.append(fraction_bits)
        state = model.state_dict()
        beta = float(state["hidden_cells.0.beta"].item())
        expected_beta = configuration.beta_numerator / float(
            1 << configuration.beta_shift
        )
        if not math.isclose(beta, expected_beta, rel_tol=0.0, abs_tol=1e-8):
            raise ValueError("冻结泄漏系数与整数配置不一致")
        threshold = float(state["hidden_cells.0.threshold"].item())
        return cls(
            weights=(weights[0], weights[1], weights[2]),
            weight_fraction_bits=(fractions[0], fractions[1], fractions[2]),
            threshold=threshold,
            time_steps=time_steps,
            readout=model.readout,
            configuration=configuration,
        )

    def _current(
        self, value: np.ndarray, layer: int, input_fraction_bits: int
    ) -> Tuple[np.ndarray, int]:
        raw = np.matmul(
            value.astype(np.int64), self.weights[layer].T.astype(np.int64)
        )
        raw, saturation_count = saturate_signed(
            raw, self.configuration.accumulator_bits
        )
        shifts = (
            self.configuration.membrane_fraction_bits
            - input_fraction_bits
            - self.weight_fraction_bits[layer].astype(np.int64)
        )
        return _shift_channels(raw, shifts), saturation_count

    def _update(
        self,
        current: np.ndarray,
        membrane: np.ndarray,
        fire: bool,
        membrane_slot: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        leaked = signed_round_shift(
            membrane * self.configuration.beta_numerator,
            self.configuration.beta_shift,
        )
        updated, saturation_count = saturate_signed(
            leaked + current, self.configuration.membrane_bits
        )
        spikes = updated >= self.threshold_q if fire else np.zeros_like(updated, dtype=bool)
        if fire:
            updated = updated - spikes.astype(np.int64) * self.threshold_q
        return spikes.astype(np.uint8), updated, saturation_count

    def forward(self, features: np.ndarray) -> FixedSNNOutput:
        value = np.asarray(features)
        if value.ndim != 3 or value.shape[0] != self.time_steps:
            raise ValueError("S1 输入必须具有 [time, batch, 128] 形状")
        if value.shape[2] != self.weights[0].shape[1]:
            raise ValueError("S1 输入维度不匹配")
        value = np.ascontiguousarray(value, dtype=np.uint8)
        batch = value.shape[1]
        hidden_membranes = [
            np.zeros((batch, self.weights[0].shape[0]), dtype=np.int64),
            np.zeros((batch, self.weights[1].shape[0]), dtype=np.int64),
        ]
        readout_membrane = np.zeros(
            (batch, self.weights[2].shape[0]), dtype=np.int64
        )
        spike_membrane = np.zeros_like(readout_membrane)
        readout_sum = np.zeros_like(readout_membrane)
        spike_counts = [0, 0, 0]
        accumulator_saturations = [0, 0, 0]
        membrane_saturations = [0, 0, 0, 0]
        for step in range(self.time_steps):
            current, count = self._current(
                value[step], 0, self.configuration.input_fraction_bits
            )
            accumulator_saturations[0] += count
            spikes_1, hidden_membranes[0], count = self._update(
                current, hidden_membranes[0], True, 0
            )
            membrane_saturations[0] += count
            spike_counts[0] += int(spikes_1.sum())

            current, count = self._current(spikes_1, 1, 0)
            accumulator_saturations[1] += count
            spikes_2, hidden_membranes[1], count = self._update(
                current, hidden_membranes[1], True, 1
            )
            membrane_saturations[1] += count
            spike_counts[1] += int(spikes_2.sum())

            output_current, count = self._current(spikes_2, 2, 0)
            accumulator_saturations[2] += count
            _unused, readout_membrane, count = self._update(
                output_current, readout_membrane, False, 2
            )
            membrane_saturations[2] += count
            readout_sum += readout_membrane
            output_spikes, spike_membrane, count = self._update(
                output_current, spike_membrane, True
            )
            membrane_saturations[3] += count
            spike_counts[2] += int(output_spikes.sum())

        logits_q = signed_round_divide(readout_sum, self.time_steps)
        logits = (
            logits_q.astype(np.float64)
            / float(1 << self.configuration.membrane_fraction_bits)
        ).astype(np.float32)
        return FixedSNNOutput(
            logits_q=logits_q,
            logits=logits,
            spike_counts=(spike_counts[0], spike_counts[1], spike_counts[2]),
            accumulator_saturations=(
                accumulator_saturations[0],
                accumulator_saturations[1],
                accumulator_saturations[2],
            ),
            membrane_saturations=(
                membrane_saturations[0],
                membrane_saturations[1],
                membrane_saturations[2],
                membrane_saturations[3],
            ),
        )


class ApproximateProposalSNN(FixedProposalSNN):
    """在 S1 指定加法位置插入 24 位 LOA 的软件黄金模型。"""

    def __init__(self, source: FixedProposalSNN, placement: LOAPlacement) -> None:
        if source.configuration.accumulator_bits != 24:
            raise ValueError("S3 首轮要求 24 位突触累加器")
        if source.configuration.membrane_bits != 24:
            raise ValueError("S3 首轮要求 24 位膜电位")
        if placement.low_bits >= 24:
            raise ValueError("S3 近似低位数必须小于 24")
        super().__init__(
            weights=(source.weights[0], source.weights[1], source.weights[2]),
            weight_fraction_bits=(
                source.weight_fraction_bits[0],
                source.weight_fraction_bits[1],
                source.weight_fraction_bits[2],
            ),
            threshold=source.threshold,
            time_steps=source.time_steps,
            readout=source.readout,
            configuration=source.configuration,
        )
        self.placement = placement
        self._approximate_additions = [0, 0, 0, 0, 0, 0]

    def _current(
        self, value: np.ndarray, layer: int, input_fraction_bits: int
    ) -> Tuple[np.ndarray, int]:
        if not self.placement.synaptic_layers[layer]:
            return super()._current(value, layer, input_fraction_bits)
        products = (
            value.astype(np.int64)[:, None, :]
            * self.weights[layer].astype(np.int64)[None, :, :]
        )
        raw, saturation_count, addition_count = fixed_balanced_add_tree(
            products,
            self.configuration.accumulator_bits,
            self.placement.low_bits,
        )
        if self.placement.low_bits:
            self._approximate_additions[layer] += addition_count
        shifts = (
            self.configuration.membrane_fraction_bits
            - input_fraction_bits
            - self.weight_fraction_bits[layer].astype(np.int64)
        )
        return _shift_channels(raw, shifts), saturation_count

    def _update(
        self,
        current: np.ndarray,
        membrane: np.ndarray,
        fire: bool,
        membrane_slot: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        leaked = signed_round_shift(
            membrane * self.configuration.beta_numerator,
            self.configuration.beta_shift,
        )
        approximate = (
            membrane_slot is not None
            and self.placement.membrane_updates[membrane_slot]
        )
        if approximate:
            assert membrane_slot is not None
            updated, saturation_count = loa_add_signed(
                leaked,
                current,
                self.configuration.membrane_bits,
                self.placement.low_bits,
            )
            if self.placement.low_bits:
                self._approximate_additions[3 + membrane_slot] += int(updated.size)
        else:
            updated, saturation_count = saturate_signed(
                leaked + current, self.configuration.membrane_bits
            )
        spikes = (
            updated >= self.threshold_q
            if fire
            else np.zeros_like(updated, dtype=bool)
        )
        if fire:
            updated = updated - spikes.astype(np.int64) * self.threshold_q
        return spikes.astype(np.uint8), updated, saturation_count

    def forward(self, features: np.ndarray) -> ApproximateSNNOutput:
        self._approximate_additions = [0, 0, 0, 0, 0, 0]
        output = super().forward(features)
        return ApproximateSNNOutput(
            logits_q=output.logits_q,
            logits=output.logits,
            spike_counts=output.spike_counts,
            accumulator_saturations=output.accumulator_saturations,
            membrane_saturations=output.membrane_saturations,
            approximate_additions=(
                self._approximate_additions[0],
                self._approximate_additions[1],
                self._approximate_additions[2],
                self._approximate_additions[3],
                self._approximate_additions[4],
                self._approximate_additions[5],
            ),
        )


def _torch_saturate_signed(
    value: torch.Tensor, bits: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    minimum = -(1 << (bits - 1))
    maximum = (1 << (bits - 1)) - 1
    count = torch.count_nonzero((value < minimum) | (value > maximum))
    return torch.clamp(value, minimum, maximum), count


def _torch_signed_round_divide(value: torch.Tensor, divisor: int) -> torch.Tensor:
    magnitude = torch.div(
        torch.abs(value) + divisor // 2, divisor, rounding_mode="floor"
    )
    return torch.where(value < 0, -magnitude, magnitude)


def _torch_loa_add_signed(
    first: torch.Tensor,
    second: torch.Tensor,
    bits: int,
    low_bits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if low_bits == 0:
        return _torch_saturate_signed(first + second, bits)
    low_mask = (1 << low_bits) - 1
    low = torch.bitwise_and(torch.bitwise_or(first, second), low_mask)
    carry = torch.bitwise_and(
        torch.bitwise_and(torch.bitwise_right_shift(first, low_bits - 1), 1),
        torch.bitwise_and(torch.bitwise_right_shift(second, low_bits - 1), 1),
    )
    high = (
        torch.bitwise_right_shift(first, low_bits)
        + torch.bitwise_right_shift(second, low_bits)
        + carry
    )
    combined = torch.bitwise_left_shift(high, low_bits) + low
    return _torch_saturate_signed(combined, bits)


def _torch_fixed_balanced_add_tree(
    values: torch.Tensor,
    bits: int,
    low_bits: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("固定加法树至少需要一个输入")
    current, saturation_count = _torch_saturate_signed(values, bits)
    addition_count = 0
    while current.shape[-1] > 1:
        pair_count = current.shape[-1] // 2
        paired, count = _torch_loa_add_signed(
            current[..., : pair_count * 2 : 2],
            current[..., 1 : pair_count * 2 : 2],
            bits,
            low_bits,
        )
        saturation_count = saturation_count + count
        addition_count += paired.numel()
        if current.shape[-1] % 2:
            current = torch.cat((paired, current[..., -1:]), dim=-1)
        else:
            current = paired
    return current[..., 0], saturation_count, addition_count


class TorchApproximateProposalSNN:
    """与 NumPy 黄金模型逐位对齐的加速整数后端。"""

    def __init__(
        self,
        source: FixedProposalSNN,
        placement: LOAPlacement,
        device: str | torch.device,
    ) -> None:
        if source.configuration.accumulator_bits != 24:
            raise ValueError("S3 首轮要求 24 位突触累加器")
        if source.configuration.membrane_bits != 24:
            raise ValueError("S3 首轮要求 24 位膜电位")
        if placement.low_bits >= 24:
            raise ValueError("S3 近似低位数必须小于 24")
        self.configuration = source.configuration
        self.placement = placement
        self.time_steps = source.time_steps
        self.readout = source.readout
        self.threshold_q = source.threshold_q
        self.device = torch.device(device)
        self.weights = tuple(
            torch.as_tensor(
                value.astype(np.int32), dtype=torch.int32, device=self.device
            )
            for value in source.weights
        )
        self.weight_fraction_bits = source.weight_fraction_bits

    def _shift_channels(self, raw: torch.Tensor, shifts: np.ndarray) -> torch.Tensor:
        result = torch.empty_like(raw)
        for shift in np.unique(shifts).tolist():
            channels = np.flatnonzero(shifts == shift).tolist()
            selected = raw[:, channels]
            integer_shift = int(shift)
            if integer_shift > 0:
                shifted = torch.bitwise_left_shift(selected, integer_shift)
            elif integer_shift < 0:
                shifted = _torch_signed_round_divide(
                    selected, 1 << -integer_shift
                )
            else:
                shifted = selected
            result[:, channels] = shifted
        return result

    def _current(
        self, value: torch.Tensor, layer: int, input_fraction_bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        products = value[:, None, :] * self.weights[layer][None, :, :]
        if self.placement.synaptic_layers[layer]:
            raw, saturation_count, addition_count = _torch_fixed_balanced_add_tree(
                products,
                self.configuration.accumulator_bits,
                self.placement.low_bits,
            )
        else:
            raw = torch.sum(products, dim=-1, dtype=torch.int32)
            raw, saturation_count = _torch_saturate_signed(
                raw, self.configuration.accumulator_bits
            )
            addition_count = 0
        shifts = (
            self.configuration.membrane_fraction_bits
            - input_fraction_bits
            - self.weight_fraction_bits[layer].astype(np.int64)
        )
        current = self._shift_channels(raw, shifts)
        return current, saturation_count, addition_count

    def _update(
        self,
        current: torch.Tensor,
        membrane: torch.Tensor,
        fire: bool,
        membrane_slot: int | None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        leaked = _torch_signed_round_divide(
            membrane * self.configuration.beta_numerator,
            1 << self.configuration.beta_shift,
        )
        approximate = (
            membrane_slot is not None
            and self.placement.membrane_updates[membrane_slot]
        )
        if approximate:
            updated, saturation_count = _torch_loa_add_signed(
                leaked,
                current,
                self.configuration.membrane_bits,
                self.placement.low_bits,
            )
            addition_count = updated.numel()
        else:
            updated, saturation_count = _torch_saturate_signed(
                leaked + current, self.configuration.membrane_bits
            )
            addition_count = 0
        spikes = (
            updated >= self.threshold_q
            if fire
            else torch.zeros_like(updated, dtype=torch.bool)
        )
        if fire:
            updated = updated - spikes.to(torch.int32) * self.threshold_q
        return spikes.to(torch.int32), updated, saturation_count, addition_count

    def forward(self, features: np.ndarray) -> ApproximateSNNOutput:
        source = np.asarray(features)
        if source.ndim != 3 or source.shape[0] != self.time_steps:
            raise ValueError("S3 输入必须具有 [time, batch, 128] 形状")
        if source.shape[2] != self.weights[0].shape[1]:
            raise ValueError("S3 输入维度不匹配")
        value = torch.as_tensor(
            np.ascontiguousarray(source, dtype=np.uint8),
            dtype=torch.int32,
            device=self.device,
        )
        batch = value.shape[1]
        hidden_membranes = [
            torch.zeros(
                (batch, self.weights[0].shape[0]),
                dtype=torch.int32,
                device=self.device,
            ),
            torch.zeros(
                (batch, self.weights[1].shape[0]),
                dtype=torch.int32,
                device=self.device,
            ),
        ]
        readout_membrane = torch.zeros(
            (batch, self.weights[2].shape[0]),
            dtype=torch.int32,
            device=self.device,
        )
        spike_membrane = torch.zeros_like(readout_membrane)
        readout_sum = torch.zeros_like(readout_membrane)
        spike_counts = [torch.zeros((), dtype=torch.int64, device=self.device) for _ in range(3)]
        accumulator_saturations = [
            torch.zeros((), dtype=torch.int64, device=self.device) for _ in range(3)
        ]
        membrane_saturations = [
            torch.zeros((), dtype=torch.int64, device=self.device) for _ in range(4)
        ]
        approximate_additions = [0, 0, 0, 0, 0, 0]

        for step in range(self.time_steps):
            current, count, additions = self._current(
                value[step], 0, self.configuration.input_fraction_bits
            )
            accumulator_saturations[0] += count
            if self.placement.low_bits:
                approximate_additions[0] += additions
            spikes_1, hidden_membranes[0], count, additions = self._update(
                current, hidden_membranes[0], True, 0
            )
            membrane_saturations[0] += count
            spike_counts[0] += spikes_1.sum(dtype=torch.int64)
            if self.placement.low_bits:
                approximate_additions[3] += additions

            current, count, additions = self._current(spikes_1, 1, 0)
            accumulator_saturations[1] += count
            if self.placement.low_bits:
                approximate_additions[1] += additions
            spikes_2, hidden_membranes[1], count, additions = self._update(
                current, hidden_membranes[1], True, 1
            )
            membrane_saturations[1] += count
            spike_counts[1] += spikes_2.sum(dtype=torch.int64)
            if self.placement.low_bits:
                approximate_additions[4] += additions

            output_current, count, additions = self._current(spikes_2, 2, 0)
            accumulator_saturations[2] += count
            if self.placement.low_bits:
                approximate_additions[2] += additions
            _unused, readout_membrane, count, additions = self._update(
                output_current, readout_membrane, False, 2
            )
            membrane_saturations[2] += count
            if self.placement.low_bits:
                approximate_additions[5] += additions
            readout_sum += readout_membrane
            output_spikes, spike_membrane, count, _additions = self._update(
                output_current, spike_membrane, True, None
            )
            membrane_saturations[3] += count
            spike_counts[2] += output_spikes.sum(dtype=torch.int64)

        logits_q = _torch_signed_round_divide(readout_sum, self.time_steps)
        logits_q_numpy = logits_q.to(torch.int64).cpu().numpy()
        logits = (
            logits_q_numpy.astype(np.float64)
            / float(1 << self.configuration.membrane_fraction_bits)
        ).astype(np.float32)
        return ApproximateSNNOutput(
            logits_q=logits_q_numpy,
            logits=logits,
            spike_counts=(
                int(spike_counts[0].item()),
                int(spike_counts[1].item()),
                int(spike_counts[2].item()),
            ),
            accumulator_saturations=(
                int(accumulator_saturations[0].item()),
                int(accumulator_saturations[1].item()),
                int(accumulator_saturations[2].item()),
            ),
            membrane_saturations=(
                int(membrane_saturations[0].item()),
                int(membrane_saturations[1].item()),
                int(membrane_saturations[2].item()),
                int(membrane_saturations[3].item()),
            ),
            approximate_additions=(
                approximate_additions[0],
                approximate_additions[1],
                approximate_additions[2],
                approximate_additions[3],
                approximate_additions[4],
                approximate_additions[5],
            ),
        )


def _quantize_power_of_two(
    weight: np.ndarray,
    fraction_bits: np.ndarray,
    minimum_exponent: int,
    maximum_exponent: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if minimum_exponent >= maximum_exponent:
        raise ValueError("PoT 最小指数必须小于最大指数")
    real = weight.astype(np.float64) * np.exp2(
        -fraction_bits.astype(np.float64)
    )[:, None]
    magnitude = np.abs(real)
    exponent = np.full(real.shape, POT_ZERO_EXPONENT, dtype=np.int8)
    sign = np.zeros(real.shape, dtype=np.int8)
    nonzero = magnitude > 0.0
    rounded = np.zeros(real.shape, dtype=np.int16)
    rounded[nonzero] = np.floor(np.log2(magnitude[nonzero]) + 0.5).astype(
        np.int16
    )
    retained = nonzero & (rounded >= minimum_exponent)
    clipped = np.clip(rounded, minimum_exponent, maximum_exponent)
    exponent[retained] = clipped[retained].astype(np.int8)
    sign[retained] = np.where(real[retained] < 0.0, -1, 1).astype(np.int8)
    integer = np.zeros(real.shape, dtype=np.int32)
    integer[retained] = (
        sign[retained].astype(np.int32)
        * np.left_shift(
            np.ones(int(np.count_nonzero(retained)), dtype=np.int32),
            exponent[retained].astype(np.int32) - minimum_exponent,
        )
    )
    return exponent, sign, integer


class PoTProposalSNN(FixedProposalSNN):
    """把指定 S1 层约束为零或带符号二次幂权重。"""

    LAYER_MODES = ("int8", "pot")

    def __init__(
        self,
        source: FixedProposalSNN,
        layer_modes: Tuple[str, str, str],
        minimum_exponent: int,
        maximum_exponent: int,
    ) -> None:
        if len(layer_modes) != 3 or any(
            mode not in self.LAYER_MODES for mode in layer_modes
        ):
            raise ValueError("每一层必须明确选择 int8 或 pot")
        if minimum_exponent < -14 or maximum_exponent > 6:
            raise ValueError("PoT 指数超出当前整数模型支持范围")
        super().__init__(
            weights=(source.weights[0], source.weights[1], source.weights[2]),
            weight_fraction_bits=(
                source.weight_fraction_bits[0],
                source.weight_fraction_bits[1],
                source.weight_fraction_bits[2],
            ),
            threshold=source.threshold,
            time_steps=source.time_steps,
            readout=source.readout,
            configuration=source.configuration,
        )
        self.layer_modes = layer_modes
        self.minimum_exponent = int(minimum_exponent)
        self.maximum_exponent = int(maximum_exponent)
        exponents = []
        signs = []
        integers = []
        for weight, fraction_bits in zip(
            self.weights, self.weight_fraction_bits
        ):
            exponent, sign, integer = _quantize_power_of_two(
                weight,
                fraction_bits,
                self.minimum_exponent,
                self.maximum_exponent,
            )
            exponents.append(exponent)
            signs.append(sign)
            integers.append(integer)
        self.pot_exponents = (exponents[0], exponents[1], exponents[2])
        self.pot_signs = (signs[0], signs[1], signs[2])
        self.pot_integers = (integers[0], integers[1], integers[2])

    def _current(
        self, value: np.ndarray, layer: int, input_fraction_bits: int
    ) -> Tuple[np.ndarray, int]:
        if self.layer_modes[layer] == "int8":
            return super()._current(value, layer, input_fraction_bits)
        raw = np.matmul(
            value.astype(np.int64),
            self.pot_integers[layer].T.astype(np.int64),
        )
        target_shift = (
            self.configuration.membrane_fraction_bits
            - input_fraction_bits
            + self.minimum_exponent
        )
        current = signed_round_shift(raw, -target_shift)
        return saturate_signed(current, self.configuration.accumulator_bits)

    def weight_statistics(self) -> List[Dict[str, Any]]:
        result = []
        for index, mode in enumerate(self.layer_modes):
            if mode == "int8":
                weight = self.weights[index]
                result.append(
                    {
                        "mode": mode,
                        "weights": int(weight.size),
                        "zero_weights": int(np.count_nonzero(weight == 0)),
                        "nonzero_weights": int(np.count_nonzero(weight)),
                    }
                )
                continue
            exponent = self.pot_exponents[index]
            retained = exponent != POT_ZERO_EXPONENT
            values, counts = np.unique(exponent[retained], return_counts=True)
            result.append(
                {
                    "mode": mode,
                    "weights": int(exponent.size),
                    "zero_weights": int(np.count_nonzero(~retained)),
                    "nonzero_weights": int(np.count_nonzero(retained)),
                    "zero_fraction": float(np.mean(~retained)),
                    "minimum_retained_exponent": int(values.min())
                    if len(values)
                    else None,
                    "maximum_retained_exponent": int(values.max())
                    if len(values)
                    else None,
                    "exponent_histogram": {
                        str(int(value)): int(count)
                        for value, count in zip(values, counts)
                    },
                }
            )
        return result


def export_fixed_proposal_snn(
    path: Path, model: FixedProposalSNN, metadata: Dict[str, Any]
) -> None:
    configuration = {
        "accumulator_bits": model.configuration.accumulator_bits,
        "membrane_bits": model.configuration.membrane_bits,
        "membrane_fraction_bits": model.configuration.membrane_fraction_bits,
        "input_fraction_bits": model.configuration.input_fraction_bits,
        "beta_numerator": model.configuration.beta_numerator,
        "beta_shift": model.configuration.beta_shift,
        "threshold": model.threshold,
        "time_steps": model.time_steps,
        "readout": model.readout,
        "rounding": "nearest_ties_away_from_zero",
        "overflow": "saturate_after_dot_and_each_membrane_update",
        "reset": "soft_reset",
    }
    arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(FIXED_S0P_EXPORT_VERSION, dtype=np.int16),
        "configuration_json": np.asarray(
            json.dumps(configuration, ensure_ascii=False, sort_keys=True)
        ),
        "metadata_json": np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        ),
    }
    for index, (weight, fraction_bits) in enumerate(
        zip(model.weights, model.weight_fraction_bits)
    ):
        arrays[f"weight_{index}"] = weight
        arrays[f"weight_fraction_bits_{index}"] = fraction_bits
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_fixed_proposal_snn(
    path: Path,
) -> Tuple[FixedProposalSNN, Dict[str, Any], Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["format_version"].item())
        if version != FIXED_S0P_EXPORT_VERSION:
            raise ValueError(f"不支持的 S1 导出版本：{version}")
        configuration = json.loads(str(archive["configuration_json"].item()))
        metadata = json.loads(str(archive["metadata_json"].item()))
        weights = tuple(archive[f"weight_{index}"].copy() for index in range(3))
        fractions = tuple(
            archive[f"weight_fraction_bits_{index}"].copy() for index in range(3)
        )
    expected_shapes = ((48, 128), (24, 48), (16, 24))
    for index, (weight, fraction_bits, shape) in enumerate(
        zip(weights, fractions, expected_shapes)
    ):
        if weight.dtype != np.int8 or weight.shape != shape:
            raise ValueError(f"S1 第 {index} 层权重类型或形状错误")
        if fraction_bits.dtype != np.int8 or fraction_bits.shape != (shape[0],):
            raise ValueError(f"S1 第 {index} 层缩放形状错误")
    fixed_configuration = FixedSNNConfig(
        accumulator_bits=int(configuration["accumulator_bits"]),
        membrane_bits=int(configuration["membrane_bits"]),
        membrane_fraction_bits=int(configuration["membrane_fraction_bits"]),
        input_fraction_bits=int(configuration["input_fraction_bits"]),
        beta_numerator=int(configuration["beta_numerator"]),
        beta_shift=int(configuration["beta_shift"]),
    )
    model = FixedProposalSNN(
        weights=(weights[0], weights[1], weights[2]),
        weight_fraction_bits=(fractions[0], fractions[1], fractions[2]),
        threshold=float(configuration["threshold"]),
        time_steps=int(configuration["time_steps"]),
        readout=str(configuration["readout"]),
        configuration=fixed_configuration,
    )
    if not isinstance(metadata, dict):
        raise ValueError("S1 元数据必须是 JSON 对象")
    return model, configuration, metadata


def export_approximate_proposal_snn(
    path: Path, model: ApproximateProposalSNN, metadata: Dict[str, Any]
) -> None:
    configuration = {
        "accumulator_bits": model.configuration.accumulator_bits,
        "membrane_bits": model.configuration.membrane_bits,
        "membrane_fraction_bits": model.configuration.membrane_fraction_bits,
        "input_fraction_bits": model.configuration.input_fraction_bits,
        "beta_numerator": model.configuration.beta_numerator,
        "beta_shift": model.configuration.beta_shift,
        "threshold": model.threshold,
        "time_steps": model.time_steps,
        "readout": model.readout,
        "low_bits": model.placement.low_bits,
        "synaptic_layers": list(model.placement.synaptic_layers),
        "membrane_updates": list(model.placement.membrane_updates),
        "addition": "signed_twos_complement_loa",
        "tree": "balanced_input_index_order",
        "overflow": "saturate_each_tree_node_and_membrane_update",
        "reset": "soft_reset",
    }
    arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(
            APPROXIMATE_S0P_EXPORT_VERSION, dtype=np.int16
        ),
        "configuration_json": np.asarray(
            json.dumps(configuration, ensure_ascii=False, sort_keys=True)
        ),
        "metadata_json": np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        ),
    }
    for index, (weight, fraction_bits) in enumerate(
        zip(model.weights, model.weight_fraction_bits)
    ):
        arrays[f"source_weight_{index}"] = weight
        arrays[f"source_weight_fraction_bits_{index}"] = fraction_bits
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_approximate_proposal_snn(
    path: Path,
) -> Tuple[ApproximateProposalSNN, Dict[str, Any], Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["format_version"].item())
        if version != APPROXIMATE_S0P_EXPORT_VERSION:
            raise ValueError(f"不支持的 S3 导出版本：{version}")
        configuration = json.loads(str(archive["configuration_json"].item()))
        metadata = json.loads(str(archive["metadata_json"].item()))
        weights = tuple(
            archive[f"source_weight_{index}"].copy() for index in range(3)
        )
        fractions = tuple(
            archive[f"source_weight_fraction_bits_{index}"].copy()
            for index in range(3)
        )
    if not isinstance(configuration, dict) or not isinstance(metadata, dict):
        raise ValueError("S3 配置和元数据必须是 JSON 对象")
    expected_shapes = ((48, 128), (24, 48), (16, 24))
    for index, (weight, fraction_bits, shape) in enumerate(
        zip(weights, fractions, expected_shapes)
    ):
        if weight.dtype != np.int8 or weight.shape != shape:
            raise ValueError(f"S3 第 {index} 层源权重类型或形状错误")
        if fraction_bits.dtype != np.int8 or fraction_bits.shape != (shape[0],):
            raise ValueError(f"S3 第 {index} 层缩放形状错误")
    fixed_configuration = FixedSNNConfig(
        accumulator_bits=int(configuration["accumulator_bits"]),
        membrane_bits=int(configuration["membrane_bits"]),
        membrane_fraction_bits=int(configuration["membrane_fraction_bits"]),
        input_fraction_bits=int(configuration["input_fraction_bits"]),
        beta_numerator=int(configuration["beta_numerator"]),
        beta_shift=int(configuration["beta_shift"]),
    )
    source = FixedProposalSNN(
        weights=(weights[0], weights[1], weights[2]),
        weight_fraction_bits=(fractions[0], fractions[1], fractions[2]),
        threshold=float(configuration["threshold"]),
        time_steps=int(configuration["time_steps"]),
        readout=str(configuration["readout"]),
        configuration=fixed_configuration,
    )
    synaptic = tuple(bool(value) for value in configuration["synaptic_layers"])
    membrane = tuple(bool(value) for value in configuration["membrane_updates"])
    if len(synaptic) != 3 or len(membrane) != 3:
        raise ValueError("S3 近似位置数量错误")
    model = ApproximateProposalSNN(
        source,
        LOAPlacement(
            low_bits=int(configuration["low_bits"]),
            synaptic_layers=(synaptic[0], synaptic[1], synaptic[2]),
            membrane_updates=(membrane[0], membrane[1], membrane[2]),
        ),
    )
    return model, configuration, metadata


def export_pot_proposal_snn(
    path: Path, model: PoTProposalSNN, metadata: Dict[str, Any]
) -> None:
    configuration = {
        "accumulator_bits": model.configuration.accumulator_bits,
        "membrane_bits": model.configuration.membrane_bits,
        "membrane_fraction_bits": model.configuration.membrane_fraction_bits,
        "input_fraction_bits": model.configuration.input_fraction_bits,
        "beta_numerator": model.configuration.beta_numerator,
        "beta_shift": model.configuration.beta_shift,
        "threshold": model.threshold,
        "time_steps": model.time_steps,
        "readout": model.readout,
        "layer_modes": list(model.layer_modes),
        "minimum_exponent": model.minimum_exponent,
        "maximum_exponent": model.maximum_exponent,
        "zero_exponent_code": POT_ZERO_EXPONENT,
        "rounding": "nearest_log2_ties_to_larger_magnitude",
        "overflow": "saturate_after_aligned_pot_sum_and_each_membrane_update",
        "reset": "soft_reset",
    }
    arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(POT_S0P_EXPORT_VERSION, dtype=np.int16),
        "configuration_json": np.asarray(
            json.dumps(configuration, ensure_ascii=False, sort_keys=True)
        ),
        "metadata_json": np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        ),
    }
    for index in range(3):
        arrays[f"source_weight_{index}"] = model.weights[index]
        arrays[f"source_weight_fraction_bits_{index}"] = (
            model.weight_fraction_bits[index]
        )
        arrays[f"pot_exponent_{index}"] = model.pot_exponents[index]
        arrays[f"pot_sign_{index}"] = model.pot_signs[index]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_pot_proposal_snn(
    path: Path,
) -> Tuple[PoTProposalSNN, Dict[str, Any], Dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["format_version"].item())
        if version != POT_S0P_EXPORT_VERSION:
            raise ValueError(f"不支持的 S2 导出版本：{version}")
        configuration = json.loads(str(archive["configuration_json"].item()))
        metadata = json.loads(str(archive["metadata_json"].item()))
        weights = tuple(
            archive[f"source_weight_{index}"].copy() for index in range(3)
        )
        fractions = tuple(
            archive[f"source_weight_fraction_bits_{index}"].copy()
            for index in range(3)
        )
        stored_exponents = tuple(
            archive[f"pot_exponent_{index}"].copy() for index in range(3)
        )
        stored_signs = tuple(
            archive[f"pot_sign_{index}"].copy() for index in range(3)
        )
    if not isinstance(configuration, dict) or not isinstance(metadata, dict):
        raise ValueError("S2 配置和元数据必须是 JSON 对象")
    fixed_configuration = FixedSNNConfig(
        accumulator_bits=int(configuration["accumulator_bits"]),
        membrane_bits=int(configuration["membrane_bits"]),
        membrane_fraction_bits=int(configuration["membrane_fraction_bits"]),
        input_fraction_bits=int(configuration["input_fraction_bits"]),
        beta_numerator=int(configuration["beta_numerator"]),
        beta_shift=int(configuration["beta_shift"]),
    )
    source = FixedProposalSNN(
        weights=(weights[0], weights[1], weights[2]),
        weight_fraction_bits=(fractions[0], fractions[1], fractions[2]),
        threshold=float(configuration["threshold"]),
        time_steps=int(configuration["time_steps"]),
        readout=str(configuration["readout"]),
        configuration=fixed_configuration,
    )
    modes = tuple(str(value) for value in configuration["layer_modes"])
    if len(modes) != 3:
        raise ValueError("S2 层模式数量错误")
    model = PoTProposalSNN(
        source,
        layer_modes=(modes[0], modes[1], modes[2]),
        minimum_exponent=int(configuration["minimum_exponent"]),
        maximum_exponent=int(configuration["maximum_exponent"]),
    )
    for index in range(3):
        if not np.array_equal(model.pot_exponents[index], stored_exponents[index]):
            raise ValueError(f"S2 第 {index} 层指数与量化规则不一致")
        if not np.array_equal(model.pot_signs[index], stored_signs[index]):
            raise ValueError(f"S2 第 {index} 层符号与量化规则不一致")
    return model, configuration, metadata
