"""S0 阶段使用的可训练 LIF 单元和 Proposal-SNN。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, NamedTuple, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import torch.nn as nn

from .h2.constants import PROPOSAL_DIMENSIONS, REFINER_DIMENSIONS


S0P_EXPORT_VERSION = 1
S0R_EXPORT_VERSION = 1


class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any, value: torch.Tensor, slope: float
    ) -> torch.Tensor:
        ctx.save_for_backward(value)  # type: ignore[attr-defined]
        ctx.slope = float(slope)  # type: ignore[attr-defined]
        return (value >= 0.0).to(value.dtype)

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any, gradient: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        (value,) = ctx.saved_tensors  # type: ignore[attr-defined]
        slope = ctx.slope  # type: ignore[attr-defined]
        surrogate = 1.0 / torch.square(1.0 + slope * value.abs())
        return gradient * surrogate, None


def surrogate_spike(value: torch.Tensor, slope: float = 10.0) -> torch.Tensor:
    """Apply a binary threshold with a fast-sigmoid surrogate gradient."""
    if slope <= 0.0:
        raise ValueError("surrogate slope must be positive")
    return cast(torch.Tensor, _SurrogateSpike.apply(value, float(slope)))


class LIFCell(nn.Module):
    """Stateless-interface leaky integrate-and-fire cell.

    State is passed in and returned instead of being kept on the module, so
    separate samples and batches cannot accidentally share membrane values.
    """

    beta: torch.Tensor
    threshold: torch.Tensor

    def __init__(
        self,
        beta: float = 0.875,
        threshold: float = 1.0,
        surrogate_slope: float = 10.0,
    ) -> None:
        super().__init__()
        if not 0.0 <= beta <= 1.0:
            raise ValueError("beta must be in [0, 1]")
        if threshold <= 0.0:
            raise ValueError("threshold must be positive")
        if surrogate_slope <= 0.0:
            raise ValueError("surrogate slope must be positive")
        self.register_buffer("beta", torch.tensor(float(beta), dtype=torch.float32))
        self.register_buffer(
            "threshold", torch.tensor(float(threshold), dtype=torch.float32)
        )
        self.surrogate_slope = float(surrogate_slope)

    def forward(
        self, current: torch.Tensor, membrane: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if membrane is None:
            membrane = torch.zeros_like(current)
        if membrane.shape != current.shape:
            raise ValueError("membrane and current must have identical shapes")
        updated = self.beta.to(current.dtype) * membrane + current
        spikes = surrogate_spike(
            updated - self.threshold.to(current.dtype), self.surrogate_slope
        )
        # Soft reset preserves overshoot. Detaching the reset path avoids an
        # artificial gradient through the discrete reset event.
        reset = spikes.detach() * self.threshold.to(current.dtype)
        return spikes, updated - reset


class ProposalSNNOutput(NamedTuple):
    logits: torch.Tensor
    spike_rates: torch.Tensor


class ProposalSNNActivity(NamedTuple):
    """Proposal-SNN 前向结果及逐层精确活动计数。"""

    output: ProposalSNNOutput
    input_nonzero: int
    spike_counts: Tuple[int, int, int]


class RefinerSNNOutput(NamedTuple):
    raw_logits: torch.Tensor
    spike_rates: torch.Tensor


class ProposalSNN(nn.Module):
    """Temporal SNN counterpart of H2's 128→48→24→16 Proposal MLP."""

    READOUTS = ("last_membrane", "mean_membrane", "spike_rate")
    output_beta: torch.Tensor

    def __init__(
        self,
        dimensions: Sequence[int] = PROPOSAL_DIMENSIONS,
        beta: float = 0.875,
        threshold: float = 0.25,
        surrogate_slope: float = 10.0,
        readout: str = "mean_membrane",
    ) -> None:
        super().__init__()
        self.dimensions = tuple(int(value) for value in dimensions)
        if self.dimensions != tuple(PROPOSAL_DIMENSIONS):
            raise ValueError("S0-P must preserve the H2 Proposal dimensions")
        if readout not in self.READOUTS:
            raise ValueError("unsupported Proposal-SNN readout")
        self.readout = readout
        self.layers = nn.ModuleList(
            nn.Linear(self.dimensions[index], self.dimensions[index + 1], bias=False)
            for index in range(len(self.dimensions) - 1)
        )
        self.hidden_cells = nn.ModuleList(
            [
                LIFCell(beta, threshold, surrogate_slope),
                LIFCell(beta, threshold, surrogate_slope),
            ]
        )
        self.output_cell = LIFCell(beta, threshold, surrogate_slope)
        self.register_buffer("output_beta", torch.tensor(float(beta), dtype=torch.float32))

    def initialize_from_h2(self, h2_proposal: nn.Module) -> None:
        """Copy compatible H2 linear weights without linking either model."""
        source_layers = getattr(h2_proposal, "layers", None)
        if source_layers is None or len(source_layers) != len(self.layers):
            raise ValueError("H2 Proposal layers do not match S0-P")
        with torch.no_grad():
            for destination, source in zip(self.layers, source_layers):
                if not isinstance(destination, nn.Linear):
                    raise TypeError("S0-P must contain linear layers")
                if not isinstance(source, nn.Linear):
                    raise TypeError("H2 Proposal must contain linear layers")
                if destination.weight.shape != source.weight.shape:
                    raise ValueError("H2 Proposal weight shape mismatch")
                destination.weight.copy_(source.weight)

    def forward(self, features: torch.Tensor) -> ProposalSNNOutput:
        if features.ndim != 3 or features.shape[2] != self.dimensions[0]:
            raise ValueError("features must have shape [time, batch, 128]")
        if features.shape[0] <= 0:
            raise ValueError("features must contain at least one time step")
        value = features.to(torch.float32)
        if not features.is_floating_point():
            value = value / 255.0

        hidden_membranes: list[Optional[torch.Tensor]] = [None, None]
        hidden_spike_sums = [value.new_zeros(()), value.new_zeros(())]
        output_membrane: Optional[torch.Tensor] = None
        output_spike_membrane: Optional[torch.Tensor] = None
        output_spike_sum = value.new_zeros(())
        output_history = []
        output_spike_history = []

        for step in range(value.shape[0]):
            current = value[step]
            for index in range(2):
                current = self.layers[index](current)
                current, hidden_membranes[index] = self.hidden_cells[index](
                    current, hidden_membranes[index]
                )
                hidden_spike_sums[index] = hidden_spike_sums[index] + current.sum()
            output_current = self.layers[2](current)
            if output_membrane is None:
                output_membrane = torch.zeros_like(output_current)
            output_membrane = (
                self.output_beta.to(output_current.dtype) * output_membrane
                + output_current
            )
            output_history.append(output_membrane)
            output_spikes, output_spike_membrane = self.output_cell(
                output_current, output_spike_membrane
            )
            output_spike_sum = output_spike_sum + output_spikes.sum()
            output_spike_history.append(output_spikes)

        if output_membrane is None:
            raise AssertionError("output membrane was not initialized")
        stacked_output = torch.stack(output_history, dim=0)
        if self.readout == "last_membrane":
            logits = output_membrane
        elif self.readout == "mean_membrane":
            logits = stacked_output.mean(dim=0)
        else:
            rate = torch.stack(output_spike_history, dim=0).mean(dim=0)
            logits = 8.0 * (rate - 0.5)

        denominators = (
            value.shape[0] * value.shape[1] * self.dimensions[1],
            value.shape[0] * value.shape[1] * self.dimensions[2],
            value.shape[0] * value.shape[1] * self.dimensions[3],
        )
        spike_rates = torch.stack(
            [
                hidden_spike_sums[0] / denominators[0],
                hidden_spike_sums[1] / denominators[1],
                output_spike_sum / denominators[2],
            ]
        )
        return ProposalSNNOutput(logits=logits, spike_rates=spike_rates)


def profile_proposal_snn(
    model: ProposalSNN, features: torch.Tensor
) -> ProposalSNNActivity:
    """执行一次不改变模型数值的前向，并统计输入非零值和各层脉冲数。"""
    captured: list[list[torch.Tensor]] = [[], [], []]

    def capture(index: int) -> Any:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            spikes = output[0]
            if not isinstance(spikes, torch.Tensor):
                raise TypeError("LIF 输出必须包含脉冲张量")
            captured[index].append(spikes.detach())

        return hook

    cells = [*model.hidden_cells, model.output_cell]
    handles = [cell.register_forward_hook(capture(index)) for index, cell in enumerate(cells)]
    try:
        output = model(features)
    finally:
        for handle in handles:
            handle.remove()
    spike_counts = tuple(
        int(sum(float(value.sum().item()) for value in values))
        for values in captured
    )
    if len(spike_counts) != 3:
        raise AssertionError("Proposal-SNN 必须包含三个脉冲统计层")
    return ProposalSNNActivity(
        output=output,
        input_nonzero=int(torch.count_nonzero(features).item()),
        spike_counts=(spike_counts[0], spike_counts[1], spike_counts[2]),
    )


class RefinerSNN(nn.Module):
    """以固定微时间步独立处理每个候选事件的 32→64→32→4 SNN。"""

    READOUTS = ("last_membrane", "mean_membrane")

    output_beta: torch.Tensor

    def __init__(
        self,
        dimensions: Sequence[int] = REFINER_DIMENSIONS,
        beta: float = 0.875,
        threshold: float = 0.25,
        surrogate_slope: float = 10.0,
        readout: str = "mean_membrane",
    ) -> None:
        super().__init__()
        self.dimensions = tuple(int(value) for value in dimensions)
        if self.dimensions != tuple(REFINER_DIMENSIONS):
            raise ValueError("S0-R must preserve the H2 Refiner dimensions")
        if readout not in self.READOUTS:
            raise ValueError("unsupported Refiner-SNN readout")
        self.readout = readout
        self.layers = nn.ModuleList(
            nn.Linear(self.dimensions[index], self.dimensions[index + 1], bias=False)
            for index in range(len(self.dimensions) - 1)
        )
        self.hidden_cells = nn.ModuleList(
            [
                LIFCell(beta, threshold, surrogate_slope),
                LIFCell(beta, threshold, surrogate_slope),
            ]
        )
        self.output_cell = LIFCell(beta, threshold, surrogate_slope)
        self.register_buffer("output_beta", torch.tensor(float(beta), dtype=torch.float32))

    def initialize_from_h2(self, h2_refiner: nn.Module) -> None:
        source_layers = getattr(h2_refiner, "layers", None)
        if source_layers is None or len(source_layers) != len(self.layers):
            raise ValueError("H2 Refiner layers do not match S0-R")
        with torch.no_grad():
            for destination, source in zip(self.layers, source_layers):
                if not isinstance(destination, nn.Linear) or not isinstance(source, nn.Linear):
                    raise TypeError("S0-R and H2 Refiner must contain linear layers")
                if destination.weight.shape != source.weight.shape:
                    raise ValueError("H2 Refiner weight shape mismatch")
                destination.weight.copy_(source.weight)

    def forward(
        self,
        features: torch.Tensor,
    ) -> RefinerSNNOutput:
        if features.ndim != 3 or features.shape[2] != self.dimensions[0]:
            raise ValueError("features must have shape [time, event, 32]")
        if features.shape[0] <= 0:
            raise ValueError("features must contain at least one time step")
        value = features.to(torch.float32)
        if not features.is_floating_point():
            value = value / 255.0
        hidden_membranes: list[Optional[torch.Tensor]] = [None, None]
        readout_membrane: Optional[torch.Tensor] = None
        output_spike_membrane: Optional[torch.Tensor] = None
        hidden_spike_sums = [value.new_zeros(()), value.new_zeros(())]
        output_spike_sum = value.new_zeros(())
        raw_history = []
        for step in range(value.shape[0]):
            current = value[step]
            for index in range(2):
                current = self.layers[index](current)
                current, hidden_membranes[index] = self.hidden_cells[index](
                    current, hidden_membranes[index]
                )
                hidden_spike_sums[index] = hidden_spike_sums[index] + current.sum()
            output_current = self.layers[2](current)
            if readout_membrane is None:
                readout_membrane = torch.zeros_like(output_current)
            readout_membrane = (
                self.output_beta.to(output_current.dtype) * readout_membrane
                + output_current
            )
            output_spikes, output_spike_membrane = self.output_cell(
                output_current, output_spike_membrane
            )
            output_spike_sum = output_spike_sum + output_spikes.sum()
            raw_history.append(readout_membrane)
        if readout_membrane is None:
            raise AssertionError("S0-R readout membrane was not initialized")
        stacked = torch.stack(raw_history, dim=0)
        raw_logits = (
            readout_membrane
            if self.readout == "last_membrane"
            else stacked.mean(dim=0)
        )
        denominators = (
            value.shape[0] * value.shape[1] * self.dimensions[1],
            value.shape[0] * value.shape[1] * self.dimensions[2],
            value.shape[0] * value.shape[1] * self.dimensions[3],
        )
        spike_rates = torch.stack(
            [
                hidden_spike_sums[0] / denominators[0],
                hidden_spike_sums[1] / denominators[1],
                output_spike_sum / denominators[2],
            ]
        )
        return RefinerSNNOutput(
            raw_logits=raw_logits,
            spike_rates=spike_rates,
        )


def export_proposal_snn(
    path: Path,
    model: ProposalSNN,
    configuration: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """将冻结 S0-P 权重导出为不依赖 pickle 的 NPZ。"""
    state = model.state_dict()
    keys = list(state)
    arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(S0P_EXPORT_VERSION, dtype=np.int16),
        "configuration_json": np.asarray(
            json.dumps(configuration, ensure_ascii=False, sort_keys=True)
        ),
        "metadata_json": np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        ),
        "state_keys": np.asarray(keys),
    }
    for index, key in enumerate(keys):
        arrays[f"state_{index}"] = state[key].detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_proposal_snn_export(
    path: Path,
) -> Tuple[ProposalSNN, Dict[str, Any], Dict[str, Any]]:
    """读取由 :func:`export_proposal_snn` 生成的冻结模型。"""
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["format_version"].item())
        if version != S0P_EXPORT_VERSION:
            raise ValueError(f"unsupported S0-P export version: {version}")
        configuration = json.loads(str(archive["configuration_json"].item()))
        metadata = json.loads(str(archive["metadata_json"].item()))
        if not isinstance(configuration, dict) or not isinstance(metadata, dict):
            raise ValueError("S0-P export metadata must contain JSON objects")
        keys = [str(value) for value in archive["state_keys"].tolist()]
        state = {
            key: torch.from_numpy(archive[f"state_{index}"].copy())
            for index, key in enumerate(keys)
        }
    model = ProposalSNN(
        beta=float(configuration["beta"]),
        threshold=float(configuration["threshold"]),
        surrogate_slope=float(configuration["surrogate_slope"]),
        readout=str(configuration["readout"]),
    )
    model.load_state_dict(state)
    return model, configuration, metadata


def export_refiner_snn(
    path: Path,
    model: RefinerSNN,
    configuration: Dict[str, Any],
    metadata: Dict[str, Any],
) -> None:
    """将冻结 S0-R 权重导出为不依赖 pickle 的 NPZ。"""
    state = model.state_dict()
    keys = list(state)
    arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(S0R_EXPORT_VERSION, dtype=np.int16),
        "configuration_json": np.asarray(
            json.dumps(configuration, ensure_ascii=False, sort_keys=True)
        ),
        "metadata_json": np.asarray(
            json.dumps(metadata, ensure_ascii=False, sort_keys=True)
        ),
        "state_keys": np.asarray(keys),
    }
    for index, key in enumerate(keys):
        arrays[f"state_{index}"] = state[key].detach().cpu().numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_refiner_snn_export(
    path: Path,
) -> Tuple[RefinerSNN, Dict[str, Any], Dict[str, Any]]:
    """读取由 :func:`export_refiner_snn` 生成的冻结模型。"""
    with np.load(path, allow_pickle=False) as archive:
        version = int(archive["format_version"].item())
        if version != S0R_EXPORT_VERSION:
            raise ValueError(f"unsupported S0-R export version: {version}")
        configuration = json.loads(str(archive["configuration_json"].item()))
        metadata = json.loads(str(archive["metadata_json"].item()))
        if not isinstance(configuration, dict) or not isinstance(metadata, dict):
            raise ValueError("S0-R export metadata must contain JSON objects")
        keys = [str(value) for value in archive["state_keys"].tolist()]
        state = {
            key: torch.from_numpy(archive[f"state_{index}"].copy())
            for index, key in enumerate(keys)
        }
    model = RefinerSNN(
        beta=float(configuration["beta"]),
        threshold=float(configuration["threshold"]),
        surrogate_slope=float(configuration["surrogate_slope"]),
        readout=str(configuration["readout"]),
    )
    model.load_state_dict(state)
    return model, configuration, metadata
