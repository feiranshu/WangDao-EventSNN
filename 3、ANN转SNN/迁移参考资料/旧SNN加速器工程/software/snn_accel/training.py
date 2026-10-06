from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("snn_accel.training requires PyTorch") from exc

from .quantization import power_of_two_ste


class _ApproximateAddSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b, width: int, approx_lsb_bits: int):
        minimum = -(1 << (width - 1))
        maximum = (1 << (width - 1)) - 1
        left = torch.round(a).to(torch.int64)
        right = torch.round(b).to(torch.int64)
        mask = (1 << approx_lsb_bits) - 1
        low = torch.bitwise_or(torch.bitwise_and(left, mask), torch.bitwise_and(right, mask))
        high = ((left >> approx_lsb_bits) + (right >> approx_lsb_bits)) << approx_lsb_bits
        return torch.clamp(torch.bitwise_or(high, low), minimum, maximum).to(a.dtype)

    @staticmethod
    def backward(ctx, gradient):
        return gradient, gradient, None, None


def approximate_add_ste(a, b, width: int, approx_lsb_bits: int):
    return _ApproximateAddSTE.apply(a, b, width, approx_lsb_bits)


class HardwareAwareLIFCell(nn.Module):
    """Training-time LIF cell matching the frozen RTL arithmetic contract."""

    def __init__(
        self,
        *,
        threshold: int,
        width: int = 20,
        leak_shift: int = 3,
        approx_lsb_bits: int = 4,
        surrogate_slope: float = 5.0,
    ):
        super().__init__()
        self.threshold = threshold
        self.width = width
        self.leak_shift = leak_shift
        self.approx_lsb_bits = approx_lsb_bits
        self.surrogate_slope = surrogate_slope

    def forward(self, membrane, synaptic_input, *, approximate: bool):
        leak = torch.zeros_like(membrane) if self.leak_shift == 0 else torch.floor(membrane / (2**self.leak_shift))
        decayed = membrane - leak
        if approximate:
            updated = approximate_add_ste(decayed, synaptic_input, self.width, self.approx_lsb_bits)
        else:
            lower = -(1 << (self.width - 1))
            upper = (1 << (self.width - 1)) - 1
            updated = torch.clamp(torch.round(decayed + synaptic_input), lower, upper)
        hard_spike = (updated >= self.threshold).to(updated.dtype)
        surrogate = torch.sigmoid(self.surrogate_slope * (updated - self.threshold))
        spike = hard_spike + surrogate - surrogate.detach()
        next_membrane = updated * (1.0 - hard_spike)
        return next_membrane, spike


def quantize_module_weights(module: nn.Module, exponent_min: int = -8, exponent_max: int = 7) -> dict[str, torch.Tensor]:
    return {
        name: power_of_two_ste(parameter, exponent_min, exponent_max)
        for name, parameter in module.named_parameters()
        if name.endswith("weight")
    }


@dataclass(frozen=True)
class DistillationTerms:
    total: torch.Tensor
    detection: torch.Tensor
    logits: torch.Tensor
    features: torch.Tensor


def distillation_loss(
    *,
    detection_loss: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_features: Sequence[torch.Tensor] = (),
    teacher_features: Sequence[torch.Tensor] = (),
    kd_weight: float = 0.5,
    temperature: float = 2.0,
    feature_weight: float = 1.0,
) -> DistillationTerms:
    if not 0.0 <= kd_weight <= 1.0:
        raise ValueError("kd_weight must be in [0, 1]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if len(student_features) != len(teacher_features):
        raise ValueError("student and teacher feature lists must have equal length")
    teacher_logits = teacher_logits.detach()
    logits_loss = functional.kl_div(
        functional.log_softmax(student_logits / temperature, dim=-1),
        functional.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature**2)
    if student_features:
        feature_loss = torch.stack(
            [functional.mse_loss(student, teacher.detach()) for student, teacher in zip(student_features, teacher_features)]
        ).mean()
    else:
        feature_loss = detection_loss.new_zeros(())
    total = (1.0 - kd_weight) * detection_loss + kd_weight * (logits_loss + feature_weight * feature_loss)
    return DistillationTerms(total, detection_loss, logits_loss, feature_loss)


def quantized_forward(model: nn.Module, inputs: torch.Tensor, hw_cfg: Mapping, **kwargs):
    """Unified inference hook used by EMS-YOLO adapters.

    An adapted model must expose ``hardware_forward`` so that this entry point
    cannot silently run a floating-point path while claiming bit-true output.
    """

    hardware_forward = getattr(model, "hardware_forward", None)
    if hardware_forward is None:
        raise TypeError("model must implement hardware_forward(inputs, hw_cfg, **kwargs)")
    return hardware_forward(inputs, hw_cfg, **kwargs)

