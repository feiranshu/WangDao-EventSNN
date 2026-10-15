"""Detached layer diagnostics and explicit failure limits; never change weights."""
import math

import numpy as np
import torch

from .refiner import LAYERS, STAT_NAMES

DEFAULT_LIMITS = {"max_unit_burst": 256.0, "max_abs_voltage_in_thresholds": 4096.0,
                  "max_abs_logit": 80.0, "max_gradient_norm": 10000.0,
                  "max_loss": 100.0}


def parameter_norms(model):
    return {n: float(p.detach().norm()) for n, p in model.named_parameters()}


def gradient_norms(model):
    norms = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("Missing/nonfinite gradient: " + name)
        norms[name] = float(parameter.grad.detach().norm())
    return norms


class HealthMonitor:
    def __init__(self, limits=None):
        self.limits = dict(DEFAULT_LIMITS if limits is None else limits)
        if set(self.limits) != set(DEFAULT_LIMITS) or any(
                not math.isfinite(v) or v <= 0 for v in self.limits.values()):
            raise ValueError("Invalid numerical failure limits")
        self.steps = 0
        self.sums = np.zeros((3, len(STAT_NAMES)), np.float64)
        self.maxima = np.zeros_like(self.sums)
        self.logits = 0
        self.logit_absmax = self.logit_square_sum = 0.0
        self.saturated_logits = 0
        self.gradient_steps = self.clipped_steps = 0
        self.gradient_sum = self.gradient_max = 0.0
        self.layer_gradient_sum = {}
        self.layer_gradient_max = {}

    def _limit(self, name, value):
        if not math.isfinite(value) or value > self.limits[name]:
            raise FloatingPointError(f"V2 stability guard: {name}={value:.6g} exceeds {self.limits[name]:.6g}")

    @torch.no_grad()
    def observe(self, model, logits, candidate, active):
        if model.last_diagnostics is None:
            raise AssertionError("Missing layer diagnostics")
        stats = model.last_diagnostics[:, active].detach().cpu().numpy().reshape(-1, 3, len(STAT_NAMES))
        if not len(stats):
            return
        if not np.isfinite(stats).all():
            raise FloatingPointError("Nonfinite layer diagnostics")
        maxima = stats.max(0)
        self._limit("max_unit_burst", float(maxima[:, 4].max()))
        self._limit("max_abs_voltage_in_thresholds", float(maxima[:, 2].max()) / model.config.threshold)
        self.steps += len(stats)
        self.sums += stats.sum(0, dtype=np.float64)
        self.maxima = np.maximum(self.maxima, maxima)
        mask = torch.as_tensor(candidate, dtype=torch.bool, device=logits.device)
        values = logits.detach()[mask]
        if values.numel():
            maximum = float(values.abs().max())
            self._limit("max_abs_logit", maximum)
            self.logit_absmax = max(self.logit_absmax, maximum)
            self.logit_square_sum += float(values.double().square().sum())
            self.logits += values.numel()
            p = values.sigmoid()
            self.saturated_logits += int(((p == 0) | (p == 1)).sum())

    def observe_loss(self, loss):
        self._limit("max_loss", float(loss.detach()))

    def observe_gradients(self, norm, layers, clip):
        norm = float(norm)
        self._limit("max_gradient_norm", norm)
        self.gradient_steps += 1
        self.clipped_steps += int(norm > clip)
        self.gradient_sum += norm
        self.gradient_max = max(self.gradient_max, norm)
        for name, value in layers.items():
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite gradient norm: " + name)
            self.layer_gradient_sum[name] = self.layer_gradient_sum.get(name, 0.0) + value
            self.layer_gradient_max[name] = max(self.layer_gradient_max.get(name, 0.0), value)

    def summary(self):
        fraction = self.saturated_logits / max(self.logits, 1)
        clipped = self.clipped_steps / max(self.gradient_steps, 1)
        warnings = []
        if fraction > .001:
            warnings.append("fp32_probability_saturation_above_0.1_percent")
        if clipped > .5:
            warnings.append("more_than_half_of_gradients_exceed_clip_limit")
        return {"active_stream_steps": self.steps,
                "layer_statistics_scope": "Per-stream full-grid statistics; means weighted by active steps, excluding padded slots",
                "layers": {name: {"mean_over_active_steps": dict(zip(STAT_NAMES, (self.sums[i] / max(self.steps, 1)).tolist())),
                                  "max_over_active_steps": dict(zip(STAT_NAMES, self.maxima[i].tolist()))}
                           for i, name in enumerate(LAYERS)},
                "candidate_logits": {"count": self.logits, "absmax": self.logit_absmax,
                    "rms": math.sqrt(self.logit_square_sum / max(self.logits, 1)),
                    "fp32_probability_saturation_fraction": fraction},
                "pre_clip_gradients": {"backward_passes": self.gradient_steps,
                    "global_norm_mean": self.gradient_sum / max(self.gradient_steps, 1),
                    "global_norm_max": self.gradient_max, "fraction_above_clip_limit": clipped,
                    "parameter_norm_mean": {k: v / max(self.gradient_steps, 1) for k, v in self.layer_gradient_sum.items()},
                    "parameter_norm_max": dict(self.layer_gradient_max)},
                "warnings": warnings, "failure_limits": dict(self.limits)}
