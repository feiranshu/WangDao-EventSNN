"""Finite logit magnitude is advisory; nonfinite values and other budgets stop."""
import math

import numpy as np
import torch

from ..h2pro_snn_v2.monitor import gradient_norms, parameter_norms
from ..h2pro_snn_v2.refiner import LAYERS, STAT_NAMES

DEFAULT_LIMITS = {"max_unit_burst": 256.0, "max_abs_voltage_in_thresholds": 4096.0,
                  "max_gradient_norm": 10000.0, "max_loss": 100.0}
DEFAULT_WARNINGS = {"max_abs_logit": 80.0}


class HealthMonitor:
    def __init__(self, limits=None, warnings=None):
        self.limits = dict(DEFAULT_LIMITS if limits is None else limits)
        self.warning_limits = dict(DEFAULT_WARNINGS if warnings is None else warnings)
        for values, expected in ((self.limits, DEFAULT_LIMITS), (self.warning_limits, DEFAULT_WARNINGS)):
            if set(values) != set(expected) or any(not math.isfinite(v) or v <= 0 for v in values.values()):
                raise ValueError("Invalid numerical monitoring limits")
        self.steps = self.logits = self.saturated_logits = 0
        self.sums = np.zeros((3, len(STAT_NAMES)), np.float64)
        self.maxima = np.zeros_like(self.sums)
        self.logit_absmax = self.logit_square_sum = 0.0
        self.logit_min = self.logit_max = None
        self.large_logits = self.logit_warning_batches = 0
        self.gradient_steps = self.clipped_steps = 0
        self.gradient_sum = self.gradient_max = 0.0
        self.layer_gradient_sum, self.layer_gradient_max = {}, {}
        self.last_loss = None
        self.last_gradient_norm = None
        self.last_alerts = []

    def _limit(self, name, value):
        if not math.isfinite(value):
            raise FloatingPointError(f"V2.1 nonfinite value: {name}={value}")
        if value > self.limits[name]:
            raise FloatingPointError(f"V2.1 hard guard: {name}={value:.6g} exceeds {self.limits[name]:.6g}")

    @torch.no_grad()
    def observe(self, model, logits, candidate, active):
        self.last_alerts = []
        if model.last_diagnostics is None:
            raise AssertionError("Missing layer diagnostics")
        stats = model.last_diagnostics[:, active].detach().cpu().numpy().reshape(-1, 3, len(STAT_NAMES))
        if not np.isfinite(stats).all():
            raise FloatingPointError("Nonfinite layer diagnostics")
        if len(stats):
            maxima = stats.max(0)
            # Retain the offending layer statistics before enforcing the budgets.
            self.steps += len(stats)
            self.sums += stats.sum(0, dtype=np.float64)
            self.maxima = np.maximum(self.maxima, maxima)
            self._limit("max_unit_burst", float(maxima[:, 4].max()))
            self._limit("max_abs_voltage_in_thresholds", float(maxima[:, 2].max()) / model.config.threshold)
        mask = torch.as_tensor(candidate, dtype=torch.bool, device=logits.device)
        values = logits.detach()[mask]
        if not torch.isfinite(values).all():
            raise FloatingPointError("Nonfinite candidate logits")
        if values.numel():
            maximum = float(values.abs().max())
            low, high = float(values.min()), float(values.max())
            self.logit_min = low if self.logit_min is None else min(low, self.logit_min)
            self.logit_max = high if self.logit_max is None else max(high, self.logit_max)
            self.logit_absmax = max(self.logit_absmax, maximum)
            self.logit_square_sum += float(values.double().square().sum())
            self.logits += values.numel()
            p = values.sigmoid()
            self.saturated_logits += int(((p == 0) | (p == 1)).sum())
            large = int((values.abs() > self.warning_limits["max_abs_logit"]).sum())
            self.large_logits += large
            if large:
                self.logit_warning_batches += 1
                self.last_alerts.append("finite_candidate_logit_magnitude")
        return list(self.last_alerts)

    def observe_loss(self, loss):
        self.last_loss = float(loss.detach())
        self._limit("max_loss", self.last_loss)

    def observe_gradients(self, norm, layers, clip):
        norm = float(norm)
        self.last_gradient_norm = norm
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
        if self.logit_warning_batches:
            warnings.append("finite_candidate_logit_magnitude")
        if fraction > .001:
            warnings.append("fp32_probability_saturation_above_0.1_percent")
        if clipped > .5:
            warnings.append("more_than_half_of_gradients_exceed_clip_limit")
        return {"active_stream_steps": self.steps,
                "layer_statistics_scope": "Full-grid per active stream step; padded slots excluded",
                "layers": {name: {"mean_over_active_steps": dict(zip(STAT_NAMES, (self.sums[i] / max(self.steps, 1)).tolist())),
                                  "max_over_active_steps": dict(zip(STAT_NAMES, self.maxima[i].tolist()))}
                           for i, name in enumerate(LAYERS)},
                "candidate_logits": {"count": self.logits, "min": self.logit_min, "max": self.logit_max,
                    "absmax": self.logit_absmax, "rms": math.sqrt(self.logit_square_sum / max(self.logits, 1)),
                    "fp32_probability_saturation_fraction": fraction,
                    "above_warning_count": self.large_logits,
                    "above_warning_fraction": self.large_logits / max(self.logits, 1),
                    "warning_batches": self.logit_warning_batches},
                "pre_clip_gradients": {"backward_passes": self.gradient_steps,
                    "global_norm_mean": self.gradient_sum / max(self.gradient_steps, 1),
                    "global_norm_max": self.gradient_max, "fraction_above_clip_limit": clipped,
                    "parameter_norm_mean": {k: v / max(self.gradient_steps, 1) for k, v in self.layer_gradient_sum.items()},
                    "parameter_norm_max": dict(self.layer_gradient_max), "last_observed_norm": self.last_gradient_norm},
                "last_observed_loss": self.last_loss,
                "warnings": warnings, "failure_limits": dict(self.limits),
                "warning_limits": dict(self.warning_limits),
                "finite_logit_policy": "Advisory only; no clipping or automatic hyperparameter change"}
