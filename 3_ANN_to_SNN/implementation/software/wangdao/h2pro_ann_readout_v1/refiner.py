"""Only the head recurrence changes; stem/hidden retain ANN-state v1 semantics.

z_t = lambda*z_(t-1) + (1-lambda)*I_t, with no head firing, rectification or reset.
Bias is applied once after integration. The positive-part head statistic exists
only to preserve the old amplitude guard; it is not used by the network.
"""
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from ..h2pro_ann_state_v1.refiner import (
    RefinerConfig as StateConfig, StreamRefiner as StateRefiner, LAYERS, STAT_NAMES,
)
from ..h2pro_snn_v2_2.refiner import layer_stats, signed_voltage_tail


@dataclass(frozen=True)
class RefinerConfig(StateConfig):
    version: str = "phase_grid_ann_readout_v1"

    def __post_init__(self):
        if self.version != "phase_grid_ann_readout_v1":
            raise ValueError("Expected the independently versioned linear-readout experiment")
        values = asdict(self)
        values["version"] = "phase_grid_ann_state_v1"
        StateConfig(**values)


def normalized_linear_state(previous, current, decay):
    """Signed EMA in physical voltage units, with true linear recurrence gradient."""
    return previous * decay + current * (1.0-decay)


class StreamRefiner(StateRefiner):
    def __init__(self, config=RefinerConfig(), foreground_prior=0.1):
        super().__init__(config, foreground_prior)
        self.last_readout_extrema = None

    def forward(self, inputs, state=None):
        c = self.config
        if (inputs.ndim != 5 or inputs.shape[2:] != (64, c.rows, c.columns)
                or inputs.shape[0] < 1 or inputs.shape[1] < 1 or inputs.dtype != self.stem.weight.dtype):
            raise ValueError("Expected FP32 [steps,batch,64,rows,columns]")
        state = self.initial_state(inputs.shape[1]) if state is None else state
        expected = (c.stem_channels, c.hidden_channels, 128, 1)
        if len(state) != 4 or any(s.shape != (inputs.shape[1], n, c.rows, c.columns)
                                 for s, n in zip(state, expected)):
            raise ValueError("Stream state shape mismatch")
        stem_v, hidden_v, head_v, previous = state
        logits, rates, costs, diagnostics, voltage_tails, readout_extrema = [], [], [], [], [], []
        for frame in inputs:
            activity = frame[:, :32].sum(1, keepdim=True)
            # No bias: zero input has zero current and zero weight gradient.
            # Project only occupied cells; this is an implemented MAC saving.
            flat = frame.permute(0, 2, 3, 1).reshape(-1, 64)
            indices = (flat != 0).any(1).nonzero(as_tuple=True)[0]
            projected = F.linear(flat[indices], self.stem.weight[:, :, 0, 0])
            full = frame.new_zeros((len(flat), c.stem_channels)).index_copy(0, indices, projected)
            current = full.view(inputs.shape[1], c.rows, c.columns, -1).permute(0, 3, 1, 2)
            stem_pre = stem_v*self.stem_decay + current
            stem_s, stem_v = self._lif(stem_pre)
            decayed = hidden_v*self.hidden_decay
            # Unchanged ANN-state v1 continuous charge and hidden dynamics.
            hidden_current = self.hidden(stem_s*c.threshold)
            hidden_pre = decayed + hidden_current
            hidden_s, hidden_v = self._lif(hidden_pre)
            head_current = self.head(hidden_s*c.threshold)
            head_pre = normalized_linear_state(head_v, head_current, self.head_decay)
            voltage_tails.append(signed_voltage_tail(head_pre, c.head_voltage_soft_limit))
            head_v = head_pre  # Signed state is retained without rectification/reset.
            # Monitor-only positive amplitude: preserve the former 256 guard.
            head_positive_units = F.relu(head_pre.detach()/c.threshold)
            with torch.no_grad():
                readout_extrema.append(torch.stack((head_pre.amin((1, 2, 3)), head_pre.amax((1, 2, 3))), -1))
            diagnostics.append(torch.stack((layer_stats(current, stem_pre, stem_s),
                                            layer_stats(hidden_current, hidden_pre, hidden_s),
                                            layer_stats(head_current, head_pre, head_positive_units)), 1))
            # Same class/prototype pooling and one-time bias; only z changed.
            raw = head_pre.reshape(inputs.shape[1], 32, 4, c.rows, c.columns)
            raw = raw + self.bias[None, :, :, None, None]
            grouped = raw.reshape(inputs.shape[1], 32, 2, 2, c.rows, c.columns).amax(3)
            logits.append(grouped[:, :, 1]-grouped[:, :, 0])
            support = F.max_pool2d((activity > 0).to(frame.dtype), 3, 1, 1)
            denominator = support.sum((1, 2, 3)).clamp_min(1)
            rates.append(torch.stack(((stem_s*support).sum((1, 2, 3))/(denominator*c.stem_channels),
                                      (hidden_s*support).sum((1, 2, 3))/(denominator*c.hidden_channels)), 1))
            # Per-stream counts retained on device until the caller summarizes.
            costs.append(torch.stack(((activity > 0).sum((1, 2, 3)),
                                      (frame != 0).sum((1, 2, 3)),
                                      stem_s.detach().sum((1, 2, 3)),
                                      hidden_s.detach().sum((1, 2, 3)),
                                      frame.new_zeros(inputs.shape[1])), 1))
            previous = activity.detach()
        self.last_diagnostics = torch.stack(diagnostics)
        self.last_voltage_tail = torch.stack(voltage_tails)
        self.last_readout_extrema = torch.stack(readout_extrema)
        return (torch.stack(logits), (stem_v, hidden_v, head_v, previous),
                torch.stack(rates), torch.stack(costs))

    def architecture(self):
        result = super().architecture()
        result.update(
            readout="Signed normalized linear EMA, no firing/reset; four prototypes/query plus one-time bias",
            control="Only head recurrence replaced; stem/hidden and all learning hyperparameters match ANN-state v1",
            numerics="stem/hidden: continuous ReLU charge with detached subtraction; head: signed linear recurrence",
            activation_note="Head positive-part/threshold is diagnostic only. Head signed extrema are also recorded; no spike or AC claim",
            gradient_note="Stem/hidden reset remains detached; head has the exact gradient of its new linear recurrence",
            head_update="z_t = exp(-10/20)*z_(t-1) + (1-exp(-10/20))*I_t",
        )
        return result
