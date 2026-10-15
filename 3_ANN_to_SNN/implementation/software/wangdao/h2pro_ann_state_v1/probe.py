"""Read-only, bounded tracing of causal state and counterfactual head readouts.

Counterfactuals share the executed SNN stem/hidden states; only their head
readout changes. Continuous-cell transplant is a separate full forward pass.
No labels or teacher values enter any network or counterfactual computation.
"""
from __future__ import annotations

import torch

from ..h2pro_snn_v2_2.refiner import StreamRefiner as SNNRefiner
from .refiner import analog_cell

SCALE_NAMES = ("current_min", "current_max", "carry_min", "carry_max", "pre_min", "pre_max",
               "residual_min", "residual_max", "negative_residual_units", "positive_current_units",
               "positive_current_blocked_by_negative_carry_units", "grid_units")
READOUTS = ("original", "current_only", "leaky_no_reset", "normalized_leaky_no_reset")


def pool_prototypes(values, bias):
    """Event prototypes [N,4] and their matching bias [N,4]."""
    raw = values + bias
    bg, bi = raw[:, :2].max(1)
    fg, fi = raw[:, 2:].max(1)
    return fg-bg, torch.stack((bi, fi+2), 1)


class ReadoutProbe(SNNRefiner):
    def __init__(self, config, foreground_prior, *, analog=False):
        super().__init__(config, foreground_prior)
        self.analog = analog
        self._in_probe = False
        self._hidden_current = self._head_current = None
        self.hidden.register_forward_hook(self._capture_hidden)
        self.head.register_forward_hook(self._capture_head)

    def _capture_hidden(self, module, args, output):
        if self._in_probe:
            self._hidden_current = output

    def _capture_head(self, module, args, output):
        if self._in_probe:
            self._head_current = output

    def _lif(self, voltage):
        activation, residual = (analog_cell(voltage, self.config.threshold) if self.analog
                                else super()._lif(voltage))
        if not self._in_probe:
            return activation, residual
        layer, step = self._cell_index % 3, self._cell_index // 3
        decay = (self.stem_decay, self.hidden_decay, self.head_decay)[layer]
        carry = self._previous[layer] * decay
        if layer == 0:
            current = voltage - carry
        else:
            current = self._hidden_current if layer == 1 else self._head_current
            if current is None:
                raise AssertionError("Missing exact convolution output")
        axes = (1, 2, 3)
        flat = voltage.flatten(1)
        scale = torch.stack((current.amin(axes), current.amax(axes), carry.amin(axes), carry.amax(axes),
                             voltage.amin(axes), voltage.amax(axes), residual.amin(axes), residual.amax(axes),
                             (residual < 0).sum(axes), (current > 0).sum(axes),
                             ((current > 0) & (voltage <= 0) & (carry < 0)).sum(axes),
                             voltage.new_full((len(voltage),), flat.shape[1])), -1)
        self._scales.append(scale)
        self._previous[layer] = residual
        if layer == 2:
            self._analog_head = self._analog_head*self.head_decay + current
            indices = self._maps[0] == step
            _, slot, phase, row, column = self._maps[:, indices]
            def query(value):
                return value.reshape(len(voltage), 32, 4, self.config.rows, self.config.columns).permute(
                    0, 1, 3, 4, 2)[slot, phase, row, column]
            self._events["pre"][indices] = query(voltage)
            self._events["current"][indices] = query(current)
            self._events["carry"][indices] = query(carry)
            self._events["unreset"][indices] = query(self._analog_head)
        self._cell_index += 1
        return activation, residual

    @torch.no_grad()
    def probe(self, inputs, maps, state=None, auxiliary_head=None):
        state = self.initial_state(inputs.shape[1]) if state is None else state
        self._previous = list(state[:3])
        self._maps = torch.as_tensor(maps, dtype=torch.long, device=inputs.device)
        self._analog_head = torch.zeros_like(state[2]) if auxiliary_head is None else auxiliary_head
        self._events = {k: inputs.new_zeros((self._maps.shape[1], 4))
                        for k in ("pre", "current", "carry", "unreset")}
        self._scales, self._cell_index = [], 0
        self._in_probe = True
        try:
            output, next_state, rates, counts = super().forward(inputs, state)
            event_bias = self.bias[self._maps[2]]
            details = self._events
            original, winners = pool_prototypes(details["pre"], event_bias)
            scores = {"original": original,
                      "current_only": pool_prototypes(details["current"], event_bias)[0],
                      "leaky_no_reset": pool_prototypes(details["unreset"], event_bias)[0],
                      "normalized_leaky_no_reset": pool_prototypes(
                          details["unreset"]*(1-self.head_decay), event_bias)[0]}
            actual = output[tuple(self._maps)]
            if not torch.equal(actual, original):
                raise AssertionError("Probe changed the original event readout")
            for key in ("pre", "current", "carry"):
                selected = details[key].gather(1, winners)
                details[key+"_winning_bg_fg"] = selected
                details[key+"_margin"] = selected[:, 1]-selected[:, 0]
            details["bias"] = event_bias
            selected_bias = event_bias.gather(1, winners)
            details["bias_margin"] = selected_bias[:, 1]-selected_bias[:, 0]
            details["winners"] = winners
            torch.testing.assert_close(original, details["current_margin"]+details["carry_margin"]+
                                       details["bias_margin"], rtol=2e-5, atol=2e-5)
            scales = torch.stack(self._scales).reshape(inputs.shape[0], 3, inputs.shape[1], -1).permute(0, 2, 1, 3)
            return scores, next_state, self._analog_head, rates, counts, details, scales
        finally:
            self._in_probe = False
            self._previous = []
            self._hidden_current = self._head_current = None
            self._events = {}
            self._scales = []
