"""Actual zero-skip MAC / binary-spike AC execution, with exact S1 verification."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch

from ..fixed import (FixedProposalSNN, _shift_channels, load_fixed_proposal_snn,
                     saturate_signed)
from ..snn import load_proposal_snn_export, profile_proposal_snn
from ..h2pro_snn.foundation import IMPLEMENTATION
from .common import MODES


@dataclass
class Cost:
    proposal_mac: int = 0
    proposal_ac: int = 0
    proposal_dense_equivalent_mac: int = 0
    refiner_mac: int = 0
    hidden1_spikes: int = 0
    hidden2_spikes: int = 0
    output_diagnostic_spikes: int = 0
    nonzero_temporal_inputs: int = 0
    proposal_layer_output_scalars: int = 0
    refiner_layer_output_scalars: int = 0
    state_update_scalars: int = 0
    readout_sum_additions: int = 0
    readout_mean_divisions: int = 0
    verification_dense_proposal_mac: int = 0
    reference_exact_windows: int = 0
    nonempty_windows: int = 0

    def merge(self, other):
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)


def sparse_dot(value, weights, *, binary=False):
    """weights is [input,output]; only nonzero input rows trigger arithmetic.

    Accumulate in int64 and saturate AFTER the full dot, as the S1 golden model.
    Zero weights are not skipped; binary values trigger additions with no multiply.
    """
    value, weights = np.asarray(value), np.asarray(weights)
    if (value.ndim != 2 or weights.ndim != 2 or value.shape[1] != weights.shape[0]
            or value.dtype != np.uint8 or weights.dtype != np.int8):
        raise ValueError('Require uint8 [batch,input] and int8 [input,output]')
    if binary and np.any(value > 1):
        raise ValueError('AC input must be binary spikes')
    out = np.zeros((len(value), weights.shape[1]), dtype=np.int64)
    w = weights.astype(np.int64)
    operations = 0
    for col in range(value.shape[1]):
        rows = np.flatnonzero(value[:, col])
        if len(rows):
            if binary:
                out[rows] += w[col]
            else:
                out[rows] += value[rows, col, None].astype(np.int64) * w[col]
            operations += len(rows) * weights.shape[1]
    return out, operations


def sparse_ann(features, layers):
    """Numerically identical original Proposal; zero-skip affects execution only."""
    value = np.ascontiguousarray(features, dtype=np.uint8)
    operations = 0
    output = np.empty((len(value), layers[-1][0].shape[1]), np.float32)
    for i, (weight, scale) in enumerate(layers):
        raw, count = sparse_dot(value, weight)
        operations += count
        floating = raw.astype(np.float64) * scale[None, :].astype(np.float64) / 255.0
        if i + 1 < len(layers):
            value = np.clip(np.rint(np.clip(floating, 0., 1.) * 255.), 0., 255.).astype(np.uint8)
        else:
            output = floating.astype(np.float32)
    return output, operations


class EventFixedProposal(FixedProposalSNN):
    """Reuse frozen S1 leak, rounding, reset and readout; change dot backend only."""
    def __init__(self, source):
        super().__init__(source.weights, source.weight_fraction_bits, source.threshold,
                         source.time_steps, source.readout, source.configuration)
        self.mac = self.ac = 0

    def _current(self, value, layer, input_fraction_bits):
        raw, count = sparse_dot(value, self.weights[layer].T, binary=layer > 0)
        if layer == 0:
            self.mac += count
        else:
            self.ac += count
        raw, saturation_count = saturate_signed(raw, self.configuration.accumulator_bits)
        shifts = (self.configuration.membrane_fraction_bits - input_fraction_bits
                  - self.weight_fraction_bits[layer].astype(np.int64))
        return _shift_channels(raw, shifts), saturation_count

    def forward(self, features):
        if np.asarray(features).dtype != np.uint8:
            raise ValueError('S1 transfer input must be uint8')
        self.mac = self.ac = 0
        return super().forward(features)


class ProposalExecutor:
    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError('Unknown Proposal transfer mode')
        self.mode = mode
        self.cost = Cost()
        self.s0 = self.fixed = self.event = None
        root = IMPLEMENTATION / 'software/weights'
        if mode == 's0p_dense':
            self.s0, _, _ = load_proposal_snn_export(root / 'S0P_T4_POS15_KD50_S42.npz')
            self.s0.eval().requires_grad_(False)
        if mode == 's1_event':
            self.fixed, _, _ = load_fixed_proposal_snn(root / 'S1P_INT8_ACC24_MEM24_Q12.npz')
            self.event = EventFixedProposal(self.fixed)
        self.saturations = {'accumulator': [0, 0, 0], 'membrane': [0, 0, 0, 0]}

    def __call__(self, patch, layers, temporal=None):
        c = Cost(nonempty_windows=1)
        n = len(patch)
        if self.mode.startswith('ann_'):
            from wangdao.h2.numpy_model import quantized_mlp
            expected = n * sum(w.size for w, _ in layers)
            c.proposal_dense_equivalent_mac = expected
            c.proposal_layer_output_scalars = n * 88
            if self.mode == 'ann_dense':
                logits = quantized_mlp(patch, layers)
                c.proposal_mac = expected
            else:
                logits, c.proposal_mac = sparse_ann(patch, layers)
                reference = quantized_mlp(patch, layers)
                if not np.array_equal(logits, reference):
                    raise AssertionError('ANN sparse Proposal is not exact')
                c.verification_dense_proposal_mac = expected
                c.reference_exact_windows = 1
        else:
            if temporal is None or temporal.shape != (4, n, 128) or temporal.dtype != np.uint8:
                raise ValueError('Require four real 25ms micro-bin feature tensors')
            c.proposal_dense_equivalent_mac = 4 * n * 7680
            c.nonzero_temporal_inputs = int(np.count_nonzero(temporal))
            c.proposal_layer_output_scalars = 4 * n * 88
            # Includes the inherited output-spike diagnostic cell (no synaptic fanout).
            c.state_update_scalars = 4 * n * (48 + 24 + 16 + 16)
            c.readout_sum_additions = 4 * n * 16
            c.readout_mean_divisions = n * 16
            if self.mode == 's0p_dense':
                assert self.s0 is not None
                with torch.no_grad():
                    activity = profile_proposal_snn(self.s0, torch.from_numpy(temporal))
                logits = activity.output.logits.numpy().copy()
                spikes = activity.spike_counts
                c.proposal_mac = c.proposal_dense_equivalent_mac
            else:
                assert self.event is not None and self.fixed is not None
                actual = self.event.forward(temporal)
                reference = self.fixed.forward(temporal)
                if (not np.array_equal(actual.logits_q, reference.logits_q)
                        or not np.array_equal(actual.logits, reference.logits)
                        or actual.spike_counts != reference.spike_counts
                        or actual.accumulator_saturations != reference.accumulator_saturations
                        or actual.membrane_saturations != reference.membrane_saturations):
                    raise AssertionError('Event S1 differs from the frozen dense golden model')
                logits, spikes = actual.logits, actual.spike_counts
                c.proposal_mac, c.proposal_ac = self.event.mac, self.event.ac
                c.verification_dense_proposal_mac = c.proposal_dense_equivalent_mac
                c.reference_exact_windows = 1
                if c.proposal_ac != 24 * spikes[0] + 16 * spikes[1]:
                    raise AssertionError('Executed binary AC differs from spike fanout')
                for key, values in (('accumulator', actual.accumulator_saturations),
                                    ('membrane', actual.membrane_saturations)):
                    self.saturations[key] = [a + b for a, b in zip(self.saturations[key], values)]
            c.hidden1_spikes, c.hidden2_spikes, c.output_diagnostic_spikes = spikes
        if not np.isfinite(logits).all():
            raise FloatingPointError('Nonfinite Proposal output')
        self.cost.merge(c)
        return logits
