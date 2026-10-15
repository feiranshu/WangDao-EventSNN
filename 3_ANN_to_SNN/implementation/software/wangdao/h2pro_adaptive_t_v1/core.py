"""Independent incremental binary-IF implementation and prefix-only exit rules."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
import torch

from ..h2pro_calibration_v1.core import probabilities, quantize_hidden, validate_layers

STEPS = (4, 8, 16)
THETA, M0, THRESHOLD = 255, 191, .35


@dataclass(frozen=True)
class Policy:
    negative: float | None = None
    positive: float | None = None
    mode: str = 'confidence'

    def validate(self):
        if (self.mode not in ('confidence', 'stable8')
                or (self.negative is not None and not 0 <= self.negative < THRESHOLD)
                or (self.positive is not None and not THRESHOLD < self.positive <= 1)):
            raise ValueError('Exit thresholds must bracket fixed classification threshold 0.35')
        return self

    def as_dict(self):
        return asdict(self)


def should_exit(probability, step, policy, previous=None):
    """Only current/past predictions; no label, teacher, density tag or future T."""
    policy.validate()
    if step not in STEPS:
        raise ValueError('Only 4/8/16 coding steps')
    if step == 16:
        return np.ones(len(probability), dtype=bool)
    if policy.mode == 'stable8' and step == 4:
        return np.zeros(len(probability), dtype=bool)
    accepted = np.zeros(len(probability), dtype=bool)
    if policy.negative is not None:
        accepted |= probability <= policy.negative
    if policy.positive is not None:
        accepted |= probability >= policy.positive
    if policy.mode == 'stable8':
        if previous is None or previous.shape != probability.shape:
            raise ValueError('T8 stability requires the available T4 output')
        accepted &= (probability >= THRESHOLD) == (previous >= THRESHOLD)
    return accepted


def choose(probabilities_by_step, policy):
    """Offline replay of the SAME causal gate, used only for Main calibration."""
    p = np.asarray(probabilities_by_step)
    if p.ndim != 2 or p.shape[0] != 3 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Finite [3,N] probabilities required')
    chosen = np.full(p.shape[1], 2, np.uint8)
    active = np.ones(p.shape[1], bool)
    for index, step in enumerate(STEPS[:2]):
        ids = np.flatnonzero(active)
        stop = should_exit(p[index, ids], step, policy, p[0, ids] if step == 8 else None)
        done = ids[stop]
        chosen[done] = index
        active[done] = False
    return chosen


def validate_features(features, layers):
    validate_layers(layers)
    if features.dtype != np.uint8 or features.ndim != 2 or features.shape[1] != 36:
        raise ValueError('Original 36-dimensional uint8 features required')


@torch.no_grad()
def trace_reference(features, layers, device='cpu', batch_size=8192):
    """Dense FP64 count reference for ALL prefixes; not deployed early exit.

    This computes calibration evidence, not an AC-executing GPU kernel. All dot
    products have bounded integer operands, hence FP64 sums are exact.
    """
    validate_features(features, layers)
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive batch size required')
    weights = [torch.tensor(w, dtype=torch.float64, device=device) for w, _ in layers]
    scales = [torch.tensor(s, dtype=torch.float64, device=device) for _, s in layers]
    raw = np.empty((3, len(features), 4), np.float32)
    spikes = np.empty((3, len(features)), np.int64)
    for lo in range(0, len(features), batch_size):
        x = torch.tensor(features[lo:lo+batch_size], dtype=torch.float64, device=device)
        h1 = torch.round(((x @ weights[0])*scales[0]/255).clamp(0,1)*255)
        for i, step in enumerate(STEPS):
            count = torch.floor((h1*step+M0)/THETA)
            h2 = torch.round(((count @ weights[1])*scales[1]/step).clamp(0,1)*255)
            y = ((h2 @ weights[2])*scales[2]/255).float()
            raw[i,lo:lo+len(x)] = y.cpu().numpy()
            spikes[i,lo:lo+len(x)] = count.sum(dim=1).to(torch.int64).cpu().numpy()
    return raw, spikes


def incremental(features, layers, policy, batch_size=8192):
    """Actual CPU binary spike-add, stopping individual candidates at 4/8/16.

    Hidden1 projection is computed once. Membrane AND signed hidden2 accumulator
    survive checkpoints; only remaining rows advance. No lookahead is computed.
    Readout is recomputed at each visited checkpoint and charged to the ledger.
    """
    validate_features(features, layers); policy.validate()
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('Positive batch size required')
    n = len(features)
    result = np.empty((n,4), np.float32)
    exit_index = np.empty(n, np.uint8)
    final_spikes = np.empty(n, np.int64)
    measured = dict(input_projection_mac=0, readout_mac=0, synaptic_ac=0,
                    if_adds=0, if_comparisons=0, if_resets=0,
                    readout_evaluations=0, active_step_rows=0,
                    gate_threshold_comparisons=0, stability_comparisons=0,
                    exit_boolean_ors=0, exit_boolean_ands=0)
    w0, w1, w2 = [w.astype(np.int32) for w, _ in layers]
    s0, s1, s2 = [s.astype(np.float64) for _, s in layers]
    for lo in range(0, n, batch_size):
        x = features[lo:lo+batch_size]
        current = quantize_hidden((x.astype(np.int32) @ w0).astype(np.float64)*s0/255).astype(np.int16)
        measured['input_projection_mac'] += len(x)*2304
        active = np.arange(len(x), dtype=np.int64)
        membrane = np.full(current.shape, M0, np.int16)
        accum = np.zeros((len(x),32), np.int32)
        count = np.zeros(len(x), np.int64)
        previous = None
        for step in range(1,17):
            if not len(active):
                break
            membrane += current
            fired = membrane >= THETA
            for channel in np.flatnonzero(fired.any(axis=0)):
                ids = np.flatnonzero(fired[:,channel])
                accum[ids] += w1[channel]
                measured['synaptic_ac'] += len(ids)*32
            per_row = fired.sum(axis=1, dtype=np.int64)
            count += per_row
            membrane[fired] -= THETA
            measured['if_adds'] += len(active)*64
            measured['if_comparisons'] += len(active)*64
            measured['if_resets'] += int(per_row.sum())
            measured['active_step_rows'] += len(active)
            if step not in STEPS:
                continue
            h2 = quantize_hidden(accum.astype(np.float64)*s1/step)
            y = ((h2.astype(np.int32) @ w2).astype(np.float64)*s2/255).astype(np.float32)
            p = probabilities(y)
            stop = should_exit(p, step, policy, previous)
            measured['readout_mac'] += len(active)*128
            measured['readout_evaluations'] += len(active)
            if step < 16 and not (policy.mode == 'stable8' and step == 4):
                measured['gate_threshold_comparisons'] += len(active)*((policy.negative is not None)+(policy.positive is not None))
                measured['exit_boolean_ors'] += len(active)*((policy.negative is not None)+(policy.positive is not None))
                if policy.mode == 'stable8':
                    measured['stability_comparisons'] += len(active)*3
                    measured['exit_boolean_ands'] += len(active)
            done = lo + active[stop]
            result[done], exit_index[done], final_spikes[done] = y[stop], STEPS.index(step), count[stop]
            keep = ~stop
            previous = p[keep]
            active, current, membrane, accum, count = (v[keep] for v in (active,current,membrane,accum,count))
    return result, final_spikes, exit_index, measured


def ledger(exit_index, spikes, policy=None, fixed=False):
    """Deployment-equivalent operation counts, excluding physical/traffic costs."""
    ix, spikes = np.asarray(exit_index), np.asarray(spikes)
    if (ix.ndim != 1 or spikes.shape != ix.shape or not np.isin(ix,[0,1,2]).all()
            or np.any(spikes < 0)):
        raise ValueError('Invalid exit/spike vectors')
    n = len(ix); counts = [int(np.count_nonzero(ix==i)) for i in range(3)]
    calls = n if fixed else sum((i+1)*c for i,c in enumerate(counts))
    steps = sum(t*c for t,c in zip(STEPS,counts))
    resets = int(spikes.sum(dtype=np.int64))
    gates = stability = ors = ands = 0
    if not fixed:
        if policy is None:
            raise ValueError('Adaptive cost requires gate policy')
        policy.validate()
        eligible = n-counts[0] if policy.mode=='stable8' else n+(n-counts[0])
        gates = eligible*((policy.negative is not None)+(policy.positive is not None))
        ors = gates
        if policy.mode=='stable8':
            stability, ands = (n-counts[0])*3, n-counts[0]
    return dict(input_projection_mac=n*2304,readout_mac=calls*128,synaptic_ac=resets*32,
        if_adds=steps*64,if_comparisons=steps*64,if_resets=resets,readout_evaluations=calls,
        active_step_rows=steps,gate_threshold_comparisons=gates,stability_comparisons=stability,
        exit_boolean_ors=ors,exit_boolean_ands=ands,
        decode_scale_mul=n*64+calls*36,decode_divisions=n*64+calls*36,
        quantizer_scale_mul=n*64+calls*32,quantizer_round=n*64+calls*32,
        quantizer_comparisons=4*(n*64+calls*32),
        probability_max_comparisons=calls*3,probability_subtractions=calls*2,
        probability_exponentials=calls*2,probability_sum_adds=calls,probability_divisions=calls,
        final_decision_comparisons=n,exit_counts=dict(zip(map(str,STEPS),counts)),
        average_steps=steps/n if n else None,weight_bytes_int8=4480,scale_bytes_fp32=400,
        per_active_candidate_state=dict(membrane_elements=64,weighted_accumulator_elements=32,
            cached_hidden1_elements=64,previous_probability_elements=1),
        memory_traffic_measured=False,energy_measured=False,
        unaccounted=['feature construction','Proposal routing/components','active-row compaction/control instructions',
                     'memory traffic','physical instruction cost'],
        scope='Refiner inference; input projection once, repeated readout included; scalar/state counts separate')
