"""Serial IF-rate conversion, retaining every frozen weight and quantizer.

This deliberately conservative baseline preserves the original layer barriers.
It is a hybrid rate-coded network, not a low-latency pipelined temporal SNN.
The count reference is never presented as a multiplication-free execution.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ..h2pro_snn.foundation import ANN_ROOT, verify_baseline

STAGES = ("ann", "r1", "r2")
BACKENDS = ("count_reference", "spike_add")
DIMENSIONS = (36, 64, 32, 4)


@dataclass
class Cost:
    executed_synaptic_mac: int = 0
    executed_synaptic_ac: int = 0
    rate_network_mac: int = 0
    rate_network_ac: int = 0
    scalar_scale_mul: int = 0
    scalar_normalization_div: int = 0
    quantizer_scale_mul: int = 0
    quantizer_round: int = 0
    quantizer_comparisons: int = 0
    if_state_adds: int = 0
    if_comparisons: int = 0
    if_reset_subtractions: int = 0
    unit_spikes: int = 0
    count_reference_integer_mul: int = 0
    count_reference_integer_add: int = 0
    count_reference_integer_div: int = 0

    def merge(self, other):
        for key, value in asdict(other).items():
            setattr(self, key, getattr(self, key) + value)


def validate_steps(steps):
    # Int32 accumulator bound for max width64 * max int8 magnitude128 * T.
    maximum = int(np.iinfo(np.int32).max)//(64*128)
    if type(steps) is not int or not 1 <= steps <= maximum:
        raise ValueError(f"T must be an integer in [1,{maximum}] (exact accumulator range)")


def validate_layers(layers):
    if len(layers) != 3:
        raise ValueError("Expected the original three Refiner layers")
    for i, (weight, scale) in enumerate(layers):
        if (weight.dtype != np.int8 or weight.shape != DIMENSIONS[i:i+2]
                or scale.dtype != np.float32 or scale.shape != (DIMENSIONS[i+1],)
                or not np.isfinite(scale).all() or np.any(scale <= 0)):
            raise ValueError("Frozen 36-64-32-4 int8/scale contract mismatch")


def load_layers(ann_root: Path = ANN_ROOT):
    verify_baseline(ann_root)
    with np.load(ann_root / "software/weights/H2-pro-Causal-ANN-v1_int8.npz", allow_pickle=False) as z:
        if str(z['variant'].item()) != 'C1-proposal-motion' or tuple(z['dimensions']) != DIMENSIONS:
            raise ValueError("Wrong frozen model")
        layers = tuple((z[f'layer{i}_weight'].copy(), z[f'layer{i}_scale'].copy()) for i in range(3))
    validate_layers(layers)
    for pair in layers:
        for a in pair:
            a.setflags(write=False)
    return layers


def quantize_hidden(value):
    # The original frozen ANN's clamp/round, not a newly added amplitude guard.
    return np.clip(np.rint(np.clip(value, 0., 1.) * 255.), 0., 255.).astype(np.uint8)


def spike_counts(activation, steps):
    validate_steps(steps)
    if activation.dtype != np.uint8 or activation.ndim != 2:
        raise ValueError("IF input must be a matrix of original uint8 activations")
    return (activation.astype(np.int32)*steps + 127)//255


def spike_accumulate(activation, weight, steps, backend):
    """Return signed accumulation and actual binary-spike count.

    In spike_add, each transmitted spike adds one original int8 weight vector.
    In count_reference, matmul collapses that stream, retaining the executed MAC.
    """
    validate_steps(steps)
    if (activation.dtype != np.uint8 or activation.ndim != 2 or weight.dtype != np.int8
            or weight.ndim != 2 or activation.shape[1] != weight.shape[0]
            or activation.shape[1] > 64 or backend not in BACKENDS):
        raise ValueError("Invalid IF transport contract")
    cost = Cost()
    rows, width = activation.shape
    fanout = weight.shape[1]
    if backend == 'count_reference':
        counts = spike_counts(activation, steps)
        raw = counts @ weight.astype(np.int32)
        spikes = int(counts.sum(dtype=np.int64))
        cost.executed_synaptic_mac = rows * weight.size
        cost.count_reference_integer_mul = activation.size
        cost.count_reference_integer_add = activation.size
        cost.count_reference_integer_div = activation.size
    else:
        current = activation.astype(np.int16)
        membrane = np.full(activation.shape, 127, dtype=np.int16)
        raw = np.zeros((rows, fanout), dtype=np.int32)
        spikes = 0
        for _ in range(steps):
            membrane += current
            fired = membrane >= 255
            # Selection and addition only; never multiply counts by weights here.
            for channel in np.flatnonzero(fired.any(axis=0)):
                selected = np.flatnonzero(fired[:, channel])
                raw[selected] += weight[channel].astype(np.int32)
            count = int(fired.sum())
            spikes += count
            membrane[fired] -= 255
        cost.executed_synaptic_ac = spikes * fanout
        cost.if_state_adds = rows * width * steps
        cost.if_comparisons = rows * width * steps
        cost.if_reset_subtractions = spikes
    cost.rate_network_ac = spikes * fanout
    cost.unit_spikes = spikes
    return raw, cost


def forward(features, layers, stage='ann', steps=255, backend='count_reference'):
    validate_steps(steps)
    validate_layers(layers)
    if stage not in STAGES or backend not in BACKENDS:
        raise ValueError("Unknown conversion stage/backend")
    if features.dtype != np.uint8 or features.ndim != 2 or features.shape[1] != 36:
        raise ValueError("Retain the original 36-dimensional uint8 features")
    value = features
    cost = Cost()
    for index, (weight, scale) in enumerate(layers):
        spiking = (index == 1 and stage in ('r1','r2')) or (index == 2 and stage == 'r2')
        if spiking:
            raw, transport = spike_accumulate(value, weight, steps, backend)
            cost.merge(transport)
            denominator = steps
        else:
            raw = value.astype(np.int32) @ weight.astype(np.int32)
            cost.executed_synaptic_mac += len(value) * weight.size
            cost.rate_network_mac += len(value) * weight.size
            denominator = 255
        floating = raw.astype(np.float64) * scale[None,:].astype(np.float64) / denominator
        cost.scalar_scale_mul += floating.size
        cost.scalar_normalization_div += floating.size
        if index < 2:
            value = quantize_hidden(floating)
            cost.quantizer_scale_mul += floating.size
            cost.quantizer_round += floating.size
            cost.quantizer_comparisons += 4 * floating.size
        else:
            return floating.astype(np.float32), cost
    raise AssertionError("Missing original readout")


def probabilities(raw):
    if raw.ndim != 2 or raw.shape[1] != 4 or not np.isfinite(raw).all():
        raise ValueError("Expected finite original four-prototype output")
    logits = raw.reshape(-1,2,2).max(axis=2)
    shifted = logits - logits.max(axis=1,keepdims=True)
    exponential = np.exp(shifted.astype(np.float64))
    return (exponential[:,1]/exponential.sum(axis=1)).astype(np.float32)
