"""Full-event metrics, fixed Proposal rejects, and explicit cost boundaries."""
from __future__ import annotations

import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

from ..h2pro_calibration_v1.core import forward, probabilities, validate_layers
from ..h2pro_snn.diagnostics import aggregate_counts, error_counts
from .common import ENCODERS, STAGE, STEPS, THRESHOLD, VERSION


def summarize(rows):
    result = aggregate_counts(rows)
    denom = 2*result['tp'] + result['fp'] + result['fn']
    result['f1'] = 2*result['tp']/denom if denom else None
    return result


def score(data, raw, spikes, role='train_dev'):
    if spikes.shape != (len(raw), 2) or np.any(spikes[:, 1]):
        raise ValueError('r1 has only hidden1 spikes; hidden2/readout remain ANN')
    records = []
    for seq in data.sequences[role]:
        row, lo, hi = seq['row'], seq['lo'], seq['hi']
        p = np.zeros(row['events'], np.float32)
        p[seq['candidate']] = probabilities(raw[lo:hi])
        counts = error_counts(seq['labels'], p, seq['candidate'], THRESHOLD)
        records.append({'sequence': row['sequence'], 'high_density': row['high_density'],
                        'counts': counts, 'hidden_spikes': spikes[lo:hi].sum(axis=0).tolist()})
    n = len(raw)
    sums = spikes.sum(axis=0)
    proposal = sum(s['row']['proposal_sparse_mac'] for s in data.sequences[role])
    dense = sum(s['row']['proposal_dense_mac'] for s in data.sequences[role])
    ac = int(32*sums[0])
    return {'overall': summarize([r['counts'] for r in records]),
            'high_density': summarize([r['counts'] for r in records if r['high_density']]),
            'ordinary': summarize([r['counts'] for r in records if not r['high_density']]),
            'sequences': records, 'threshold': THRESHOLD, 'role': role,
            'cost': {'inference_equivalent_mac_sparse_proposal': proposal+n*2432,
                     'inference_equivalent_mac_dense_proposal': dense+n*2432,
                     'inference_equivalent_ac': ac, 'refiner_mac': n*2432,
                     'proposal_sparse_mac': proposal, 'proposal_dense_mac': dense,
                     'frozen_ann_sparse_mac': proposal+n*4480, 'frozen_ann_dense_mac': dense+n*4480,
                     'executed_reference_refiner_mac': n*4480, 'executed_reference_refiner_ac': 0,
                     'hidden1_spikes': int(sums[0]), 'hidden2_spikes': int(sums[1]),
                     'equivalent_if_state_adds': n*64*STEPS, 'equivalent_if_comparisons': n*64*STEPS,
                     'equivalent_if_resets': int(sums.sum()),
                     'decode_scale_mul': n*100, 'decode_divisions': n*100,
                     'hidden_quantizer_mul': n*96, 'hidden_quantizer_round': n*96,
                     'count_reference_mul_add_div_each': n*64,
                     'weight_bytes_int8': 4480, 'scale_bytes_fp32': 400,
                     'peak_serial_if_state_elements_per_candidate': 64,
                     'memory_traffic_measured': False, 'energy_measured': False,
                     'scope': 'neural inference; extra training/QAT/backward/optimizer cost excluded',
                     'unaccounted': ['feature construction', 'routing/components', 'prototype readout',
                                     'memory traffic', 'physical instruction costs']}}


@torch.no_grad()
def evaluate(model, data, batch_size, pause=None, verify_explicit=True):
    started = time.perf_counter()
    model.eval()
    raw_parts, spike_parts = [], []
    for lo in range(0, data.size('train_dev'), batch_size):
        if pause is not None:
            pause()
        x = data.roles['train_dev']['features'][lo:lo+batch_size]
        raw, spikes = model(x)
        raw_parts.append(raw.cpu().numpy())
        spike_parts.append(spikes.cpu().numpy())
    raw = np.concatenate(raw_parts)
    spikes = np.concatenate(spike_parts)
    verified = 0
    if verify_explicit:
        layers = model.export_layers()
        # Fixed, label-independent coverage of EVERY development sequence.
        for seq in data.sequences['train_dev']:
            lo, hi = seq['lo'], seq['hi']
            ids = np.unique(np.linspace(lo, hi-1, min(64, hi-lo), dtype=np.int64)) if hi > lo else np.empty(0, np.int64)
            x = data.roles['train_dev']['features'][torch.as_tensor(ids, device=data.device)].cpu().numpy()
            reference, cost = forward(x, layers, STAGE, STEPS, ENCODERS, 'spike_add')
            np.testing.assert_array_equal(reference, raw[ids])
            if (cost.executed_synaptic_ac != int(spikes[ids, 0].sum()*32)
                    or cost.executed_synaptic_mac != len(ids)*2432):
                raise AssertionError('Spike-add cost differs from the exported student')
            verified += len(ids)
    result = score(data, raw, spikes)
    result.update(inference_seconds=time.perf_counter()-started, explicit_verified_candidates=verified,
                  export_verified=True, student_backend='dense count reference, exact int8 forward')
    return result, raw, spikes


def export(path, model):
    path = Path(path)
    model.verify_frozen_scales()
    arrays: dict[str, Any] = {'dimensions': np.array([36, 64, 32, 4]), 'stage': np.array(STAGE), 'steps': np.array(STEPS),
              'threshold': np.array(THRESHOLD), 'variant': np.array('C1-proposal-motion'),
              'version': np.array(VERSION),
              'encoder_thresholds': np.array([e.threshold for e in ENCODERS]),
              'encoder_initial_membranes': np.array([e.initial_membrane for e in ENCODERS]),
              'spiking_transport_layers': np.array([1])}
    for i, (w, s) in enumerate(model.export_layers()):
        arrays[f'layer{i}_weight'], arrays[f'layer{i}_scale'] = w, s
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_export(path):
    """Reject accidentally decoding this calibrated export with default m0=127."""
    with np.load(path, allow_pickle=False) as z:
        if (z['version'].item() != VERSION or z['stage'].item() != STAGE
                or z['steps'].item() != STEPS or z['threshold'].item() != THRESHOLD
                or z['variant'].item() != 'C1-proposal-motion'
                or not np.array_equal(z['dimensions'], [36, 64, 32, 4])
                or not np.array_equal(z['encoder_thresholds'], [e.threshold for e in ENCODERS])
                or not np.array_equal(z['encoder_initial_membranes'], [e.initial_membrane for e in ENCODERS])
                or not np.array_equal(z['spiking_transport_layers'], [1])):
            raise ValueError('Wrong r1/T16/task_rescore export contract')
        layers = tuple((z[f'layer{i}_weight'].copy(), z[f'layer{i}_scale'].copy()) for i in range(3))
    validate_layers(layers)
    return layers


def delta(result, baseline):
    return {group: {key: 100*(result[group][key]-baseline[group][key])
                    if result[group][key] is not None and baseline[group][key] is not None else None
                    for key in ('iou', 'precision', 'recall', 'f1')}
            for group in ('overall', 'high_density', 'ordinary')}
