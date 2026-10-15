"""Build label-independent, all-event feature evidence once per sequence."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from ..h2pro_proposal_transfer_v1.core import ProposalExecutor
from ..h2pro_proposal_transfer_v1.pipeline import TransferPipeline, temporal_features
from ..h2pro_proposal_transfer_v1.workflow import metrics
from ..h2pro_snn.diagnostics import error_counts
from .common import load_bundle, save_bundle
from .core import ss, context_features, event_packet, event_scores, refiner


def build(arrays, metadata):
    packet = event_packet(arrays)
    t0, nw = metadata['t0_ms'], metadata['closed_windows']
    boundaries = t0+np.arange(nw+1, dtype=np.float64)*100.
    windows = (np.searchsorted(boundaries, packet['t'], side='right')-1).astype(np.int64)
    base = np.empty((len(packet), 28), np.uint8)
    heatmaps = {m: np.empty((nw, 34, 44), np.float32) for m in ('ann', 's1')}
    executors = {'ann': ProposalExecutor('ann_sparse_proposal'), 's1': ProposalExecutor('s1_event')}
    pipeline = TransferPipeline('ann_dense')
    previous = np.empty(0, dtype=ss.EVENT_DTYPE)
    for w in range(nw):
        lo, hi = np.searchsorted(windows, [w, w+1])
        current = packet[lo:hi]
        if len(current):
            history = np.concatenate((previous, current))
            base[lo:hi] = ss.base_features(history, len(previous), t0)
            fine = np.zeros((1, 2, 68, 88), np.int32)
            np.add.at(fine[0], (current['p'], current['y']//4, current['x']//4), 1)
            patch = ss._fine_counts_to_patch_features(fine).reshape(-1, 128)
            temporal = temporal_features(current, t0, w)
            for mode, executor in executors.items():
                raw = executor(patch, pipeline.proposal.proposal_layers, temporal if mode == 's1' else None)
                scores = ss._sigmoid(raw).reshape(1, ss.PATCHES_PER_WINDOW, 16)
                heatmaps[mode][w] = ss.patch_scores_to_heatmaps(scores)[0]
        else:
            for value in heatmaps.values():
                value[w].fill(.5)
        previous = current
    result = {'base': base, 'windows': windows}
    for mode, hm in heatmaps.items():
        features = context_features(packet, windows, base, hm, .51, t0)
        result[mode+'_heatmaps'] = hm
        result[mode+'_scores'] = event_scores(packet, windows, hm)
        result[mode+'_prob_all'] = refiner(features, pipeline.refiner_layers)
    return result, {m: asdict(e.cost) for m, e in executors.items()}


def cache_sequence(directory, store, role, name):
    record = store.get(role, name)
    arrays, meta = record.arrays, record.metadata
    row = store.entries[name][1]
    stem = Path(directory)/'cache'/role/Path(name).stem
    if stem.with_suffix('.json').exists():
        data, receipt = load_bundle(stem)
        if receipt['teacher_cache_sha256'] != row['cache_sha256'] or receipt['role'] != role:
            raise ValueError('Cached sequence identity mismatch')
        return data, receipt, arrays, meta
    started = time.perf_counter()
    data, costs = build(arrays, meta)
    candidate = data['ann_scores'] >= .51
    prob = np.where(candidate, data['ann_prob_all'], 0).astype(np.float32)
    np.testing.assert_array_equal(candidate, arrays['candidate'])
    np.testing.assert_array_equal(prob, arrays['teacher'])
    counts = {}
    for mode in ('ann', 's1'):
        chosen = data[mode+'_scores'] >= .51
        counts[mode] = error_counts(arrays['label'], np.where(chosen, data[mode+'_prob_all'], 0), chosen)
    receipt = {'role': role, 'sequence': name, 'high_density': meta['high_density'],
               'teacher_cache_sha256': row['cache_sha256'], 'costs': costs, 'default_counts': counts,
               'events': len(arrays['t']), 'closed_windows': meta['closed_windows'],
               'ann_teacher_exact': True, 'seconds': time.perf_counter()-started,
               'all_event_refiner_extra_mac': 2*len(arrays['t'])*4480,
               'labels_used_by_feature_inference': False}
    save_bundle(stem, data, receipt)
    print({'stage': 'cache', 'role': role, 'sequence': name, 'events': len(arrays['t']),
           'seconds': round(receipt['seconds'], 2)}, flush=True)
    return data, receipt, arrays, meta


def collect_role(directory, store, role):
    parts = {k: [] for k in ('scores', 'labels', 'probabilities', 'hd')}
    receipts = []
    moments = np.zeros(6, np.float64)
    for name in store.names(role):
        data, rec, arrays, meta = cache_sequence(directory, store, role, name)
        receipts.append(rec)
        for key, value in (('scores', data['s1_scores']), ('labels', arrays['label']),
                           ('probabilities', data['s1_prob_all']),
                           ('hd', np.full(len(arrays['t']), meta['high_density'], bool))):
            parts[key].append(value)
        if role == 'train_main':
            from .core import logit
            # Each occupied event cell contributes once per event. Fit is driven by
            # actual Refiner locations, not by mostly empty sensor cells.
            packet = event_packet(arrays)
            w, x, y = data['windows'], packet['x']//8, packet['y']//8
            a = logit(data['s1_heatmaps'][w, y, x])
            b = logit(data['ann_heatmaps'][w, y, x])
            moments += [len(a), a.sum(), b.sum(), np.dot(a, a), np.dot(a, b), np.dot(b, b)]
    return {k: np.concatenate(v) for k, v in parts.items()}, receipts, moments


def aggregate_baseline(receipts):
    rows = [r['default_counts']['ann'] for r in receipts]
    total = metrics(rows)
    hd = metrics([r['default_counts']['ann'] for r in receipts if r['high_density']])
    mac = sum(r['costs']['ann']['proposal_mac'] for r in receipts)+4480*total['candidate_events']
    dense = sum(r['costs']['ann']['proposal_dense_equivalent_mac'] for r in receipts)+4480*total['candidate_events']
    return {'overall': total, 'high_density': hd, 'mac': mac, 'dense_mac': dense, 'ac': 0}
