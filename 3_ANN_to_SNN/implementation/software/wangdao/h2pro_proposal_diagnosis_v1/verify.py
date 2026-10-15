"""Recount saved predictions and replay selected gates using the frozen pipeline."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from ..h2pro_proposal_transfer_v1.pipeline import TransferPipeline
from ..h2pro_snn.diagnostics import error_counts
from .common import identity, open_store, read_json, sha256, load_bundle
from .core import context_features, event_packet


def verify(teacher, directory):
    directory = Path(directory).resolve()
    report = read_json(directory/'comparison.json')
    if report['identity'] != identity(teacher):
        raise ValueError('Verification source identity changed')
    for name, digest in report['evidence_sha256'].items():
        if sha256(directory/name) != digest:
            raise ValueError('Evidence changed: '+name)
    store = open_store(teacher)
    recounted = 0
    for role in ('train_main', 'train_dev'):
        for name in store.names(role):
            record = store.get(role, name)
            for p in (directory/'points'/role).glob('*/'+Path(name).stem+'.json'):
                output, receipt = load_bundle(p.with_suffix(''))
                assert receipt['counts'] == error_counts(record.arrays['label'], output['probabilities'], output['candidate'])
                assert receipt['refiner_mac'] == 4480*int(output['candidate'].sum())
                recounted += 1
    selected = {r['specification']['gate']: r for r in report['results']
                if not r['diagnostic_only'] and r['specification']['context'] == 's1'
                and r['specification']['context_gate'] == r['specification']['gate']
                and r['specification']['gate'] != .51}
    exact_windows, old_points = 0, 0
    for name in store.names('train_dev'):
        record = store.get('train_dev', name)
        arrays, meta = record.arrays, record.metadata
        packet = event_packet(arrays)
        data, _ = load_bundle(directory/'cache/train_dev'/Path(name).stem)
        # Compare independent all-event feature construction with preserved old
        # ANN and S1 payloads at their original candidates.
        old_root = directory.parent/'h2pro_proposal_transfer_v1_dev'
        for mode, old_mode in (('ann', 'ann_dense'), ('s1', 's1_event')):
            receipt = read_json(old_root/old_mode/(Path(name).stem+'.json'))
            payload = old_root/old_mode/(Path(name).stem+'.npz')
            assert sha256(payload) == receipt['payload_sha256']
            with np.load(payload, allow_pickle=False) as z:
                f = context_features(packet, data['windows'], data['base'], data[mode+'_heatmaps'], .51, meta['t0_ms'])
                np.testing.assert_array_equal(f[z['candidate']], z['features'])
                np.testing.assert_array_equal(data[mode+'_heatmaps'], z['heatmaps'])
                np.testing.assert_array_equal(np.where(data[mode+'_scores'] >= .51, data[mode+'_prob_all'], 0), z['probabilities'])
            old_points += 1
        for gate, r in selected.items():
            pipe = TransferPipeline('s1_event')
            pipe.proposal = replace(pipe.proposal, proposal_threshold=gate)
            pipe.reset(meta['t0_ms'])
            pipe.push(packet, meta['t0_ms']+100*meta['closed_windows'])
            windows, tail = pipe.finalize()
            assert tail == 0
            saved, _ = load_bundle(directory/'points/train_dev'/r['name']/Path(name).stem)
            np.testing.assert_array_equal(saved['candidate'], np.concatenate([w.candidate_mask for w in windows]))
            np.testing.assert_array_equal(saved['probabilities'], np.concatenate([w.probabilities for w in windows]))
            assert pipe.executor.cost.refiner_mac == 4480*int(saved['candidate'].sum())
            exact_windows += len(windows)
        print({'stage': 'verify', 'sequence': name, 'coupled_gates': len(selected)}, flush=True)
    return {'status': 'passed', 'saved_predictions_recounted': recounted,
            'original_transfer_points_exact': old_points, 'independent_coupled_replay_windows': exact_windows,
            'independent_coupled_gates': sorted(selected), 'training_started': False,
            'reserved_sequences_read': 0, 'source_and_artifact_hashes_verified': True}
