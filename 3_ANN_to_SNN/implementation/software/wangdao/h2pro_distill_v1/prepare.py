"""Prepare unchanged ANN candidates/features once, in the frozen ANN namespace."""
from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np

from ..h2pro_direct_v1.core import forward, load_layers, probabilities
from ..h2pro_direct_v1.workflow import cache_manifest, load_sequence
from ..h2pro_proposal_diagnosis_v1.common import identity as diagnosis_identity, load_bundle
from ..h2pro_proposal_diagnosis_v1.core import context_features, event_packet
from ..h2pro_snn.foundation import role_records
from ..h2pro_snn.stream_data import SequenceStore
from .common import (CACHE_VERSION, IMPLEMENTATION, atomic_json, lock_directory,
                     read_json, sha256, verify_baseline)


def save_arrays(path, arrays):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def prepare(teacher, evidence, output):
    teacher, evidence = Path(teacher), Path(evidence)
    store = SequenceStore(teacher, cache_mib=0)
    if not store.manifest['complete_development_roles'] or store.manifest['max_windows'] is not None:
        raise ValueError('Require the complete frozen 79/10 teacher cache')
    if read_json(evidence/'identity.json') != diagnosis_identity(teacher):
        raise ValueError('The existing frozen-feature evidence identity changed')
    baseline = verify_baseline()
    identity = {'version': CACHE_VERSION, 'baseline_lock_sha256': baseline['baseline_lock_sha256'],
                'teacher_manifest_sha256': sha256(teacher/'manifest.json'),
                'evidence_identity_sha256': sha256(evidence/'identity.json'),
                'prepare_source_sha256': sha256(Path(__file__)),
                'roles': ['train_main', 'train_dev'], 'training_started': False}
    direct = IMPLEMENTATION/'results/runs/h2pro_direct_v1_features'
    direct_rows = {r['sequence']: r for r in cache_manifest(direct)['sequences']}
    layers = load_layers()
    started = time.perf_counter()
    with lock_directory(output, identity) as directory:
        if (directory/'manifest.json').exists():
            from .data import manifest as validate_manifest
            return validate_manifest(directory)
        roles = {}
        for role in ('train_main', 'train_dev'):
            rows = []
            for record in role_records(role):
                name = record['path']
                target = directory/(role+'_'+Path(name).name)
                receipt = target.with_suffix('.json')
                if receipt.exists():
                    row = read_json(receipt)
                    if (row['source_sha256'] != record['sha256'] or row['role'] != role
                            or row['sequence'] != name or sha256(target) != row['cache_sha256']):
                        raise ValueError('Partial prepared cache changed')
                    rows.append(row)
                    continue
                original = store.get(role, name)
                arrays, meta = original.arrays, original.metadata
                cached, rec = load_bundle(evidence/'cache'/role/Path(name).stem)
                if (rec['teacher_cache_sha256'] != store.entries[name][1]['cache_sha256']
                        or not rec['ann_teacher_exact'] or rec['sequence'] != name):
                    raise ValueError('Wrong source evidence')
                mask = arrays['candidate']
                all_features = context_features(event_packet(arrays), cached['windows'], cached['base'],
                                                cached['ann_heatmaps'], .51, meta['t0_ms'])
                features = all_features[mask].copy()
                raw, _ = forward(features, layers)
                np.testing.assert_array_equal(probabilities(raw), arrays['teacher'][mask])
                np.testing.assert_array_equal(mask, cached['ann_scores'] >= .51)
                if role == 'train_dev':
                    prior = load_sequence(direct, direct_rows[name])
                    for key, value in [('features', features), ('logits', raw),
                                       ('candidate', mask), ('label', arrays['label'])]:
                        np.testing.assert_array_equal(value, prior[key])
                save_arrays(target, {'features': features, 'teacher': raw,
                                     'labels': arrays['label'], 'candidate': mask})
                row = {'sequence': name, 'role': role, 'source_sha256': record['sha256'],
                       'cache_file': target.name, 'cache_sha256': sha256(target),
                       'events': len(mask), 'target_events': int(arrays['label'].sum()),
                       'candidate_events': int(mask.sum()),
                       'candidate_targets': int(arrays['label'][mask].sum()),
                       'closed_windows': meta['closed_windows'], 'trailing_events': meta['trailing_events'],
                       'high_density': meta['high_density'],
                       'proposal_sparse_mac': rec['costs']['ann']['proposal_mac'],
                       'proposal_dense_mac': rec['costs']['ann']['proposal_dense_equivalent_mac'],
                       'teacher_cache_sha256': rec['teacher_cache_sha256'],
                       'evidence_sha256': rec['payload_sha256'], 'teacher_exact': True,
                       'direct_features_exact': role == 'train_dev'}
                atomic_json(receipt, row)
                rows.append(row)
                print({'stage': 'prepare', 'role': role, 'completed': len(rows),
                       'sequence': name, 'seconds': round(time.perf_counter()-started, 1)}, flush=True)
            roles[role] = rows
        manifest = {'version': CACHE_VERSION, 'identity': identity, 'status': 'complete',
                    'roles': roles, 'training_started': False, 'reserved_sequences_read': 0,
                    'input_is_unchanged_frozen_ann_features': True}
        atomic_json(directory/'manifest.json', manifest)
        return manifest
