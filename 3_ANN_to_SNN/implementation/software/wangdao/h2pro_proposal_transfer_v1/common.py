"""Frozen lineage, independent artifacts, and development-only inputs."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..h2pro_snn.foundation import ANN_ROOT, IMPLEMENTATION, read_json, role_records, sha256, verify_baseline
from ..h2pro_snn.stream_data import atomic_json, validate_sequence
from ..h2pro_direct_v1.common import lock_directory

PACKAGE = Path(__file__).absolute().parent
VERSION = 'h2pro_proposal_transfer_v1'
WEIGHTS = {
    'H2_D1_seed42.npz': '21019377617e429a599497a0a6b0f541d2c05bf7bc8fee83005deae72bb578b2',
    'S0P_T4_POS15_KD50_S42.npz': 'fb5fd635ee8f4b9660f98d6c71299de9e40b9d14fa83e51b2306853b2db86952',
    'S1P_INT8_ACC24_MEM24_Q12.npz': 'f05f40359e70face33a0d513abcfacc5a1519ba2d99df73076b41193abe4ef69',
}
MODES = ('ann_dense', 'ann_sparse_proposal', 's0p_dense', 's1_event')
PROTOCOL = {
    'version': VERSION, 'direction': 7, 'role': 'train_dev', 'modes': list(MODES),
    'time_steps': 4, 'micro_ms': 25, 'window_ms': 100,
    'event_threshold': 0.35, 'proposal_threshold': 0.51,
    'refiner': 'frozen H2-pro exact-int8 36-64-32-4, dense backend for every mode',
    'features': 'rebuild candidate mask, heatmap context, motion and history from each Proposal',
    'snn_state_reset': 'every 100ms window; no cross-window membrane',
    'feature_history': 'previous events and last two own Proposal heatmaps',
    'tail': 'same frozen teacher closed prefix',
    'training_started': False, 'weights_updated': False, 'threshold_search': False,
    'cost_scope': 'executed Proposal+Refiner synaptic arithmetic; scalar/state work separately',
    'reference_cost': 'S1 dense golden verification excluded from deployment ledger and reported separately',
    'ann_sparse_control': 'same ANN weights/outputs; zero-skip Proposal MAC only; no new model',
}


def source_hashes():
    paths = sorted(PACKAGE.glob('*.py'))
    paths += [IMPLEMENTATION / 'software/scripts/h2pro_proposal_transfer.py',
              IMPLEMENTATION / 'transfer_h2pro_proposal.ps1']
    paths += [PACKAGE.parent / p for p in ('fixed.py', 'snn.py', 'h2/representation.py', 'h2pro_snn/foundation.py',
              'h2pro_snn/diagnostics.py', 'h2pro_snn/stream_data.py', 'h2pro_direct_v1/common.py')]
    return {p.relative_to(IMPLEMENTATION).as_posix():
            hashlib.sha256(p.read_text('utf-8').encode('utf-8')).hexdigest() for p in paths}


def lineage():
    import json
    baseline = verify_baseline()
    for name, expected in WEIGHTS.items():
        if sha256(IMPLEMENTATION / 'software/weights' / name) != expected:
            raise ValueError('Frozen legacy export changed: ' + name)
    with np.load(IMPLEMENTATION / 'software/weights/S0P_T4_POS15_KD50_S42.npz', allow_pickle=False) as z:
        s0 = json.loads(str(z['configuration_json'].item()))
        sm = json.loads(str(z['metadata_json'].item()))
    with np.load(IMPLEMENTATION / 'software/weights/S1P_INT8_ACC24_MEM24_Q12.npz', allow_pickle=False) as z:
        s1 = json.loads(str(z['configuration_json'].item()))
        fm = json.loads(str(z['metadata_json'].item()))
    if (s0['h2_weight_sha256'] != WEIGHTS['H2_D1_seed42.npz']
            or fm['source_s0p_weight_sha256'] != WEIGHTS['S0P_T4_POS15_KD50_S42.npz']
            or baseline['verified_files']['software/weights/H2_D1_seed42.npz'] != WEIGHTS['H2_D1_seed42.npz']
            or s0['dataset_index_sha256'] != baseline['verified_files']['data/index.json']
            or fm['dataset_index_sha256'] != s0['dataset_index_sha256']
            or sm['checkpoint_epoch'] != 20 or s0['time_steps'] != 4 or s1['time_steps'] != 4
            or sm['proposal_threshold'] != .51 or fm['proposal_threshold'] != .51
            or s0['readout'] != 'mean_membrane' or s1['readout'] != 'mean_membrane'):
        raise ValueError('Legacy Proposal and frozen H2-pro lineage/contract mismatch')
    return {'baseline': baseline, 'exports_sha256': WEIGHTS,
            'same_original_ann_proposal': True, 'legacy_s0p_trained_epochs': 20,
            's0p_model_id': sm['model_id'], 's1_model_id': fm['model_id'],
            'legacy_s1_calibration_role': fm['calibration_role'],
            'new_training_started': False, 'new_weights_updated': False,
            'refiner_is_h2pro_not_legacy_h2': True}


def teacher_manifest(directory):
    directory = Path(directory)
    m = read_json(directory / 'manifest.json')
    baseline = verify_baseline()
    if (m['status'] != 'complete' or not m['complete_development_roles']
            or m['max_windows'] is not None or m['baseline_lock_sha256'] != baseline['baseline_lock_sha256']):
        raise ValueError('Require complete frozen teacher cache')
    records = role_records('train_dev')
    rows = m['roles']['train_dev']
    if [r['sequence'] for r in rows] != [r['path'] for r in records]:
        raise ValueError('Wrong development role/order/coverage')
    for row, record in zip(rows, records):
        if (row['source_sha256'] != record['sha256'] or Path(row['cache_file']).name != row['cache_file']
                or sha256(directory / row['cache_file']) != row['cache_sha256']):
            raise ValueError('Teacher cache integrity mismatch')
    return m


def load_teacher(directory, row, manifest):
    import json
    with np.load(Path(directory) / row['cache_file'], allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in ('x', 'y', 't', 'p', 'label', 'candidate', 'teacher')}
        meta = json.loads(str(z['metadata'].item()))
    if (meta['sequence'] != row['sequence'] or meta['role'] != 'train_dev'
            or meta['source_sha256'] != row['source_sha256'] or meta['max_windows'] is not None
            or meta['baseline_lock_sha256'] != manifest['baseline_lock_sha256']
            or meta['closed_windows'] != row['closed_windows']
            or meta['published_events'] != row['published_events']
            or meta['high_density'] != row['high_density']
            or int(arrays['candidate'].sum()) != row['candidate_events']
            or int(arrays['label'][arrays['candidate']].sum()) != row['candidate_targets']):
        raise ValueError('Teacher payload identity mismatch')
    validate_sequence(arrays, meta)
    return arrays, meta


def identity(manifest_path):
    return {'protocol': PROTOCOL, 'lineage': lineage(), 'sources_utf8_lf': source_hashes(),
            'teacher_manifest_sha256': sha256(Path(manifest_path)), 'reserved_sequences_read': 0}
