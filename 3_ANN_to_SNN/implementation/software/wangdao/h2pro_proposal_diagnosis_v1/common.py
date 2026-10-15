"""Independent, resumable evidence; main-only calibration and development checks."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from ..h2pro_proposal_transfer_v1.common import lineage, source_hashes, read_json, sha256
from ..h2pro_direct_v1.common import lock_directory
from ..h2pro_snn.foundation import IMPLEMENTATION
from ..h2pro_snn.stream_data import SequenceStore, atomic_json
from ..h2pro_proposal_transfer_v1.workflow import write_npz

PACKAGE = Path(__file__).absolute().parent
VERSION = 'h2pro_proposal_diagnosis_v1'
GATES = [0., .35, .40, .45, .48, .49, .50, .51, .52, .55, .60, 1.]
PROTOCOL = {
    'version': VERSION, 'direction': 7, 'training_started': False,
    'weights_updated': False, 'reserved_sequences_read': 0,
    'refiner': 'unchanged H2-pro exact-int8, event threshold 0.35',
    'proposal': 'unchanged pretrained S1, T4 real 25ms bins, per-window reset',
    'selection_role': 'all 79 train_main sequences; freeze before train_dev analysis',
    'evaluation_role': 'all 10 train_dev sequences; development evidence, not held-out Test',
    'gate_grid': GATES,
    'recovery_rule': 'highest S1 gate meeting ANN overall AND HD Proposal recall on train_main',
    'exact_gate_curve': 'every distinct event dilation score, S1 context fixed at 0.51',
    'coupled_gate': 'route AND component-size feature use selected gate; other context remains S1',
    'factorial': 'ANN/S1 routing crossed with ANN/S1 complete context at gate 0.51',
    'context_ablation': 'motion, current, history or component changed individually on ANN routes',
    'calibration': 'positive affine logit map fitted to paired ANN/S1 heatmaps on train_main only',
    'calibration_scope': 'context-only and joint context/routing; no neural parameter updates',
    'success_accuracy': 'IoU, precision, overall recall and HD recall each >= frozen ANN',
    'success_cost': 'MAC less than exact zero-skip ANN; AC reported separately, no energy claim',
    'counterfactuals': 'ANN context/S1 routing or ANN routing/S1 context need both Proposals; diagnostic only',
    'cost': 'Proposal sparse kernel counts plus 4480 MAC per selected Refiner event; sweep overhead separate',
    'scope_limit': 'feasibility of tested frozen-weight gate/context calibrations, not all possible Proposal SNNs',
}


def identity(teacher):
    sources = source_hashes()
    paths = list(PACKAGE.glob('*.py')) + [IMPLEMENTATION/'software/scripts/h2pro_proposal_diagnose.py',
                                        IMPLEMENTATION/'diagnose_h2pro_proposal.ps1']
    for p in paths:
        sources[p.relative_to(IMPLEMENTATION).as_posix()] = hashlib.sha256(p.read_text('utf-8').encode('utf-8')).hexdigest()
    return {'protocol': PROTOCOL, 'lineage': lineage(), 'sources': sources,
            'teacher_manifest_sha256': sha256(Path(teacher)/'manifest.json')}


def open_store(path):
    store = SequenceStore(path, cache_mib=0)
    if not store.manifest['complete_development_roles'] or store.manifest['max_windows'] is not None:
        raise ValueError('Complete development cache required')
    return store


def save_bundle(stem, arrays, receipt):
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    payload = stem.with_suffix('.npz')
    write_npz(payload, arrays)
    atomic_json(stem.with_suffix('.json'), dict(receipt, payload_sha256=sha256(payload)))


def load_bundle(stem):
    stem = Path(stem)
    receipt = read_json(stem.with_suffix('.json'))
    payload = stem.with_suffix('.npz')
    if sha256(payload) != receipt['payload_sha256']:
        raise ValueError('Evidence payload integrity mismatch: ' + str(payload))
    with np.load(payload, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    return arrays, receipt
