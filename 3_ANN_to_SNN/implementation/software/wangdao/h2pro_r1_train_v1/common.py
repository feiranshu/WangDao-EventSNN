"""Independent r1 identity; conversion parameters are not training options."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

from ..h2pro_calibration_v1.core import Encoder
from ..h2pro_distill_v1.common import (
    Config, CACHE_VERSION, DEFAULT_CACHE, DEFAULT_TEACHER, DEFAULT_EVIDENCE,
    IMPLEMENTATION, atomic_json, lock_directory, output_path, read_json, sha256,
    verify_baseline, source_hashes as shared_hashes,
)

PACKAGE = Path(__file__).resolve().parent
VERSION = 'h2pro_r1t16_weight_train_v1'
DEFAULT_RUN = IMPLEMENTATION/'results/runs/h2pro_r1t16_train_v1_s42'
ANCHOR_MAIN = IMPLEMENTATION/'results/runs/h2pro_task_calibration_v1_fit/calibration.json'
ANCHOR_DEV = IMPLEMENTATION/'results/runs/h2pro_task_calibration_v1_dev/r1_t16_task_rescore.json'
STAGE = 'r1'
STEPS = 16
THRESHOLD = .35
ENCODERS = (Encoder(255, 191), Encoder(255, 127))

PROTOCOL = {
    'version': VERSION, 'exploration': 'A', 'parent_directions': [1, 4],
    'anchor': 'original r1/T16/task_rescore, not the later single/group m0 candidates',
    'dimensions': [36, 64, 32, 4], 'stage': STAGE, 'steps': STEPS,
    'encoders': [asdict(e) for e in ENCODERS], 'spiking_transport_layers': [1],
    'proposal': 'frozen exact ANN; routing, all original candidates and C1 features unchanged',
    'teacher': 'frozen H2-pro exact-int8; original cached four-prototype logits',
    'student': 'all three weight matrices initialized from original frozen H2-pro; fresh Adam',
    'weight_update': 'FP32 master, int8 round/clamp STE, fixed original per-output scales',
    'activation': 'unchanged clip [0,1] then round-to-even uint8',
    'if': 'hidden1 only: binary soft reset, theta255/m0=191/T16, per-event reset; hidden2 ANN',
    'gradient': 'straight-through round and hidden1 IF count, derivative 16/255',
    'readout': 'unchanged 32->4 ANN, max of two prototypes per class, probability threshold 0.35',
    'loss': '(1-alpha)*unweighted CE + alpha*temperature^2*KL(teacher||student)',
    'sampling': 'every Main candidate exactly once per epoch, uniform deterministic shuffle',
    'optimizer': 'Adam, constant learning rate, weight_decay=0, foreach=False',
    'hard_mining': False, 'spike_regularization': False, 'gradient_clipping': False,
    'amplitude_stop': False, 'process_memory_cap': False, 'early_stopping': False,
    'roles': ['train_main', 'train_dev'], 'reserved_splits_read': False,
    'best_selection': 'Dev full-event IoU at fixed 0.35, tie by recall; epoch zero eligible',
    'retention': 'last/best and every epoch preserve weights, Adam, RNG, cursor and metrics',
    'training_backend': 'dense FP64 integer-count reference; FP32 logits/masters/Adam',
    'inference_backend': 'existing calibrated explicit binary spike-add compatible',
    'cost': 'r1 MAC=Proposal+N*(2304+128), AC=32*N1; dense GPU MAC separately reported',
    'timing': 'T16 is coding steps, original 100ms closed-prefix window unchanged',
    'numerical_checks': 'nonfinite and identity/integrity only; original quantizers retained',
}


def anchor_identity():
    prior = read_json(ANCHOR_MAIN)
    if (prior['data_role'] != 'train_main' or prior['dev_used_for_selection']
            or prior['weights_updated']):
        raise ValueError('Require the original frozen Main-only calibration')
    point = next(p for p in prior['points'] if p['stage'] == STAGE and p['steps'] == STEPS)
    expected = [asdict(e) for e in ENCODERS]
    old = read_json(ANCHOR_DEV)
    if (point['task_rescore']['encoders'] != expected or old['encoders'] != expected
            or old['stage'] != STAGE or old['steps'] != STEPS
            or old['mode'] != 'task_rescore' or old['threshold'] != THRESHOLD):
        raise ValueError('Original r1/T16/task_rescore conversion identity changed')
    return {'main_selection_sha256': sha256(ANCHOR_MAIN),
            'original_dev_result_sha256': sha256(ANCHOR_DEV), 'encoders': expected}


def source_hashes():
    result = shared_hashes()
    paths = list(PACKAGE.glob('*.py')) + [
        PACKAGE.parent/'h2pro_calibration_v1/core.py',
        IMPLEMENTATION/'software/scripts/train_h2pro_r1.py',
        IMPLEMENTATION/'train_local_4060_r1.ps1',
    ]
    result.update({p.relative_to(IMPLEMENTATION).as_posix():
                   hashlib.sha256(p.read_text('utf-8').encode('utf-8')).hexdigest()
                   for p in sorted(paths)})
    return result


def run_identity(cache, config, engine):
    return {'version': VERSION, 'protocol': PROTOCOL, 'config': asdict(config.validate()),
            'baseline': verify_baseline(), 'anchor': anchor_identity(),
            'cache_manifest_sha256': sha256(Path(cache)/'manifest.json'),
            'sources': source_hashes(), 'engine': engine}
