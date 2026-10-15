"""Independent identity; historical source checkpoints are read-only inputs."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..h2pro_distill_v1.common import (Config, DEFAULT_CACHE, DEFAULT_RUN as SOURCE_RUN,
    IMPLEMENTATION, atomic_json, lock_directory, output_path, read_json, run_identity, sha256)
from ..h2pro_distill_v1.runtime import engine
from ..h2pro_longtrain_v1.storage import load

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = IMPLEMENTATION/'results/runs/h2pro_hardcase_v1_s42'
DEFAULT_DIAGNOSIS = IMPLEMENTATION/'results/runs/h2pro_fixed_diagnosis_v1'
ARMS = ('control', 'hard_weighted')
SOURCE_EPOCH = 88
MILESTONES = (5, 10, 20)
HARD_WEIGHT = 2.
PROTOCOL = {
    'version': 'h2pro_hardcase_v1', 'direction': 5, 'source_epoch': SOURCE_EPOCH,
    'arms': list(ARMS), 'planned_additional_epochs_per_arm': 20, 'milestones': list(MILESTONES),
    'initialization': 'identical FP32 master weights, original fixed scales, Adam moments/steps and RNG from epoch88 best.pt',
    'control': 'unchanged direction4 update/loss implementation',
    'hard_selection': 'Main only; frozen ANN correct AND frozen epoch88 student wrong at probability >= FP32(0.35)',
    'hard_positive': 'label=1, teacher positive, source student negative',
    'hard_negative': 'label=0, teacher negative, source student positive',
    'hard_weight': HARD_WEIGHT, 'ordinary_weight': 1., 'hard_mask_refresh': 'never; fixed before training',
    'weighted_loss': 'sum(w*((1-alpha)*CE + alpha*Tdistill^2*KL(teacher||student)))/sum(w) per batch',
    'sampling': 'all Main candidates once per epoch; same seed42 permutation using absolute epoch88+added_epoch',
    'unchanged': ['36-64-32-4 structure', 'T8 theta255 m0=127', 'teacher', 'ANN Proposal',
                  'features', 'scales', 'STE', 'lr1e-4', 'batch8192', 'alpha0.5', 'temperature2', 'Adam settings'],
    'ac_regularization': False, 'hd_resampling': False, 'density_specific_loss_weight': False,
    'best_selection': 'Dev IoU at 0.35, tie by recall; source epoch88 eligible',
    'milestone_calibration': 'Main-only main_iou and main_recall_matched policies from fixed_diagnosis_v1, sealed before that milestone Dev inference',
    'cost': 'full-event inference-equivalent MAC/AC separate from dense GPU training and scalar/state operations',
    'numerics': 'nonfinite/integrity checks only; no finite-activation stopping threshold',
    'roles': ['train_main', 'train_dev'], 'reserved_splits_read': False,
}


def digest(value):
    """Serialization-independent, exact content digest including Adam and RNG."""
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v):
            a = v.detach().cpu().numpy()
            h.update(b'tensor'); h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        elif isinstance(v, np.ndarray):
            h.update(b'array'); h.update(str(v.dtype).encode()); h.update(str(v.shape).encode()); h.update(v.tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v, key=lambda x: (type(x).__name__, str(x))):
                visit(k); visit(v[k])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for item in v:
                visit(item)
            h.update(b'end')
        else:
            h.update(type(v).__name__.encode()); h.update(json.dumps(v, allow_nan=False).encode()); h.update(b';')
    visit(value)
    return h.hexdigest()


def sources():
    paths = list(PACKAGE.glob('*.py')) + [IMPLEMENTATION/'software/scripts/train_h2pro_hardcase.py',
             IMPLEMENTATION/'train_local_4060_hardcase.ps1',
             PACKAGE.parent/'h2pro_fixed_diagnosis_v1/thresholds.py']
    return {p.relative_to(IMPLEMENTATION).as_posix(): hashlib.sha256(p.read_text('utf-8').encode()).hexdigest()
            for p in sorted(paths)}


def inputs(cache, source, diagnosis, device):
    cache, source, diagnosis = map(output_path, (cache, source, diagnosis))
    saved = load(source/'best.pt')
    old_identity = read_json(source/'identity.json')
    config = Config(**old_identity['config']).validate()
    if config != Config() or run_identity(cache, config, engine(device)) != old_identity:
        raise ValueError('Original configuration, source, teacher, cache or execution engine changed')
    state = saved['state']
    if (saved.get('resumable') is not True or saved['identity'] != old_identity
            or state['cursor'] != {'epoch': 89, 'batch': 0, 'phase': 'train'}
            or len(state['history']) != SOURCE_EPOCH or state['best_epoch'] != SOURCE_EPOCH
            or state['global_step'] != 23848
            or any(float(s['step']) != state['global_step'] for s in saved['optimizer']['state'].values())):
        raise ValueError('Need the original completed epoch88 master-weight/Adam checkpoint')
    diag_identity = read_json(diagnosis/'identity.json')
    if diag_identity['training_identity'] != old_identity:
        raise ValueError('Diagnosis came from another teacher/training run')
    for name in ('best.pt', 'training.json', 'epochs/epoch_088_int8.npz'):
        if sha256(source/name) != diag_identity['source_run_files_sha256'][name]:
            raise ValueError('Original epoch88 evidence has changed')
    main_seal = read_json(diagnosis/'main_checksums.json')
    selected_names = ('train_main_ann_predictions.npz', 'train_main_epoch_088_predictions.npz')
    for name in selected_names:
        if sha256(diagnosis/name) != main_seal[name]:
            raise ValueError('Frozen Main predictions changed')
    identity = {'protocol': PROTOCOL, 'config': old_identity['config'], 'source_identity': old_identity,
        'source_checkpoint_sha256': sha256(source/'best.pt'), 'sources': sources(), 'engine': engine(device),
        'source_model_digest': digest(saved['model']), 'source_optimizer_digest': digest(saved['optimizer']),
        'source_rng_digest': digest(saved['rng']), 'diagnosis_predictions': {n: main_seal[n] for n in selected_names},
        'paths': {'cache': str(cache), 'source': str(source), 'diagnosis': str(diagnosis)}}
    return identity, saved, config


def independent_output(directory, *input_paths):
    directory = output_path(directory)
    for path in input_paths:
        path = Path(path).resolve()
        if directory == path or directory in path.parents or path in directory.parents:
            raise ValueError('Use a new independent suite directory, outside all source runs/caches')
    return directory


def arm_identity(identity, arm, mask_sha256):
    if arm not in ARMS:
        raise ValueError('Unknown paired arm')
    return {'suite': identity, 'arm': arm, 'main_hard_categories_sha256': mask_sha256}


def save_arrays(path, **arrays):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)
