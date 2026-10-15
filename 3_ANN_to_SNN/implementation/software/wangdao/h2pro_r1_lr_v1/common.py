"""Versioned experiment identity; no changes to the existing r1 model or trainer."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..h2pro_r1_train_v1.common import (
    Config, DEFAULT_CACHE, DEFAULT_RUN as SOURCE_RUN, IMPLEMENTATION, atomic_json,
    lock_directory, output_path, read_json, run_identity, sha256,
)
from ..h2pro_r1_train_v1.runtime import engine, fresh_totals
from ..h2pro_longtrain_v1.storage import load

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = IMPLEMENTATION / 'results/runs/h2pro_r1_lr_v1_s42_ep159'
SOURCE_EPOCH = 159
ARMS = ('control', 'low_lr')
LEARNING_RATES = {'control': 1e-4, 'low_lr': 3e-5}
MILESTONES = (10, 25, 50)
PROTOCOL = {
    'version': 'h2pro_r1_lr_v1', 'source_epoch': SOURCE_EPOCH,
    'source_checkpoint': 'epochs/epoch_159.pt', 'arms': LEARNING_RATES,
    'planned_additional_epochs_per_arm': 50, 'milestones': list(MILESTONES),
    'initialization': 'same FP32 masters, frozen scales, Adam moments/steps and RNG; only optimizer lr differs',
    'order': 'same deterministic seed42 shuffle at absolute epoch159+added_epoch',
    'unchanged': ['original r1/T16 model', 'features', 'ANN Proposal', 'H2-pro teacher',
                  'theta255', 'm0=191', 'T16', 'fixed scales', 'quantization/STE',
                  'batch8192', 'CE/KD alpha0.5 temperature2', 'threshold0.35', 'Adam settings except lr'],
    'ac_regularization': False, 'hard_example_weighting': False, 'scheduler': False,
    'early_stopping': False, 'finite_amplitude_stop': False, 'memory_cap': False,
    'milestone_comparison': 'fixed 0.35 endpoint, best-so-far and window trends; no threshold recalibration',
    'best_selection': 'Dev full-event IoU, tie by recall; source epoch159 eligible as added epoch0',
    'retention': 'every epoch saves complete master/Adam/RNG checkpoint and int8 export',
    'cost': 'same r1 inference-equivalent MAC/AC, scalar/state and dense GPU reference counts',
    'roles': ['train_main', 'train_dev'], 'reserved_splits_read': False,
}


def digest(value):
    """Exact content hash independent of torch serialization storage identities."""
    h = hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v) or isinstance(v, np.ndarray):
            a = v.detach().cpu().numpy() if torch.is_tensor(v) else v
            h.update(b'tensor' if torch.is_tensor(v) else b'array')
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
        elif isinstance(v, dict):
            h.update(b'dict')
            for key in sorted(v, key=lambda k: (type(k).__name__, str(k))):
                visit(key); visit(v[key])
        elif isinstance(v, (tuple, list)):
            h.update(type(v).__name__.encode())
            for item in v:
                visit(item)
            h.update(b'end')
        else:
            h.update(type(v).__name__.encode())
            h.update(json.dumps(v, allow_nan=False).encode()); h.update(b';')
    visit(value)
    return h.hexdigest()


def sources():
    paths = list(PACKAGE.glob('*.py')) + [
        IMPLEMENTATION / 'software/scripts/train_h2pro_r1_lr.py',
        IMPLEMENTATION / 'train_local_4060_r1_lr.ps1',
    ]
    return {p.relative_to(IMPLEMENTATION).as_posix():
            hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest() for p in sorted(paths)}


def inputs(cache, source, device):
    cache, source = map(output_path, (cache, source))
    old_id = read_json(source / 'identity.json')
    config = Config(**old_id['config']).validate()
    if config != Config() or run_identity(cache, config, engine(device)) != old_id:
        raise ValueError('Source model/config/cache/engine identity has changed')
    path = source / 'epochs/epoch_159.pt'
    origin = load(path)
    state = origin['state']
    historical = read_json(source / 'epochs/epoch_159.json')
    if (origin.get('resumable') is not True or origin['identity'] != old_id
            or state['cursor'] != dict(epoch=160, batch=0, phase='train')
            or len(state['history']) != SOURCE_EPOCH or state['history'][-1] != historical
            or state['global_step'] != 43089 or historical['global_step'] != 43089
            or len(origin['optimizer']['state']) != 3
            or any(float(s['step']) != 43089 for s in origin['optimizer']['state'].values())
            or any(g['lr'] != 1e-4 for g in origin['optimizer']['param_groups'])):
        raise ValueError('Require the complete original epoch159 weights/Adam/RNG checkpoint')
    old_ready = read_json(source / 'ready.json')
    if old_ready['identity'] != old_id:
        raise ValueError('Source preparation identity mismatch')
    identity = {
        'protocol': PROTOCOL, 'config': asdict(config), 'source_identity': old_id,
        'source_files_sha256': {name: sha256(source / name) for name in
            ('epochs/epoch_159.pt', 'epochs/epoch_159.json', 'epochs/epoch_159_int8.npz', 'ready.json')},
        'source_model_digest': digest(origin['model']),
        'source_optimizer_digest': digest(origin['optimizer']), 'source_rng_digest': digest(origin['rng']),
        'sources': sources(), 'engine': engine(device),
        'paths': {'cache': str(cache), 'source': str(source)},
    }
    return identity, origin, config, old_ready


def independent_output(directory, *input_paths):
    directory = output_path(directory)
    for path in input_paths:
        path = Path(path).resolve()
        if directory == path or directory in path.parents or path in directory.parents:
            raise ValueError('Use an independent suite outside source runs and caches')
    return directory


def arm_config(config, arm):
    if arm not in ARMS:
        raise ValueError('Unknown learning-rate arm')
    return replace(config, learning_rate=LEARNING_RATES[arm]).validate()


def arm_identity(identity, arm):
    return {'suite': identity, 'arm': arm,
            'config': asdict(arm_config(Config(**identity['config']), arm))}


def initialized_payload(origin, identity, baseline):
    optimizer = deepcopy(origin['optimizer'])
    for group in optimizer['param_groups']:
        group['lr'] = identity['config']['learning_rate']
    return {'identity': identity, 'model': deepcopy(origin['model']), 'optimizer': optimizer,
            'rng': deepcopy(origin['rng']), 'resumable': True,
            'state': {'cursor': dict(epoch=1, batch=0, phase='train'),
                      'global_step': origin['state']['global_step'],
                      'origin_global_step': origin['state']['global_step'], 'source_epoch': SOURCE_EPOCH,
                      'totals': fresh_totals(), 'history': [], 'best_epoch': 0,
                      'best_score': [baseline['overall']['iou'], baseline['overall']['recall']]}}


def verify_prepared(suite, identity):
    ready = read_json(suite / 'ready.json')
    if ready.get('identity') != identity or ready.get('status') != 'passed':
        raise ValueError('Run the matching preflight check first')
    for name, expected in ready['immutable_files_sha256'].items():
        if sha256(suite / name) != expected:
            raise ValueError('Immutable initialization changed: ' + name)
    return ready
