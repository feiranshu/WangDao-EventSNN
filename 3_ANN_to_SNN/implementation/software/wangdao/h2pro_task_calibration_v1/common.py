"""Version and provenance checks for independent task-oriented calibration."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..h2pro_snn.foundation import IMPLEMENTATION, read_json, sha256, verify_baseline
from ..h2pro_calibration_v1.common import (
    calibration_manifest, identity as old_identity, source_hashes as old_sources,
)
from ..h2pro_direct_v1.common import atomic_json, lock_directory

PACKAGE = Path(__file__).absolute().parent
VERSION = 'h2pro_task_if_calibration_v1'


def protocol():
    return read_json(PACKAGE / 'protocol.json')


def source_hashes():
    result = old_sources()
    paths = list(PACKAGE.glob('*.py')) + [PACKAGE / 'protocol.json',
        IMPLEMENTATION / 'software/scripts/h2pro_task_calibrate.py',
        IMPLEMENTATION / 'calibrate_h2pro_task.ps1']
    for path in paths:
        result[path.relative_to(IMPLEMENTATION).as_posix()] = hashlib.sha256(
            path.read_text('utf-8').encode('utf-8')).hexdigest()
    return result


def identity(kind):
    return {'version': VERSION, 'kind': kind, 'training_started': False,
            'weights_updated': False, 'labels_used': True,
            'baseline_lock_sha256': verify_baseline()['baseline_lock_sha256'],
            'protocol_sha256': sha256(PACKAGE / 'protocol.json'),
            'sources_utf8_lf': source_hashes()}


def inputs(cache, prior_directory):
    """Only original Train-main caches and the frozen Train-main MSE search."""
    manifest = calibration_manifest(cache)
    prior = read_json(Path(prior_directory) / 'calibration.json')
    expected = old_identity('encoder_calibration')
    expected['cache_manifest_sha256'] = sha256(Path(cache) / 'manifest.json')
    if (prior.get('identity') != expected or prior.get('status') != 'complete'
            or prior.get('data_role') != 'train_main' or prior.get('labels_used') is not False
            or prior.get('dev_used_for_selection') is not False):
        raise ValueError('Require the unchanged completed Train-main MSE calibration')
    requested = {(s, t) for s in protocol()['stages'] for t in protocol()['steps']}
    if (len(prior['points']) != len(requested)
            or {(p['stage'], p['steps']) for p in prior['points']} != requested):
        raise ValueError('Incomplete previous calibration pool')
    return manifest, prior
