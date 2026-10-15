"""Identities and deterministic Train-main-only calibration selection."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np

from ..h2pro_snn.foundation import IMPLEMENTATION, read_json, role_records, sha256, verify_baseline
from ..h2pro_direct_v1.common import lock_directory, atomic_json

PACKAGE = Path(__file__).absolute().parent
VERSION = 'h2pro_short_if_calibration_v1'


def protocol():
    return read_json(PACKAGE/'protocol.json')


def source_hashes(cache_only=False):
    paths = [PACKAGE/'common.py', PACKAGE/'cache_worker.py', PACKAGE/'protocol.json']
    if not cache_only:
        paths += list(PACKAGE.glob('*.py'))
        paths += [IMPLEMENTATION/'software/scripts/h2pro_calibrate.py', IMPLEMENTATION/'calibrate_h2pro.ps1']
    # Include the immutable dependencies actually consumed by this version.
    paths += list((PACKAGE.parent/'h2pro_direct_v1').glob('*.py'))
    paths += [PACKAGE.parent/'h2pro_direct_v1/protocol.json']
    paths += [PACKAGE.parent/'h2pro_snn'/n for n in ('foundation.py','diagnostics.py','stream_data.py')]
    return {p.relative_to(IMPLEMENTATION).as_posix():
            hashlib.sha256(p.read_text('utf-8').encode('utf-8')).hexdigest() for p in sorted(set(paths))}


def identity(kind, cache_only=False):
    return {'version':VERSION, 'kind':kind, 'training_started':False, 'weights_updated':False,
            'baseline_lock_sha256':verify_baseline()['baseline_lock_sha256'],
            'protocol_sha256':sha256(PACKAGE/'protocol.json'),
            'sources_utf8_lf':source_hashes(cache_only)}


def select_records(teacher_manifest):
    """Use only pre-existing Train-main metadata, never labels or Dev scores."""
    settings=protocol()
    records={r['path']:r for r in role_records('train_main')}
    rows=teacher_manifest['roles']['train_main']
    if len(rows)!=len(records) or {r['sequence'] for r in rows}!=set(records):
        raise ValueError('Require the complete frozen Train-main teacher manifest')
    rng=np.random.default_rng(settings['sample_seed'])
    chosen=[]
    for dense,key in ((False,'ordinary_sequences'),(True,'high_density_sequences')):
        group=sorted((r for r in rows if bool(r['high_density'])==dense),
                     key=lambda r:(r['candidate_events'],r['sequence']))
        count=settings[key]
        if len(group)<count:
            raise ValueError('Not enough sequences for the frozen density-stratified sample')
        for indices in np.array_split(np.arange(len(group)),count):
            chosen.append(group[int(rng.choice(indices))]['sequence'])
    return [records[name] for name in sorted(chosen)]


def calibration_manifest(directory):
    directory=Path(directory)
    m=read_json(directory/'manifest.json')
    expected=identity('train_main_original_features',cache_only=True)
    if any(m.get(k)!=v for k,v in expected.items()) or m.get('role')!='train_main' or m.get('status')!='complete':
        raise ValueError('Require this version\'s completed Train-main calibration cache')
    allowed={r['path']:r for r in role_records('train_main')}
    rows=m['sequences']
    if [r['sequence'] for r in rows]!=m['selected_sequences'] or len({r['sequence'] for r in rows})!=len(rows):
        raise ValueError('Incomplete or duplicated calibration sequences')
    for row in rows:
        name=row['sequence']; filename=row['cache_file']
        if (name not in allowed or Path(filename).name!=filename or
                row['source_sha256']!=allowed[name]['sha256'] or sha256(directory/filename)!=row['cache_sha256']):
            raise ValueError('Invalid Train-main calibration source or cache hash')
    return m
