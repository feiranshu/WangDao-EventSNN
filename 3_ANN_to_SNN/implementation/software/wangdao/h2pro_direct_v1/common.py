"""Independent artifacts, identities and restart-safe locks for direct conversion."""
from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

from ..h2pro_snn.foundation import IMPLEMENTATION, read_json, sha256, verify_baseline
from ..h2pro_snn.stream_data import atomic_json

PACKAGE = Path(__file__).absolute().parent
VERSION = 'h2pro_direct_if_rate_v1'


def sources():
    paths = sorted(PACKAGE.glob('*.py')) + [PACKAGE/'protocol.json',
        IMPLEMENTATION/'software/scripts/h2pro_convert.py', IMPLEMENTATION/'convert_h2pro.ps1']
    paths += [PACKAGE.parent/'h2pro_snn'/name for name in
              ('foundation.py','diagnostics.py','stream_data.py')]
    return {p.relative_to(IMPLEMENTATION).as_posix():
            hashlib.sha256(p.read_text('utf-8').encode('utf-8')).hexdigest() for p in paths}


def output_path(path):
    path = Path(path).resolve()
    root = (IMPLEMENTATION/'results/runs').resolve()
    if root not in path.parents:
        raise ValueError('Use a separate directory inside implementation/results/runs')
    return path


@contextlib.contextmanager
def lock_directory(directory, identity):
    directory = output_path(directory)
    marker = directory/'identity.json'
    if directory.exists() and not marker.exists() and any(directory.iterdir()):
        raise ValueError('Refusing to adopt or overwrite a pre-existing experiment directory')
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.lock').open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError('This conversion directory is already in use') from error
        if marker.exists():
            if read_json(marker) != identity:
                raise ValueError('Different conversion identity; use a new output directory')
        else:
            atomic_json(marker, identity)
        yield directory


def base_identity(kind):
    baseline = verify_baseline()
    return {'version':VERSION, 'kind':kind, 'baseline_lock_sha256':baseline['baseline_lock_sha256'],
            'protocol_sha256':sha256(PACKAGE/'protocol.json'), 'training_started':False}
