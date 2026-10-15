import hashlib
import json
import os
from dataclasses import dataclass,asdict
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
WINDOW=ROOT.parent/'H2-pro_dynamic_window_v1'
sys.path.insert(0,str(WINDOW))
from src.core import ANN_ROOT,SNN_ROOT,SequenceContext,POLICY_MAP,EVENT_DTYPE,exact_mlp,counts,metrics
from src.workflow import (sha,read,digest,manifests,load_sequence,verify_baseline,
                          preservation_snapshot as previous_snapshot,identity as previous_identity)

PRIOR=WINDOW/'results/runs/frozen_ann_v1'
FLOAT_SOURCE=ANN_ROOT/'software/weights/H2-pro-Causal-ANN-v1_float.pt'
INT8_SOURCE=ANN_ROOT/'software/weights/H2-pro-Causal-ANN-v1_int8.npz'
DEFAULT_CACHE=ROOT/'results/cache/paired_features_v1'
DEFAULT_SUITE=ROOT/'results/runs/paired_ann_s42'
ARMS={'fixed100':'fixed100','dynamic2000':'adaptive2000_rate100'}
ROLES=('train_main','train_dev')


@dataclass(frozen=True)
class Config:
    seed:int=42
    learning_rate:float=1e-5
    weight_decay:float=1e-4
    source_batch_size:int=32768
    evaluation_batch_size:int=16384

    def validate(self):
        import math
        if (type(self.seed) is not int or self.seed<0 or type(self.source_batch_size) is not int
                or self.source_batch_size<1 or type(self.evaluation_batch_size) is not int
                or self.evaluation_batch_size<1 or not math.isfinite(self.learning_rate)
                or self.learning_rate<=0 or not math.isfinite(self.weight_decay) or self.weight_decay<0):
            raise ValueError('Invalid training configuration')
        return self


def local_output(path):
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'results').resolve()):
        raise ValueError('Keep new run/cache outputs inside this independent experiment/results')
    return path


def atomic_json(path,value,immutable=False):
    path=Path(path)
    if immutable and path.exists():
        if read(path)!=value:raise ValueError(f'Existing identity differs: {path}; use a new directory')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    with temp.open('w',encoding='utf-8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def atomic_npz(path,**arrays):
    import numpy as np
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    with temp.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def verify_prior():
    if previous_identity()!=read(PRIOR/'identity.json'):
        raise ValueError('Frozen window implementation or source identity changed')
    if read(PRIOR/'audit.json')['status']!='passed':raise ValueError('Prior full-window audit required')
    selection=read(PRIOR/'selection.json')
    if 'adaptive2000_rate100' not in selection['selection']['dev_policies']:
        raise ValueError('Expected preselected dynamic2000 rate100 policy')
    return {p:sha(PRIOR/p) for p in ('identity.json','selection.json','audit.json',
                                     'train_main_summary.json','train_dev_summary.json')}


def source_hashes():
    paths=sorted((ROOT/'dynamic_ann').glob('*.py'))+[ROOT/'run.py',ROOT/'train_local_4060.ps1',ROOT/'protocol.json']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest() for p in paths}


def cache_identity():
    return dict(version='h2pro_dynamic_pair_features_v1',prior=verify_prior(),policies=ARMS,
        baseline_lock_sha256=verify_baseline(),source_float_sha256=sha(FLOAT_SOURCE),
        builder_sources={name:sha(ROOT/'dynamic_ann'/name) for name in ('common.py','data.py')})


def run_identity(cache,config,engine):
    return dict(version='h2pro_dynamic_ann_refiner_train_v1',protocol=read(ROOT/'protocol.json'),
        config=asdict(config.validate()),sources=source_hashes(),cache_manifest_sha256=sha(Path(cache)/'manifest.json'),
        source_float_sha256=sha(FLOAT_SOURCE),source_int8_sha256=sha(INT8_SOURCE),
        frozen_baseline_sha256=verify_baseline(),prior=verify_prior(),engine=engine)


def preserve_before():
    path=ROOT/'results/checks/preservation_before.json'
    if path.exists():return read(path)
    result=previous_snapshot()
    for p in WINDOW.rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:
            s=p.stat();result[str(p)]=dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(p))
    atomic_json(path,result,immutable=True)
    return result


def preserve_check():
    before=preserve_before();changed=[]
    for key,value in before.items():
        p=Path(key)
        if not p.is_file():changed.append(key);continue
        s=p.stat()
        if (s.st_size!=value['bytes'] or s.st_mtime_ns!=value['mtime_ns']
                or ('sha256' in value and sha(p)!=value['sha256'])):changed.append(key)
    result=dict(original_files=len(before),content_hashed=sum('sha256' in v for v in before.values()),
                changed_or_missing=changed,readme_excluded=True)
    atomic_json(ROOT/'results/checks/preservation_after.json',result)
    if changed:raise ValueError(f'Original artifacts changed during preparation: {changed[:5]}')
    return result
