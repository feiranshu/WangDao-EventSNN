"""Frozen source identity and predeclared Main-only exit calibration design."""
from pathlib import Path
import hashlib
import os
import numpy as np

from ..h2pro_direct_v1.common import lock_directory, output_path
from ..h2pro_r1_train_v1.common import IMPLEMENTATION, DEFAULT_CACHE, Config, run_identity
from ..h2pro_r1_train_v1.evaluation import load_export
from ..h2pro_snn.foundation import read_json, sha256
from ..h2pro_snn.stream_data import atomic_json
from ..h2pro_longtrain_v1.storage import load

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = IMPLEMENTATION/'results/runs/h2pro_adaptive_t_v1_s42_ep159'
SOURCE = IMPLEMENTATION/'results/runs/h2pro_r1t16_train_v1_s42'
SOURCE_SHA = '3e3ef613c71d7eb8dfd55b1e3640db2fb09126d13b0e6a10e9bc3c3db36c98d9'
DESIGN = dict(version='h2pro_adaptive_t_v1',source_epoch=159,steps=[4,8,16],theta=255,m0=191,
    final_threshold=.35,event_window_ms=100,weights_updated=False,training_started=False,
    proposal='original frozen ANN',features='original frozen 36 columns and candidate mask',
    hidden1_projection='computed once',transport='binary IF; preserve membrane and weighted accumulator',
    readout='original 32->4 recomputed at every visited checkpoint; divide accumulated current by elapsed T',
    negative_grid=[None,.0001,.001,.003,.01,.03,.05,.1,.2,.3],
    positive_grid=[None,.9999,.999,.99,.97,.95,.9,.8,.6,.4],modes=['confidence','stable8'],
    stability='stable8 cannot exit at4; at8 require class agreement with4 and confidence',
    split='within Main only, separately sort ordinary/HD by SHA256(seed42:sequence), every fourth validates',
    objectives={'accuracy':0.,'balanced':.0025,'efficiency':.01},
    objective_formula='IoU - lambda*(AC / fixedT16 AC) on the same subset; no hard acceptance/stop threshold',
    shortlist='union of top8 per objective on Main-fit; identical objectives select on Main-validation',
    tie_break='IoU, HD recall, lower AC, lower MAC; deterministic grid order',
    main_validation_scope='held out for exit-parameter selection only; source weights used all Main during training',
    no_runtime_inputs=['labels','teacher predictions','future prefix logits','Dev-selected thresholds','HD ground-truth tags'],
    full_event_scoring=True,roles=['train_main','train_dev'],reserved_splits_read=False,
    calibration_backend='dense count reference for all prefixes',
    verification_backend='full Dev selected policies execute actual incremental CPU spike-add with early stop',
    cost='include repeated readout MAC; AC, scalar, IF state and control/traffic limitations separate')


def sources():
    paths = list(PACKAGE.glob('*.py'))+[IMPLEMENTATION/'software/scripts/evaluate_h2pro_adaptive_t.py',
                                         IMPLEMENTATION/'evaluate_h2pro_adaptive_t.ps1']
    return {p.relative_to(IMPLEMENTATION).as_posix():hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
            for p in sorted(paths)}


def identity(cache):
    old = read_json(SOURCE/'identity.json')
    if run_identity(cache,Config(**old['config']),old['engine']) != old:
        raise ValueError('Source model/baseline/cache identity changed')
    cp_path = SOURCE/'epochs/epoch_159.pt'
    if sha256(cp_path) != SOURCE_SHA:
        raise ValueError('Require the immutable original epoch159 checkpoint')
    cp=load(cp_path); layers=load_export(SOURCE/'epochs/epoch_159_int8.npz')
    if cp['identity'] != old or cp['state']['global_step'] != 43089:
        raise ValueError('Source159 identity/step mismatch')
    for i,(w,s) in enumerate(layers):
        master=cp['model'][f'weights.{i}'].numpy()
        if not (np.array_equal(np.clip(np.rint(master.astype(np.float64)/s.astype(np.float64)),-128,127).astype(np.int8),w)
                and np.array_equal(cp['model'][f'scale{i}'].numpy(),s)):
            raise ValueError('Source159 master and export differ')
    result=dict(design=DESIGN,source_identity=old,sources=sources(),cache=str(Path(cache).resolve()),
                source_files={str((SOURCE/name).relative_to(IMPLEMENTATION)).replace('\\','/'):sha256(SOURCE/name)
                    for name in ('epochs/epoch_159.pt','epochs/epoch_159.json','epochs/epoch_159_int8.npz','ready.json')})
    return result,layers


def atomic_npz(path, **arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def sealed_json(path,value):
    if path.exists():
        if read_json(path)!=value:
            raise ValueError('Refusing to change sealed result: '+str(path))
    else:
        atomic_json(path,value)


def split_main(rows):
    if any(r['role']!='train_main' for r in rows):
        raise ValueError('Exit calibration uses Main only')
    fit,validation=[],[]
    for high in (False,True):
        group=sorted((r for r in rows if r['high_density']==high),
                     key=lambda r:hashlib.sha256(('42:'+r['sequence']).encode()).hexdigest())
        for i,r in enumerate(group):
            (validation if i%4==0 else fit).append(r['sequence'])
    if not fit or not validation:
        raise ValueError('Nonempty sequence-disjoint Main fit/validation required')
    return dict(fit=fit,validation=validation)
