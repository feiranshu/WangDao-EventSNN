"""Predeclared four-threshold search, frozen source and immutable v1 evidence."""
from pathlib import Path
import hashlib
from ..h2pro_adaptive_t_v1.common import (
    IMPLEMENTATION,DEFAULT_CACHE,SOURCE,identity as prior_identity,DEFAULT_RUN as REFERENCE,
    lock_directory,output_path,read_json,sha256,atomic_json,atomic_npz,sealed_json,split_main)

PACKAGE=Path(__file__).resolve().parent
DEFAULT_RUN=IMPLEMENTATION/'results/runs/h2pro_adaptive_t_v2_s42_ep159'
NEGATIVE=[None,.0001,.001,.003,.01,.03,.05,.1,.2,.3]
POSITIVE=[None,.9999,.999,.99,.97,.95,.9,.8,.6,.4]
DESIGN=dict(version='h2pro_adaptive_t_v2',source_epoch=159,steps=[4,8,16],theta=255,m0=191,
    final_threshold=.35,event_window_ms=100,weights_updated=False,training_started=False,
    only_changed_factor='independent negative/positive exit thresholds at T4 and T8; original v1 grid and confidence rule unchanged',
    negative_grid=NEGATIVE,positive_grid=POSITIVE,grid_size=len(NEGATIVE)**2*len(POSITIVE)**2,
    grid_order=['negative4','positive4','negative8','positive8'],
    control=dict(negative4=.01,positive4=.99,negative8=.01,positive8=.99),
    objectives={'accuracy':0.,'balanced':.0025,'efficiency':.01},
    objective_formula='IoU - lambda*(AC/fixedT16 AC), same formula on Main-fit and Main-validation',
    shortlist='union of top8 per objective on Main-fit plus the original shared .01/.99 control',
    tie_break='objective, IoU, HD recall, lower AC, lower MAC, earlier grid index',
    split='same v1 Main split: 59 fit/20 validation, sequence-disjoint and stratified by ordinary/HD',
    main_validation_scope='held out for exit parameter choice only; epoch159 weights trained on all Main',
    search_backend='exact joint T4/T8 score histograms with separate equality buckets and integer sufficient statistics',
    search_scope='all full Main candidates, every grid combination retained; no accuracy/AC rejection threshold',
    prefix_source='reuse immutable v1 prefixes; verify identity, cache and trace hashes; never recompute into old directories',
    runtime_inputs='only current prediction at the current checkpoint; no label, teacher, HD tag or future prediction',
    transport='same original binary IF, continuous membrane and weighted accumulator; projection once, readout per checkpoint',
    cost='same v1 MAC/AC/scalar accounting, repeated readouts included; separate step-specific gate comparisons',
    deployment_verification='full Dev actual CPU binary spike-add for each unique selected policy and shared control',
    roles=['train_main','train_dev'],reserved_splits_read=False)


def sources():
    paths=list(PACKAGE.glob('*.py'))+[IMPLEMENTATION/'software/scripts/evaluate_h2pro_adaptive_t_v2.py',
                                      IMPLEMENTATION/'evaluate_h2pro_adaptive_t_v2.ps1']
    return {p.relative_to(IMPLEMENTATION).as_posix():hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
            for p in sorted(paths)}


def identity(cache):
    old,layers=prior_identity(cache)
    if old!=read_json(REFERENCE/'identity.json'):
        raise ValueError('Original adaptive-T identity/source changed')
    seal=read_json(REFERENCE/'main_seal.json')
    if (sha256(REFERENCE/'identity.json')!=seal['identity_sha256']
            or sha256(REFERENCE/'selection.json')!=seal['selection_sha256']):
        raise ValueError('Original Main seal changed')
    # Only hash the old comparison here: no Dev outputs are used for new selection.
    return dict(design=DESIGN,prior_identity=old,reference_directory=str(REFERENCE),
        reference_files={name:sha256(REFERENCE/name) for name in ('identity.json','main_seal.json','selection.json','comparison.json')},
        source_files=old['source_files'],sources=sources(),cache=str(Path(cache).resolve())),layers
