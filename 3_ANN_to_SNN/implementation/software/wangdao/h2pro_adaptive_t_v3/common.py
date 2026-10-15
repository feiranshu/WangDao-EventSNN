"""Predeclared single-factor grids and immutable inherited numerical contract."""
from pathlib import Path
import hashlib
from ..h2pro_adaptive_t_v2.common import (
    IMPLEMENTATION, DEFAULT_CACHE, SOURCE, REFERENCE, DEFAULT_RUN as PRIMARY_V2,
    identity as prior_identity, output_path, lock_directory, read_json, sha256,
    atomic_json, sealed_json, split_main)
from ..h2pro_adaptive_t_v2.core import Policy, CONTROL

PACKAGE = Path(__file__).resolve().parent
DEFAULT_RUN = IMPLEMENTATION/'results/runs/h2pro_adaptive_t_v3_s42_ep159'
EXTRA_V2 = PRIMARY_V2.parent/'h2pro_adaptive_t_v2_alternatives_s42_ep159'
NEGATIVE = [None,.0001,.001,.003,.005,.01,.015,.02,.025,.03,.04,.05,.06,.07,.08,.09,.1,.12,.15,.2,.3]
POSITIVE = [None,.9999,.999,.995,.99,.985,.98,.975,.97,.965,.96,.955,.95,.94,.93,.92,.91,.9,.8,.6,.4]
ANCHORS = dict(shared=CONTROL, negative_only=Policy(.01,.99,.1,.99),
               positive_only=Policy(.01,.99,.01,.95), combined=Policy(.01,.99,.1,.95))
OBJECTIVES = {
    'accuracy': dict(iou=1., recall=0., high_density_recall=0., ac_penalty=0.),
    'balanced': dict(iou=1., recall=0., high_density_recall=0., ac_penalty=.0025),
    'efficiency': dict(iou=1., recall=0., high_density_recall=0., ac_penalty=.01),
    'recovery': dict(iou=1/3, recall=1/3, high_density_recall=1/3, ac_penalty=.0025),
}
DESIGN = dict(version='h2pro_adaptive_t_v3', source_epoch=159, event_window_ms=100,
    steps=[4,8,16], theta=255, m0=191, final_threshold=.35,
    frozen='original ANN Proposal, features, candidate mask, epoch159 weights/scales, binary IF and readout',
    t4=dict(negative=.01,positive=.99),
    arms={'negative_only':'change negative8 only; positive8=.99',
          'positive_only':'change positive8 only; negative8=.01'},
    negative_grid=NEGATIVE, positive_grid=POSITIVE, grid_evaluations=42,
    anchors={k:v.as_dict() for k,v in ANCHORS.items()}, objectives=OBJECTIVES,
    objective_formula='weighted IoU, Recall, HD Recall minus ac_penalty*(AC/fixedT16 AC)',
    recovery_rationale='predeclared equal weights for IoU, overall recall and HD recall; separate tradeoff, not replacement accuracy objective',
    shortlist='per arm: union of top5 per objective on Main-fit plus shared and fixed single-factor anchor',
    validation='same objective on the same original 20 Main validation sequences; no Dev reselection',
    tie_break='objective, IoU, overall Recall, HD Recall, lower AC, lower MAC, earlier grid order',
    split='original Main sequence-disjoint stratified 59 fit / 20 validation',
    main_validation_scope='held out only for exits; epoch159 training already used all Main',
    dev_evaluation='all four fixed factorial anchors plus each Main-selected arm/objective, deduplicated',
    attribution='paired event decisions and original combined-vs-shared missed-target cohort; analysis only',
    full_event_scoring=True, no_quality_hard_rejection=True, runtime_uses_labels=False,
    source_prefixes='immutable v1 hash-verified T4/T8/T16 traces',
    execution='same v2 actual incremental binary spike-add; full Dev for every unique policy',
    cost='same sparse ANN Proposal + repeated readout MAC; AC/scalar/state separate; no measured energy',
    weights_updated=False, training_started=False, reserved_splits_read=False)


def sources():
    paths = list(PACKAGE.glob('*.py')) + [IMPLEMENTATION/'software/scripts/evaluate_h2pro_adaptive_t_v3.py',
                                         IMPLEMENTATION/'evaluate_h2pro_adaptive_t_v3.ps1']
    return {p.relative_to(IMPLEMENTATION).as_posix():hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()
            for p in sorted(paths)}


def identity(cache):
    prior,layers = prior_identity(cache)
    if prior != read_json(PRIMARY_V2/'identity.json'):
        raise ValueError('Immutable v2 identity changed')
    files = [d/name for d in (PRIMARY_V2,EXTRA_V2)
             for name in ('identity.json','selection.json','main_seal.json','comparison.json')]
    return dict(design=DESIGN, sources=sources(), prior_identity=prior,
                reference_files={p.relative_to(IMPLEMENTATION).as_posix():sha256(p) for p in files},
                cache=str(Path(cache).resolve())),layers
