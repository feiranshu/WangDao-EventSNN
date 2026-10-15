"""Main-only one-axis calibration; fixed factorial diagnostics declared in advance."""
from ..h2pro_adaptive_t_v2.core import Policy
from ..h2pro_adaptive_t_v2.scoring import score, subset, policy_id
from .common import DESIGN, NEGATIVE, POSITIVE, OBJECTIVES, ANCHORS, split_main


def grids():
    return dict(negative_only=[Policy(.01,.99,n,.99).validate() for n in NEGATIVE],
                positive_only=[Policy(.01,.99,.01,p).validate() for p in POSITIVE])


def rank(row, objective, baseline):
    o=row['overall'];cost=row['cost'];spec=OBJECTIVES[objective]
    value=(spec['iou']*o['iou']+spec['recall']*o['recall']+
           spec['high_density_recall']*row['high_density']['recall']-
           spec['ac_penalty']*cost['inference_equivalent_ac']/baseline['cost']['inference_equivalent_ac'])
    return (value,o['iou'],o['recall'],row['high_density']['recall'],
            -cost['inference_equivalent_ac'],-cost['inference_equivalent_mac_sparse_proposal'])


def select_main(data, progress=lambda value:None):
    if data['role']!='train_main':
        raise ValueError('Main-only calibration; Dev cannot select exit gates')
    split=split_main([s['row'] for s in data['sequences']])
    fit,valid=(subset(data,split[k]) for k in ('fit','validation'))
    baselines={k:score(b,fixed_step=16) for k,b in (('fit',fit),('validation',valid))}
    arms={};evaluations={k:p.as_dict() for k,p in ANCHORS.items()}
    for arm,policies in grids().items():
        fit_rows={str(i):score(fit,p) for i,p in enumerate(policies)}
        short={policies.index(ANCHORS['shared']),policies.index(ANCHORS[arm])}
        for objective in OBJECTIVES:
            short.update(sorted(range(len(policies)),
                key=lambda i:(rank(fit_rows[str(i)],objective,baselines['fit']),-i),reverse=True)[:5])
        validation={str(i):score(valid,policies[i]) for i in sorted(short)}
        selected={}
        for objective in OBJECTIVES:
            i=max(sorted(short),key=lambda i:(rank(validation[str(i)],objective,baselines['validation']),-i))
            policy=policies[i]
            selected[objective]=dict(grid_index=i,policy_id=policy_id(policy),policy=policy.as_dict(),
                fit=fit_rows[str(i)],validation=validation[str(i)])
            evaluations[arm+'_'+objective]=policy.as_dict()
        arms[arm]=dict(fit_candidates=fit_rows,shortlist=sorted(short),validation_candidates=validation,selected=selected)
        progress(dict(stage='main_arm_selected',arm=arm,fit_points=len(policies),validation_points=len(short),
                      selected={k:v['policy'] for k,v in selected.items()}))
    full={}
    for spec in evaluations.values():
        policy=Policy(**spec);key=policy_id(policy)
        if key not in full:full[key]=score(data,policy,details=True)
    return dict(design=DESIGN,role='train_main',split=split,arms=arms,
                evaluation_policies=evaluations,full_main=full,fixed_main=score(data,fixed_step=16,details=True),
                baselines=baselines,dev_used_for_selection=False,weights_updated=False,selection_sealed=True)
