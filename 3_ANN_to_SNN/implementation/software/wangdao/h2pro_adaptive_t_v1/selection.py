"""Full-event metrics, transparent cost tradeoffs and Main-only selection."""
import hashlib
import json
import numpy as np

from ..h2pro_snn.diagnostics import aggregate_counts
from .core import STEPS,THRESHOLD,Policy,choose,ledger
from .common import DESIGN,split_main


def policy_id(policy):
    return hashlib.sha256(json.dumps(policy.as_dict(),sort_keys=True).encode()).hexdigest()[:12]


def policies():
    return [Policy(neg,pos,mode).validate() for mode in DESIGN['modes']
            for neg in DESIGN['negative_grid'] for pos in DESIGN['positive_grid']]


def summarize_counts(rows):
    c=aggregate_counts(rows);d=2*c['tp']+c['fp']+c['fn']
    c['f1']=2*c['tp']/d if d else None
    return c


def counts(row,labels,predicted):
    pos=labels==1;tp=int(np.count_nonzero(pos & predicted));fp=int(np.count_nonzero(~pos & predicted))
    fn=row['target_events']-tp
    return dict(events=row['events'],target_events=row['target_events'],candidate_events=row['candidate_events'],
        candidate_targets=row['candidate_targets'],proposal_fn=row['target_events']-row['candidate_targets'],
        refiner_fn=row['candidate_targets']-tp,tp=tp,fp=fp,fn=fn,tn=row['events']-tp-fp-fn)


def score(bundle,policy=None,fixed_step=None,details=False):
    if (policy is None)==(fixed_step is None):
        raise ValueError('Exactly one fixed step or adaptive policy required')
    p=bundle['p'];n=p.shape[1];ids=np.arange(n)
    ix=np.full(n,STEPS.index(fixed_step),np.uint8) if fixed_step else choose(p,policy)
    probability=p[ix,ids];spikes=bundle['spikes'][ix,ids]
    predicted=probability>=THRESHOLD
    records=[]
    for seq in bundle['sequences']:
        lo,hi=seq['lo'],seq['hi'];r=seq['row']
        rec=dict(sequence=r['sequence'],high_density=r['high_density'],
                 counts=counts(r,bundle['labels'][lo:hi],predicted[lo:hi]),
                 hidden_spikes=[int(spikes[lo:hi].sum()),0],
                 exit_counts={str(t):int(np.count_nonzero(ix[lo:hi]==i)) for i,t in enumerate(STEPS)})
        records.append(rec)
    c=ledger(ix,spikes,policy,fixed_step is not None)
    prop=sum(s['row']['proposal_sparse_mac'] for s in bundle['sequences'])
    dense=sum(s['row']['proposal_dense_mac'] for s in bundle['sequences'])
    c.update(proposal_sparse_mac=prop,proposal_dense_mac=dense,
        refiner_mac=c['input_projection_mac']+c['readout_mac'],
        inference_equivalent_mac_sparse_proposal=prop+c['input_projection_mac']+c['readout_mac'],
        inference_equivalent_mac_dense_proposal=dense+c['input_projection_mac']+c['readout_mac'],
        inference_equivalent_ac=c['synaptic_ac'],extra_readout_mac_vs_fixed=c['readout_mac']-n*128)
    result={group:summarize_counts([r['counts'] for r in records if group=='overall' or r['high_density']==(group=='high_density')])
            for group in ('overall','ordinary','high_density')}
    result.update(cost=c,threshold=THRESHOLD,role=bundle['role'],
                  policy=policy.as_dict() if policy else None,fixed_step=fixed_step)
    if details:
        result['sequences']=records
        early=ix<2;y=bundle['labels']==1;old=p[2]>=THRESHOLD
        result['early_exit_errors_vs_t16']={
            'early_candidates':int(early.sum()),'changed_predictions':int(np.count_nonzero(predicted!=old)),
            't16_correct_to_wrong':int(np.count_nonzero(early & (old==y) & (predicted!=y))),
            't16_wrong_to_correct':int(np.count_nonzero(early & (old!=y) & (predicted==y))),
            'additional_target_misses':int(np.count_nonzero(early & y & old & ~predicted)),
            'recovered_targets':int(np.count_nonzero(early & y & ~old & predicted)),
            'extra_background_false_positives':int(np.count_nonzero(early & ~y & ~old & predicted)),
            'removed_background_false_positives':int(np.count_nonzero(early & ~y & old & ~predicted))}
    return result


def subset(bundle,names):
    names=set(names);parts={k:[] for k in ('p','spikes','labels')};seqs=[];offset=0
    for seq in bundle['sequences']:
        if seq['row']['sequence'] not in names:continue
        lo,hi=seq['lo'],seq['hi'];n=hi-lo
        for k in ('p','spikes'):parts[k].append(bundle[k][:,lo:hi])
        parts['labels'].append(bundle['labels'][lo:hi])
        seqs.append(dict(row=seq['row'],lo=offset,hi=offset+n));offset+=n
    if len(seqs)!=len(names):raise ValueError('Sequence subset mismatch')
    return dict(role=bundle['role'],sequences=seqs,p=np.concatenate(parts['p'],axis=1),
                spikes=np.concatenate(parts['spikes'],axis=1),labels=np.concatenate(parts['labels']))


def rank(row,coefficient,baseline):
    objective=row['overall']['iou']-coefficient*row['cost']['inference_equivalent_ac']/baseline['cost']['inference_equivalent_ac']
    return (objective,row['overall']['iou'],row['high_density']['recall'],
            -row['cost']['inference_equivalent_ac'],-row['cost']['inference_equivalent_mac_sparse_proposal'])


def select_main(bundle,progress=lambda value:None):
    if bundle['role']!='train_main':raise ValueError('Never select exit parameters on Dev')
    split=split_main([s['row'] for s in bundle['sequences']])
    fit,valid=(subset(bundle,split[k]) for k in ('fit','validation'))
    base_fit,base_val=(score(b,fixed_step=16) for b in (fit,valid))
    grid=policies();fit_results={}
    for i,policy in enumerate(grid):
        fit_results[policy_id(policy)]=score(fit,policy)
        if (i+1)%25==0:progress(dict(stage='main_fit_search',completed=i+1,total=len(grid)))
    shortlist=set()
    for coefficient in DESIGN['objectives'].values():
        shortlist.update(sorted(fit_results,key=lambda k:rank(fit_results[k],coefficient,base_fit),reverse=True)[:8])
    by_id={policy_id(p):p for p in grid}
    validation={k:score(valid,by_id[k]) for k in sorted(shortlist)}
    selected={}
    for name,coefficient in DESIGN['objectives'].items():
        key=max(validation,key=lambda k:rank(validation[k],coefficient,base_val))
        selected[name]=dict(policy_id=key,policy=by_id[key].as_dict(),coefficient=coefficient,
                            fit=fit_results[key],validation=validation[key],
                            full_main=score(bundle,by_id[key],details=True))
    # All fit candidates are retained, including accuracy/cost tradeoffs that do
    # not meet any putative acceptance threshold. Nothing stops this experiment.
    return dict(design=DESIGN,split=split,grid_size=len(grid),shortlist=sorted(shortlist),
                fit_candidates=fit_results,validation_candidates=validation,selected=selected,
                fixed_main={str(t):score(bundle,fixed_step=t,details=True) for t in STEPS},
                role='train_main',dev_used_for_selection=False,weights_updated=False,selection_sealed=True)
