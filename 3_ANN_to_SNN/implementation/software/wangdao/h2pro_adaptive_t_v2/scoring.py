"""Full-event scoring with the same frozen v1 metric and cost definitions."""
import numpy as np
from ..h2pro_adaptive_t_v1.selection import counts,summarize_counts,policy_id,subset,rank
from .core import STEPS,THRESHOLD,CONTROL,choose,ledger


def error_changes(labels,predicted,reference):
    y=labels==1
    return dict(changed_predictions=int(np.count_nonzero(predicted!=reference)),
        correct_to_wrong=int(np.count_nonzero((reference==y)&(predicted!=y))),
        wrong_to_correct=int(np.count_nonzero((reference!=y)&(predicted==y))),
        additional_target_misses=int(np.count_nonzero(y&reference&~predicted)),
        recovered_targets=int(np.count_nonzero(y&~reference&predicted)),
        extra_background_false_positives=int(np.count_nonzero(~y&~reference&predicted)),
        removed_background_false_positives=int(np.count_nonzero(~y&reference&~predicted)))


def score(bundle,policy=None,fixed_step=None,details=False):
    if (policy is None)==(fixed_step is None):raise ValueError('Exactly one fixed step or policy')
    p=bundle['p'];n=p.shape[1];ids=np.arange(n)
    ix=np.full(n,STEPS.index(fixed_step),np.uint8) if fixed_step else choose(p,policy)
    spikes=bundle['spikes'][ix,ids];predicted=p[ix,ids]>=THRESHOLD
    records=[]
    for seq in bundle['sequences']:
        lo,hi=seq['lo'],seq['hi'];r=seq['row']
        records.append(dict(sequence=r['sequence'],high_density=r['high_density'],
            counts=counts(r,bundle['labels'][lo:hi],predicted[lo:hi]),
            hidden_spikes=[int(spikes[lo:hi].sum()),0],
            exit_counts={str(t):int(np.count_nonzero(ix[lo:hi]==i)) for i,t in enumerate(STEPS)}))
    c=ledger(ix,spikes,policy,fixed_step is not None)
    prop=sum(s['row']['proposal_sparse_mac'] for s in bundle['sequences'])
    dense=sum(s['row']['proposal_dense_mac'] for s in bundle['sequences'])
    ref=c['input_projection_mac']+c['readout_mac']
    c.update(proposal_sparse_mac=prop,proposal_dense_mac=dense,refiner_mac=ref,
        inference_equivalent_mac_sparse_proposal=prop+ref,inference_equivalent_mac_dense_proposal=dense+ref,
        inference_equivalent_ac=c['synaptic_ac'],extra_readout_mac_vs_fixed=c['readout_mac']-n*128)
    result={g:summarize_counts([r['counts'] for r in records if g=='overall' or r['high_density']==(g=='high_density')])
            for g in ('overall','ordinary','high_density')}
    result.update(cost=c,threshold=THRESHOLD,role=bundle['role'],policy=policy.as_dict() if policy else None,fixed_step=fixed_step)
    if details:
        control_ix=choose(p,CONTROL);control=p[control_ix,ids]>=THRESHOLD
        result.update(sequences=records,changes_vs_t16=error_changes(bundle['labels'],predicted,p[2]>=THRESHOLD),
                      changes_vs_shared=error_changes(bundle['labels'],predicted,control))
    return result
