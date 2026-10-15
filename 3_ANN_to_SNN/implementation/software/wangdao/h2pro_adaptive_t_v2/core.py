"""Causal step-specific exits; original binary IF, weights and readout retained."""
from dataclasses import asdict,dataclass
import numpy as np
from ..h2pro_adaptive_t_v1.core import (
    STEPS,THETA,M0,THRESHOLD,Policy as SharedPolicy,validate_features,ledger as shared_ledger)
from ..h2pro_calibration_v1.core import quantize_hidden,probabilities


@dataclass(frozen=True)
class Policy:
    negative4: float | None = None
    positive4: float | None = None
    negative8: float | None = None
    positive8: float | None = None

    def validate(self):
        for step in (4,8):
            n,p=self.at(step)
            if ((n is not None and not 0<=n<THRESHOLD)
                    or (p is not None and not THRESHOLD<p<=1)):
                raise ValueError('Each negative/positive exit threshold must bracket fixed 0.35')
        return self

    def at(self,step):
        if step==4:return self.negative4,self.positive4
        if step==8:return self.negative8,self.positive8
        raise ValueError('Exit parameters exist only at T4 and T8')

    def as_dict(self):return asdict(self)


CONTROL=Policy(.01,.99,.01,.99)


def should_exit(probability,step,policy):
    policy.validate()
    if step==16:return np.ones(len(probability),bool)
    neg,pos=policy.at(step)
    stop=np.zeros(len(probability),bool)
    if neg is not None:stop|=probability<=neg
    if pos is not None:stop|=probability>=pos
    return stop


def choose(probabilities_by_step,policy):
    p=np.asarray(probabilities_by_step)
    if p.ndim!=2 or p.shape[0]!=3 or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Finite [3,N] probabilities required')
    result=np.full(p.shape[1],2,np.uint8)
    for i,t in enumerate(STEPS[:2]):
        remaining=np.flatnonzero(result==2)
        result[remaining[should_exit(p[i,remaining],t,policy)]]=i
    return result


def ledger(exit_index,spikes,policy=None,fixed=False):
    # All projection, readout, IF/scalar/state conventions are inherited unchanged.
    result=shared_ledger(exit_index,spikes,SharedPolicy(),fixed)
    if not fixed:
        if policy is None:raise ValueError('Adaptive cost needs the step-specific policy')
        policy.validate()
        visited4=len(exit_index);visited8=visited4-result['exit_counts']['4']
        comparisons=sum(n*((neg is not None)+(pos is not None))
                        for n,(neg,pos) in ((visited4,policy.at(4)),(visited8,policy.at(8))))
        result.update(gate_threshold_comparisons=comparisons,exit_boolean_ors=comparisons)
    return result


def incremental(features,layers,policy,batch_size=8192):
    """Execute actual binary spike-add only for still-active candidates.

    Same projection/IF/reset/quantization/readout sequence as v1. Only the pair
    of confidence thresholds changes at T8. No labels, teacher or future output.
    """
    validate_features(features,layers);policy.validate()
    if type(batch_size) is not int or batch_size<1:raise ValueError('Positive batch size required')
    n=len(features);result=np.empty((n,4),np.float32);exit_index=np.empty(n,np.uint8)
    final_spikes=np.empty(n,np.int64)
    measured=dict(input_projection_mac=0,readout_mac=0,synaptic_ac=0,if_adds=0,
        if_comparisons=0,if_resets=0,readout_evaluations=0,active_step_rows=0,
        gate_threshold_comparisons=0,stability_comparisons=0,exit_boolean_ors=0,exit_boolean_ands=0)
    w0,w1,w2=[w.astype(np.int32) for w,_ in layers];s0,s1,s2=[s.astype(np.float64) for _,s in layers]
    for lo in range(0,n,batch_size):
        x=features[lo:lo+batch_size]
        current=quantize_hidden((x.astype(np.int32)@w0).astype(np.float64)*s0/255).astype(np.int16)
        measured['input_projection_mac']+=len(x)*2304
        active=np.arange(len(x),dtype=np.int64);membrane=np.full(current.shape,M0,np.int16)
        accum=np.zeros((len(x),32),np.int32);count=np.zeros(len(x),np.int64)
        for step in range(1,17):
            if not len(active):break
            membrane+=current;fired=membrane>=THETA
            for channel in np.flatnonzero(fired.any(axis=0)):
                ids=np.flatnonzero(fired[:,channel]);accum[ids]+=w1[channel]
                measured['synaptic_ac']+=len(ids)*32
            per_row=fired.sum(axis=1,dtype=np.int64);count+=per_row;membrane[fired]-=THETA
            measured['if_adds']+=len(active)*64;measured['if_comparisons']+=len(active)*64
            measured['if_resets']+=int(per_row.sum());measured['active_step_rows']+=len(active)
            if step not in STEPS:continue
            h2=quantize_hidden(accum.astype(np.float64)*s1/step)
            y=((h2.astype(np.int32)@w2).astype(np.float64)*s2/255).astype(np.float32)
            stop=should_exit(probabilities(y),step,policy)
            measured['readout_mac']+=len(active)*128;measured['readout_evaluations']+=len(active)
            if step<16:
                neg,pos=policy.at(step);comparisons=len(active)*((neg is not None)+(pos is not None))
                measured['gate_threshold_comparisons']+=comparisons;measured['exit_boolean_ors']+=comparisons
            done=lo+active[stop]
            result[done],exit_index[done],final_spikes[done]=y[stop],STEPS.index(step),count[stop]
            keep=~stop
            active,current,membrane,accum,count=(v[keep] for v in (active,current,membrane,accum,count))
    return result,final_spikes,exit_index,measured
