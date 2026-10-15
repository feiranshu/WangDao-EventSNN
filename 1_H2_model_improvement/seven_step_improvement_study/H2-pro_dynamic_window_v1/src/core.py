"""Causal variable-duration event slices with physical-time H2-pro context.

No labels enter this module. All input intervals are half-open. A 25 ms
watermark grid makes count-triggered boundaries independent of input packet size.
The original 100 ms anchor heatmaps retain the trained motion/history meaning;
their extra inference cost is included even when offline replay memoizes them.
"""
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
import time

import numpy as np

EXPERIMENT = Path(__file__).resolve().parents[1]
ANN_ROOT = EXPERIMENT.parents[1] / 'implementation'
REPOSITORY = EXPERIMENT.parents[2]
SNN_ROOT = REPOSITORY / '3_ANN_to_SNN/implementation'
sys.path.insert(0, str(ANN_ROOT / 'software'))
from wangdao.h2_pro.stream_state import CausalPipeline, EVENT_DTYPE
from wangdao.h2_pro.causal_feature_builder import base_features, neighbor_mean
from wangdao.h2_pro.motion_coherence import proposal_motion
from wangdao.h2.constants import PATCHES_PER_WINDOW
from wangdao.h2.features import compress_counts
from wangdao.h2.numpy_model import _sigmoid, _refiner_probability
from wangdao.h2.representation import _fine_counts_to_patch_features, patch_scores_to_heatmaps, dilate_cells, _component_sizes


@dataclass(frozen=True)
class Policy:
    name: str
    fixed_ms: int = 0
    event_budget: int = 0
    count_scale: str = 'raw'

    def validate(self):
        if self.count_scale not in ('raw', 'rate100'):
            raise ValueError('Unknown count normalization')
        if (self.fixed_ms not in (0,25,50,100) or type(self.event_budget) is not int
                or self.event_budget < 0 or bool(self.fixed_ms) == bool(self.event_budget)):
            raise ValueError('Use fixed 25/50/100 ms OR a positive event budget')
        return self


POLICIES = [Policy('fixed100',100)] + [Policy(f'fixed{ms}_{scale}',ms,0,scale)
    for scale in ('raw','rate100') for ms in (50,25)] + [
    Policy(f'adaptive{budget}_{scale}',0,budget,scale)
    for scale in ('raw','rate100') for budget in (500,1000,2000)]
POLICY_MAP = {p.name:p for p in POLICIES}


class BoundaryController:
    """Online controller: receives only the count in each CLOSED 25 ms bin."""
    def __init__(self, policy):
        self.policy=policy.validate(); self.tick=0; self.start=0; self.count=0

    def advance(self, observed_count):
        if type(observed_count) is not int or observed_count < 0:
            raise ValueError('A nonnegative observed count is required')
        self.tick+=1; self.count+=observed_count
        close = self.tick % 4 == 0
        if self.policy.fixed_ms:
            close |= (self.tick-self.start)*25 >= self.policy.fixed_ms
        else:
            close |= self.count >= self.policy.event_budget
        if close:
            result=(self.start,self.tick,self.count)
            self.start=self.tick; self.count=0
            return result
        return None


def schedule(times, t0, closed_windows, policy):
    controller=BoundaryController(policy)
    ends=np.searchsorted(times,t0+np.arange(1,closed_windows*4+1)*25.,side='left')
    rows=[]; before=0
    for end in ends:
        result=controller.advance(int(end-before)); before=int(end)
        if result is not None: rows.append(result)
    if before != len(times) or controller.count:
        raise ValueError('Require exactly the common complete 100 ms event prefix')
    return rows


def exact_mlp(features, layers):
    """FP64 integral accumulations exactly reproduce int32 frozen dot products.

    Tiny dense BLAS matrices are used for replay speed, not claimed zero-skip
    execution. Return both dense and input-zero-skipping equivalent MAC counts.
    """
    value=np.asarray(features,dtype=np.uint8)
    sparse=dense=outputs=0
    activity=[]
    for i,(w,s) in enumerate(layers):
        nonzero=int(np.count_nonzero(value)); activity.append(nonzero)
        sparse+=nonzero*w.shape[1]; dense+=len(value)*w.size; outputs+=len(value)*w.shape[1]
        raw=value.astype(np.float64) @ w.astype(np.float64)
        floating=raw*s.astype(np.float64)[None,:]/255.
        if i+1<len(layers): value=np.rint(np.clip(floating,0.,1.)*255.).astype(np.uint8)
        else: value=floating.astype(np.float32)
    return value,dict(sparse_mac=sparse,dense_mac=dense,layer_output_scalars=outputs,nonzero_inputs=activity)


class SequenceContext:
    """Offline replay cache; each result reads only events before its own boundary.

    Cache sharing accelerates the experiment. Reported inference MAC includes
    work each independent online policy would perform, including anchor updates.
    """
    def __init__(self, events, t0, closed_windows):
        if type(closed_windows) is not int or closed_windows < 1 or not np.isfinite(t0):
            raise ValueError('Invalid closed prefix')
        self.packet=np.empty(len(events),dtype=EVENT_DTYPE)
        for k in ('x','y','t','p'): self.packet[k]=events[k]
        e=self.packet
        if (not np.isfinite(e['t']).all() or np.any(np.diff(e['t'])<0) or np.any(e['t']<t0)
                or np.any(e['t']>=t0+100*closed_windows) or np.any((e['x']<0)|(e['x']>=346))
                or np.any((e['y']<0)|(e['y']>=260)) or np.any(e['p']>1)):
            raise ValueError('Invalid timestamp, coordinates or prefix')
        self.t0=float(t0); self.closed_windows=closed_windows
        self.positions=np.searchsorted(e['t'],t0+np.arange(closed_windows*4+1)*25.,side='left')
        self.frozen=CausalPipeline(ANN_ROOT/'software/weights/H2_D1_seed42.npz',
            ANN_ROOT/'software/weights/H2-pro-Causal-ANN-v1_int8.npz',
            variant='C1-proposal-motion',threshold=.35,initial_refiner=False)
        self.proposals={}; self.bases={}; self.slices={}
        self.cache_hits=0; self.cache_misses=0

    def proposal(self,start,end,scale):
        key=(start,end,scale)
        if key in self.proposals:return self.proposals[key]
        lo,hi=self.positions[[start,end]]; current=self.packet[lo:hi]
        if not len(current):
            result=(np.full((34,44),.5,np.float32),dict(sparse_mac=0,dense_mac=0,layer_output_scalars=0,nonzero_inputs=[0,0,0]))
        else:
            fine=np.zeros((1,2,68,88),np.int32)
            np.add.at(fine[0],(current['p'],current['y']//4,current['x']//4),1)
            if scale=='rate100' and end-start<4:
                # Normalize before logarithmic compression, preserving exact
                # original path at 100 ms. This is a separate matched factor.
                fine=fine.astype(np.float64)*(4./(end-start))
            patch=_fine_counts_to_patch_features(fine).reshape(-1,128)
            raw,cost=exact_mlp(patch,self.frozen.proposal.proposal_layers)
            heat=patch_scores_to_heatmaps(_sigmoid(raw).reshape(1,PATCHES_PER_WINDOW,16))[0]
            result=(heat,cost)
        self.proposals[key]=result
        # At 100 ms duration the two normalizers are mathematically identical.
        if end-start==4:self.proposals[(start,end,'rate100' if scale=='raw' else 'raw')]=result
        return result

    def base(self,start,end):
        key=(start,end)
        if key not in self.bases:
            lo,hi=self.positions[[start,end]]
            # Two full 50 ms lag bins, even when a slice begins mid-bin.
            earliest=self.t0+(start//2*2-4)*25.
            history_lo=int(np.searchsorted(self.packet['t'],earliest,side='left'))
            self.bases[key]=base_features(self.packet[history_lo:hi],int(lo-history_lo),self.t0)
        return self.bases[key]

    def slice(self,start,end,scale):
        if not (0<=start<end<=4*self.closed_windows and start//4==(end-1)//4):
            raise ValueError('Slice must remain inside one physical 100 ms anchor')
        key=(start,end,scale)
        if key in self.slices:
            self.cache_hits+=1;return self.slices[key]
        self.cache_misses+=1
        lo,hi=self.positions[[start,end]];current=self.packet[lo:hi]
        heat,proposal_cost=self.proposal(start,end,scale)
        mask=dilate_cells((heat>=self.frozen.proposal.proposal_threshold)[None])[0]
        x,y=current['x'].astype(np.int64),current['y'].astype(np.int64)
        candidate=mask[y//8,x//8]; selected=np.flatnonzero(candidate)
        features=np.empty((len(selected),36),np.uint8)
        # Only COMPLETE earlier 100 ms maps are legal context for this slice.
        macro=start//4
        previous=[self.proposal(m*4,(m+1)*4,'raw')[0] for m in range(max(0,macro-2),macro)]
        if len(selected):
            features[:,:28]=self.base(start,end)[selected]
            sx,sy=x[selected],y[selected];cx,cy=sx//8,sy//8
            features[:,2]=proposal_motion(sx,sy,previous)
            sizes=_component_sizes(mask)
            # Preserve the trained PHYSICAL 100 ms phase rather than compress
            # its meaning into an arbitrary new slice duration.
            phase=np.mod(current['t'][selected]-self.t0,100.)/100.
            features[:,28:32]=np.column_stack((np.rint(255*heat[cy,cx]),
                np.rint(255*neighbor_mean(heat)[cy,cx]),
                np.rint(255*np.minimum(sizes[cy,cx].astype(np.float64)/64.,1.)),
                np.rint(255*np.clip(phase,0,1)))).astype(np.uint8)
            history=np.zeros((len(selected),4),np.float32)
            if previous:
                history[:,0]=previous[-1][cy,cx];history[:,1]=neighbor_mean(previous[-1])[cy,cx]
                history[:,2:]=history[:,:2]
            if len(previous)>1:
                history[:,2]=np.maximum(history[:,2],previous[-2][cy,cx])
                history[:,3]=np.maximum(history[:,3],neighbor_mean(previous[-2])[cy,cx])
            features[:,32:36]=np.rint(255*np.clip(history,0,1)).astype(np.uint8)
            raw,refiner_cost=exact_mlp(features,self.frozen.refiner_layers)
            cp=_refiner_probability(raw,2)
        else:
            cp=np.empty(0,np.float32);refiner_cost=dict(sparse_mac=0,dense_mac=0,layer_output_scalars=0,nonzero_inputs=[0,0,0])
        probabilities=np.zeros(len(current),np.float32);probabilities[selected]=cp
        result=dict(lo=int(lo),hi=int(hi),candidate=candidate,probabilities=probabilities,
                    features=features,proposal_cost=proposal_cost,refiner_cost=refiner_cost)
        self.slices[key]=result
        return result

    def run(self,policy):
        started=time.perf_counter(); plan=schedule(self.packet['t'],self.t0,self.closed_windows,policy)
        n=len(self.packet); candidate=np.zeros(n,bool);prob=np.zeros(n,np.float32);latency=np.empty(n,np.float64)
        cost=dict(proposal_current_sparse_mac=0,proposal_current_dense_mac=0,
                  proposal_history_extra_sparse_mac=0,proposal_history_extra_dense_mac=0,
                  refiner_mac=0,refiner_input_zero_skip_equivalent_mac=0,
                  proposal_output_scalars=0,refiner_output_scalars=0,neural_ac=0)
        window_rows=[];all_features=[];last=0;full_macros=set()
        for start,end,count in plan:
            r=self.slice(start,end,policy.count_scale);lo,hi=r['lo'],r['hi']
            if lo!=last or hi-lo!=count:raise AssertionError('Event coverage changed')
            candidate[lo:hi]=r['candidate'];prob[lo:hi]=r['probabilities'];latency[lo:hi]=self.t0+end*25-self.packet['t'][lo:hi]
            all_features.append(r['features']);last=hi
            pc,rc=r['proposal_cost'],r['refiner_cost']
            cost['proposal_current_sparse_mac']+=pc['sparse_mac'];cost['proposal_current_dense_mac']+=pc['dense_mac']
            cost['proposal_output_scalars']+=pc['layer_output_scalars']
            cost['refiner_mac']+=rc['dense_mac'];cost['refiner_input_zero_skip_equivalent_mac']+=rc['sparse_mac']
            cost['refiner_output_scalars']+=rc['layer_output_scalars']
            if end-start==4:full_macros.add(start//4)
            window_rows.append([start,end,lo,hi,int(r['candidate'].sum())])
        if last!=n:raise AssertionError('Missing published events')
        # The online policy maintains an anchor even at the final boundary so it
        # can continue processing a future packet. No free teacher/context work.
        for m in range(self.closed_windows):
            if m in full_macros:continue
            _,pc=self.proposal(m*4,(m+1)*4,'raw')
            cost['proposal_history_extra_sparse_mac']+=pc['sparse_mac']
            cost['proposal_history_extra_dense_mac']+=pc['dense_mac']
            cost['proposal_output_scalars']+=pc['layer_output_scalars']
        cost['mac_sparse_proposal']=cost['proposal_current_sparse_mac']+cost['proposal_history_extra_sparse_mac']+cost['refiner_mac']
        cost['mac_dense_proposal']=cost['proposal_current_dense_mac']+cost['proposal_history_extra_dense_mac']+cost['refiner_mac']
        cost['count_controller_tick_checks']=4*self.closed_windows
        cost['proposal_anchor_reused']=len(full_macros)
        features=np.concatenate(all_features) if all_features else np.empty((0,36),np.uint8)
        return dict(candidate=candidate,probabilities=prob,features=features,
                    windows=np.asarray(window_rows,np.int64),cost=cost,
                    wait_ms=dict(mean=float(latency.mean()) if n else 0.,p95=float(np.quantile(latency,.95)) if n else 0.,
                                 maximum=float(latency.max()) if n else 0.),
                    replay_seconds=time.perf_counter()-started)


def counts(labels,probabilities,candidate):
    y=np.asarray(labels)==1;p=np.asarray(probabilities)>=.35;c=np.asarray(candidate,bool)
    if not (y.shape==p.shape==c.shape):raise ValueError('Metric shape mismatch')
    return dict(events=len(y),target_events=int(y.sum()),candidate_events=int(c.sum()),
        candidate_targets=int((y&c).sum()),tp=int((y&p).sum()),fp=int((~y&p).sum()),
        fn=int((y&~p).sum()),tn=int((~y&~p).sum()),proposal_fn=int((y&~c).sum()),refiner_fn=int((y&c&~p).sum()))


def metrics(c):
    r=dict(c);tp,fp,fn=c['tp'],c['fp'],c['fn']
    r.update(iou=tp/(tp+fp+fn) if tp+fp+fn else 0.,precision=tp/(tp+fp) if tp+fp else 0.,
             recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
             proposal_recall=c['candidate_targets']/c['target_events'] if c['target_events'] else 0.)
    return r
