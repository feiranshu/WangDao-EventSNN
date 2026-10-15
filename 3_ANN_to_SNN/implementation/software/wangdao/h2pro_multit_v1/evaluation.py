"""Full Dev prefixes and a predeclared exit rule; no threshold fitting."""
from pathlib import Path
import time
import numpy as np
import torch

from ..h2pro_calibration_v1.core import forward, Encoder, probabilities, validate_layers
from ..h2pro_r1_train_v1.evaluation import score as original_score, delta
from ..h2pro_adaptive_t_v1.common import atomic_npz
from ..h2pro_adaptive_t_v2.core import choose, incremental, ledger
from ..h2pro_adaptive_t_v2.scoring import score
from .common import STEPS, EXIT_POLICY, VERSION


def score_prefixes(data,raw,spikes):
    if raw.shape!=(3,data.size('train_dev'),4) or spikes.shape!=raw.shape[:2]:
        raise ValueError('Full Dev prefix arrays required')
    bundle=dict(role='train_dev',sequences=data.sequences['train_dev'],
                labels=data.roles['train_dev']['labels'].cpu().numpy(),
                p=np.stack([probabilities(v) for v in raw]),spikes=spikes)
    fixed={str(t):score(bundle,fixed_step=t,details=True) for t in STEPS}
    adaptive=score(bundle,EXIT_POLICY,details=True)
    # Preserve the historical T16 fields/cost contract for exact control replay.
    old=original_score(data,raw[2],np.column_stack((spikes[2],np.zeros(len(spikes[2]),np.int64))))
    for group in ('overall','ordinary','high_density'):
        if old[group]!=fixed['16'][group]:raise AssertionError('T16 metric contract changed')
    return dict(old,fixed=fixed,adaptive=adaptive,exit_policy=EXIT_POLICY.as_dict())


@torch.no_grad()
def evaluate(model,data,batch_size,pause=None,verify_explicit=True):
    started=time.perf_counter();model.eval()
    outputs=[];counts=[]
    for lo in range(0,data.size('train_dev'),batch_size):
        if pause is not None:pause()
        raw,spikes=model.prefixes(data.roles['train_dev']['features'][lo:lo+batch_size])
        outputs.append(raw.cpu().numpy());counts.append(spikes.cpu().numpy())
    raw=np.concatenate(outputs,axis=1);spikes=np.concatenate(counts,axis=1)
    checked=0
    if verify_explicit:
        layers=model.export_layers()
        for seq in data.sequences['train_dev']:
            lo,hi=seq['lo'],seq['hi']
            ids=np.unique(np.linspace(lo,hi-1,min(64,hi-lo),dtype=np.int64)) if hi>lo else np.empty(0,np.int64)
            x=data.roles['train_dev']['features'][torch.as_tensor(ids,device=data.device)].cpu().numpy()
            for i,t in enumerate(STEPS):
                expected,cost=forward(x,layers,'r1',t,(Encoder(255,191),Encoder()),'spike_add')
                np.testing.assert_array_equal(expected,raw[i,ids])
                if cost.executed_synaptic_ac!=int(spikes[i,ids].sum())*32:
                    raise AssertionError('Prefix explicit AC mismatch')
            out,n,ix,measured=incremental(x,layers,EXIT_POLICY,batch_size)
            expected_ix=choose(np.stack([probabilities(r[ids]) for r in raw]),EXIT_POLICY)
            np.testing.assert_array_equal(ix,expected_ix)
            np.testing.assert_array_equal(out,raw[ix,ids])
            np.testing.assert_array_equal(n,spikes[ix,ids])
            cost=ledger(ix,n,EXIT_POLICY)
            if any(cost[k]!=v for k,v in measured.items()):raise AssertionError('Adaptive explicit cost mismatch')
            checked+=len(ids)
    result=score_prefixes(data,raw,spikes)
    result.update(inference_seconds=time.perf_counter()-started,explicit_verified_candidates_per_prefix=checked,
                  adaptive_explicit_verified_candidates=checked,
                  student_backend='dense FP64 prefix count reference; full spike-derived equivalent AC; explicit subset audit')
    return result,raw,spikes


def export(path,model):
    model.verify_frozen_scales()
    arrays=dict(version=np.array(VERSION),dimensions=np.array([36,64,32,4]),stage=np.array('r1'),
                coding_steps=np.array(STEPS),threshold=np.array(.35),theta=np.array(255),m0=np.array(191),
                variant=np.array('C1-proposal-motion'),spiking_transport_layers=np.array([1]))
    for i,(w,s) in enumerate(model.export_layers()):
        arrays[f'layer{i}_weight'],arrays[f'layer{i}_scale']=w,s
    atomic_npz(Path(path),**arrays)


def load_export(path):
    with np.load(path,allow_pickle=False) as z:
        if (z['version'].item()!=VERSION or z['stage'].item()!='r1' or z['theta'].item()!=255
                or z['m0'].item()!=191 or z['threshold'].item()!=.35
                or not np.array_equal(z['coding_steps'],STEPS)
                or not np.array_equal(z['dimensions'],[36,64,32,4])
                or z['variant'].item()!='C1-proposal-motion'
                or not np.array_equal(z['spiking_transport_layers'],[1])):
            raise ValueError('Wrong multi-T export contract')
        layers=tuple((z[f'layer{i}_weight'].copy(),z[f'layer{i}_scale'].copy()) for i in range(3))
    validate_layers(layers);return layers
