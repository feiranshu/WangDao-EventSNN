"""Random-feature/random-label fixtures only; never constructs a real-data loader."""
from pathlib import Path
from dataclasses import replace
from uuid import uuid4
import numpy as np
import torch
from .common import ROOT,Config
from .data import ResidentData
from .model import Refiner,optimizer_for,load_export
from .runtime import initial_state,run_epochs,checkpoint,restore
from .storage import seed_all,rng_state,set_rng,load_checkpoint
from .evaluation import evaluate,export


class NeverPause:
    def check(self):pass


def fixture_directory(label):
    p=ROOT/'results/checks/fixtures'/(label+'_'+str(uuid4()));p.mkdir(parents=True)
    return p


def synthetic_data(device,arm='fixed100'):
    rng=np.random.default_rng(612);data=ResidentData.__new__(ResidentData)
    data.arm=arm;data.device=device;data.roles={};data.sequences={}
    for role,n in (('train_main',257),('train_dev',98)):
        y=rng.integers(0,2,n,dtype=np.uint8)
        keep=(np.arange(n)%5!=0) if arm=='fixed100' else np.ones(n,bool)
        ids=np.flatnonzero(keep).astype(np.int32);x=rng.integers(0,256,(len(ids),36),dtype=np.uint8)
        mapping=np.full(n,-1,np.int32);mapping[ids]=np.arange(len(ids),dtype=np.int32)
        data.roles[role]={'features':torch.from_numpy(x).to(device),'labels':torch.from_numpy(y[ids]).to(device),
                          'mapping':torch.from_numpy(mapping).to(device)}
        seqs=[];offset=0
        for i,(lo,hi) in enumerate(((0,n//2),(n//2,n))):
            local=ids[(ids>=lo)&(ids<hi)]-lo
            cost=dict(proposal_current_sparse_mac=16,proposal_current_dense_mac=32,
                      proposal_history_extra_sparse_mac=0,proposal_history_extra_dense_mac=0)
            entry=dict(cost=cost,waiting_ms={'mean':25.},candidate_events=len(local))
            row=dict(sequence='synthetic_'+str(i),metadata={'high_density':bool(i)},arms={arm:entry})
            seqs.append(dict(row=row,lo=offset,hi=offset+len(local),source_lo=lo,source_hi=hi,
                             full_labels=y[lo:hi],candidate_ids=local,initial_probability=np.zeros(hi-lo,np.float32)))
            offset+=len(local)
        data.sequences[role]=seqs
    data.stats=dict(synthetic_only=True,source_events=data.source_count,candidate_events=data.candidate_count)
    return data


def equal_tree(a,b):
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or not torch.equal(a.detach().cpu(),b.detach().cpu()):raise AssertionError('Tensor state differs')
    elif isinstance(a,dict):
        if a.keys()!=b.keys():raise AssertionError('State keys differ')
        for k in a:equal_tree(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        if len(a)!=len(b):raise AssertionError('State lengths differ')
        for x,y in zip(a,b):equal_tree(x,y)
    elif a!=b:raise AssertionError('Scalar state differs')


def resume_check(device,include_power_loss=True):
    config=Config(learning_rate=.003,source_batch_size=64,evaluation_batch_size=64)
    output=fixture_directory('resume_'+device.type);data=synthetic_data(device)
    identity={'synthetic_only':True,'check':'actual_production_loop_resume','device':str(device)}
    seed_all(412);initial=Refiner().to(device);initial_rng=rng_state()
    first,_,_=evaluate(initial,data,64)
    def fresh(directory):
        m=Refiner().to(device);m.load_state_dict(initial.state_dict());o=optimizer_for(m,config);set_rng(initial_rng)
        state=initial_state(first);checkpoint(directory/'last.pt',m,o,state,identity,'fixed100')
        checkpoint(directory/'best.pt',m,o,state,identity,'fixed100');export(directory/'best_int8.npz',m,'fixed100',0)
        return m,o,state
    ad=output/'uninterrupted';a,oa,sa=fresh(ad)
    if run_epochs(ad,'fixed100',a,oa,sa,identity,data,config,first,NeverPause(),2,1)!='complete':raise AssertionError('Oracle failed')
    expected_rng=rng_state()
    bd=output/'resumed';b,ob,sb=fresh(bd)
    if run_epochs(bd,'fixed100',b,ob,sb,identity,data,config,first,NeverPause(),2,1,_test_pause_after=2)!='paused':raise AssertionError('Pause failed')
    c=Refiner().to(device);oc=optimizer_for(c,config);sc=restore(bd/'last.pt',c,oc,identity,'fixed100')
    run_epochs(bd,'fixed100',c,oc,sc,identity,data,config,first,NeverPause(),2,1)
    equal_tree(a.state_dict(),c.state_dict());equal_tree(oa.state_dict(),oc.state_dict());equal_tree(expected_rng,rng_state())
    if sc['global_step']!=10 or len(sc['history'])!=2:raise AssertionError('Wrong resumed update count')
    for aa,bb in zip(sa['history'],sc['history']):
        for kind in ('float','int8'):equal_tree(aa['evaluation'][kind],bb['evaluation'][kind])
    power=False
    if include_power_loss:
        cd=output/'power_loss';m,o,s=fresh(cd)
        def crash(epoch):raise RuntimeError('Injected synthetic crash before last epoch commit')
        try:run_epochs(cd,'fixed100',m,o,s,identity,data,config,first,NeverPause(),2,1,_test_after_artifacts=crash)
        except RuntimeError as e:
            if 'Injected synthetic' not in str(e):raise
        saved=load_checkpoint(cd/'last.pt')
        if saved['state']['cursor']['phase']!='evaluate' or saved['state']['global_step']!=5:raise AssertionError('Lost safe epoch checkpoint')
        m2=Refiner().to(device);o2=optimizer_for(m2,config);s2=restore(cd/'last.pt',m2,o2,identity,'fixed100')
        run_epochs(cd,'fixed100',m2,o2,s2,identity,data,config,first,NeverPause(),2,1)
        equal_tree(a.state_dict(),m2.state_dict());equal_tree(oa.state_dict(),o2.state_dict());equal_tree(expected_rng,rng_state())
        power=True
    layers=load_export(bd/'epochs/epoch_002_int8.npz')
    for (w,s),(ew,es) in zip(layers,c.export_layers()):
        np.testing.assert_array_equal(w,ew);np.testing.assert_array_equal(s,es)
    return dict(status='passed',data='257 synthetic source events, random features/labels; no real data',
        optimizer_updates_per_branch=10,paused_after_step=2,model_adam_rng_exact=True,
        epoch_commit_power_loss_exact=power,epoch_export_exact=True,fixture=str(output))
