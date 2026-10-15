"""Derivative-free calibration of IF parameters only; all network weights fixed."""
from __future__ import annotations
import hashlib
import json
import os
import time
import numpy as np
import torch

from .common import PROTOCOL, atomic_json, parameter_id, parameter_key, read_json
from .core import Channels, defaults, validate


def setup(device='cuda'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu explicitly for a slower calibration')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    d = torch.device(device)
    return d, {'torch':torch.__version__,'numpy':np.__version__,'cuda':torch.version.cuda,
               'device':str(d),'gpu':torch.cuda.get_device_name(d) if d.type == 'cuda' else None,
               'integer_reference':'FP64 exact sums; original FP32 logits','deterministic':True}


def sample(sequences):
    xs,ys,ws,hs,ts,receipts=[],[],[],[],[],[]
    limit=PROTOCOL['sample']['max_per_class_per_main_sequence']
    seed=PROTOCOL['sample']['seed']
    for index,s in enumerate(sequences):
        if s['row']['role'] != 'train_main':
            raise ValueError('Only Main may provide calibration samples')
        y=s['labels'][s['candidate']]
        selected=[]; weights=[]; strata=[]
        for cls in (0,1):
            candidates=np.flatnonzero(y == cls); k=min(limit,len(candidates))
            rng=np.random.default_rng(np.random.SeedSequence([seed,index,cls]))
            ids=np.sort(rng.choice(candidates,k,replace=False)) if k else np.empty(0,np.int64)
            selected.extend(ids.tolist()); weights.extend([len(candidates)/k]*k if k else [])
            strata.append({'label':cls,'population':len(candidates),'sampled':k,
                           'indices_sha256':hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()})
        order=np.argsort(selected); ids=np.asarray(selected,np.int64)[order]; w=np.asarray(weights,np.float64)[order]
        xs.append(s['features'][ids]); ys.append(y[ids]); ws.append(w)
        ts.append(s['teacher'][ids]); hs.append(np.full(len(ids),s['row']['high_density'],bool))
        receipts.append({'sequence':s['row']['sequence'],'sampled':len(ids),'candidate_population':len(y),'strata':strata})
    fn={g:sum(s['row']['target_events']-s['row']['candidate_targets'] for s in sequences
              if g == 'overall' or s['row']['high_density'] == (g == 'high_density')) for g in ('overall','ordinary','high_density')}
    return {'features':np.concatenate(xs),'labels':np.concatenate(ys),'weights':np.concatenate(ws),
            'teacher':np.concatenate(ts),'hd':np.concatenate(hs),'proposal_fn':fn,'receipts':receipts}


def quantize(x):
    return torch.round(x.clamp(0.,1.)*255.).clamp(0.,255.)


def class_margin(raw):
    classes=raw.reshape(-1,2,2).max(dim=2).values
    return (classes[:,1].double()-classes[:,0].double())


def prob(raw):
    classes=raw.reshape(-1,2,2).max(dim=2).values
    shifted=classes-classes.max(dim=1,keepdim=True).values
    exp=shifted.double().exp()
    return (exp[:,1]/exp.sum(dim=1)).float()


class Runner:
    def __init__(self, features, layers, device):
        self.device=device
        self.weights=[torch.tensor(w.copy(),dtype=torch.float64,device=device) for w,_ in layers]
        self.scales=[torch.tensor(s.copy(),dtype=torch.float64,device=device) for _,s in layers]
        x=torch.tensor(features,dtype=torch.float64,device=device)
        self.first=quantize((x@self.weights[0])*self.scales[0]/255.)
        self.ann_second=quantize((self.first@self.weights[1])*self.scales[1]/255.)
        self.teacher=((self.ann_second@self.weights[2])*self.scales[2]/255.).float()

    @torch.no_grad()
    def transport(self, value, index, steps, encoder):
        theta=torch.tensor(encoder.theta,dtype=torch.float64,device=self.device)
        initial=torch.tensor(encoder.membrane,dtype=torch.float64,device=self.device)
        c=torch.floor((value*steps+initial)/theta).clamp(max=steps)
        if len(set(encoder.theta)) == 1:
            raw=(c@self.weights[index])*self.scales[index]/steps
            if encoder.theta[0] != 255:
                raw=raw*encoder.theta[0]/255.
        else:
            raw=((c*theta)@self.weights[index])*self.scales[index]/steps/255.
        return raw,c.sum(dim=1)

    @torch.no_grad()
    def forward(self, stage, steps, encoders):
        validate(encoders,steps)
        zero=torch.zeros(len(self.first),dtype=torch.float64,device=self.device)
        if stage == 'ann':
            return self.teacher,zero,zero
        if stage not in ('r1','r2'):
            raise ValueError('Unknown stage')
        h,n1=self.transport(self.first,1,steps,encoders[0]); h=quantize(h)
        if stage == 'r1':
            return ((h@self.weights[2])*self.scales[2]/255.).float(),n1,zero
        raw,n2=self.transport(h,2,steps,encoders[1])
        return raw.float(),n1,n2


class TaskRunner(Runner):
    def __init__(self, data, layers, device):
        super().__init__(data['features'],layers,device)
        np.testing.assert_array_equal(self.teacher.cpu().numpy(),data['teacher'])
        self.labels=torch.tensor(data['labels'],device=device).bool()
        self.weight=torch.tensor(data['weights'],dtype=torch.float64,device=device)
        self.hd=torch.tensor(data['hd'],device=device).bool()
        self.proposal_fn=data['proposal_fn']
        self.teacher_right=(prob(self.teacher) >= .35) == self.labels
        self.teacher_margin=class_margin(self.teacher)

    @torch.no_grad()
    def assess(self, stage, steps, encoders):
        raw,n1,n2=self.forward(stage,steps,encoders)
        pred=prob(raw) >= .35
        err=(class_margin(raw)-self.teacher_margin).square()
        vals=[]
        for name,mask in [('ordinary',~self.hd),('high_density',self.hd)]:
            w=self.weight*mask
            tp=(w*(pred & self.labels)).sum(); fp=(w*(pred & ~self.labels)).sum()
            fn=(w*(~pred & self.labels)).sum()+self.proposal_fn[name]
            vals.extend([tp/(tp+fp+fn).clamp_min(1.),tp/(tp+fn).clamp_min(1.),tp/(tp+fp).clamp_min(1.),
                         (w*((pred != self.labels) & self.teacher_right)).sum()/w.sum(),(w*err).sum()/w.sum()])
        vals.extend([(self.weight*n1).sum(),(self.weight*n2).sum()])
        v=torch.stack(vals).cpu().numpy()
        return {'balanced_iou':float((v[0]+v[5])/2),'ordinary_iou':float(v[0]),'hd_iou':float(v[5]),
                'ordinary_recall':float(v[1]),'hd_recall':float(v[6]),
                'teacher_harm':float((v[3]+v[8])/2),'margin_mse':float((v[4]+v[9])/2),
                'estimated_ac':float(32*v[10]+4*v[11]),'parameter_id':parameter_id(encoders)}

    def grouping(self):
        result=[]; diagnostics=[]
        for activation in (self.first,self.ann_second):
            mean=activation.mean(0).cpu().numpy()
            order=np.lexsort((np.arange(len(mean)),mean))
            result.append([ids.tolist() for ids in np.array_split(order,len(order)//8)])
            diagnostics.append({'mean':mean.tolist(),'zero_fraction':(activation==0).double().mean(0).cpu().tolist(),
                                'saturated_fraction':(activation==255).double().mean(0).cpu().tolist()})
        return result,diagnostics


def objective(record, encoders):
    return (-record['balanced_iou'],record['teacher_harm'],record['margin_mse'],record['estimated_ac'],parameter_key(encoders))


def proposals(encoder, ids, kind, iteration):
    t,m=encoder.theta[ids[0]],encoder.membrane[ids[0]]
    if any(encoder.theta[i] != t or encoder.membrane[i] != m for i in ids):
        raise ValueError('A calibration group must share current parameters')
    pairs={(t,m)}
    for dm in PROTOCOL['local_membrane_offsets']:
        pairs.add((t,min(t-1,max(0,m+dm))))
    for q in PROTOCOL['phase_quarters']:
        pairs.add((t,t*q//4))
    if kind == 'joint':
        theta_choices=PROTOCOL['threshold_grid'] if iteration == 0 else [t+dt for dt in PROTOCOL['local_threshold_offsets']]
        for nt in theta_choices:
            if not 128 <= nt <= 510:
                continue
            if iteration == 0:
                pairs.update((nt,nt*q//4) for q in PROTOCOL['phase_quarters'])
            pairs.add((nt,min(nt-1,m*nt//t)))
    out=[]
    for nt,nm in sorted(pairs):
        theta=list(encoder.theta); membrane=list(encoder.membrane)
        for i in ids:
            theta[i],membrane[i]=nt,nm
        out.append(Channels(tuple(theta),tuple(membrane)))
    return out


def fit_branch(runner,stage,steps,branch,initial,grouping,directory):
    """Resume at the next completed coordinate; no optimizer or weight mutation."""
    path=directory/f'{branch}.json'
    if path.exists():
        return read_json(path)
    state_path=directory/f'{branch}_cursor.json'
    active=1 if stage == 'r1' else 2
    groups=[([list(range(width))] if branch == 'shared' else grouping[i] if branch.startswith('group')
             else [[j] for j in range(width)]) for i,width in enumerate((64,32))]
    kind='membrane' if branch.endswith('m0') else 'joint'
    schedule=[(iteration,layer,unit,ids) for iteration in range(PROTOCOL['coordinate_passes'])
              for layer in range(active) for unit,ids in enumerate(groups[layer])]
    if state_path.exists():
        state=read_json(state_path)
        if state['initial'] != [e.record() for e in initial]:
            raise ValueError('Calibration resume initial state changed')
        current=tuple(Channels.read(e) for e in state['current'])
        history=state['history']; next_step=state['next_step']; evaluated=state['evaluated']
    else:
        current=initial; history=[{'at':'initial','encoders':[e.record() for e in current],
                                 'sample':runner.assess(stage,steps,current)}]
        next_step=0; evaluated=1
    started=time.perf_counter()
    audit=directory/f'{branch}_candidates.jsonl'
    for position in range(next_step,len(schedule)):
        iteration,layer,unit,ids=schedule[position]
        winner=current; score=runner.assess(stage,steps,current)
        with audit.open('a',encoding='utf-8') as f:
            for encoder in proposals(current[layer],ids,kind,iteration):
                trial=list(current);trial[layer]=encoder;trial=tuple(trial)
                value=runner.assess(stage,steps,trial);evaluated+=1
                f.write(json.dumps({'coordinate':position,'layer':layer,'channels':ids,
                                    'encoders':[e.record() for e in trial],'metrics':value},allow_nan=False)+'\n')
                if objective(value,trial) < objective(score,winner):
                    winner,score=trial,value
        current=winner
        # Keep end-of-layer sweeps for a fully independent full-Main selection.
        if unit == len(groups[layer])-1:
            history.append({'at':f'pass{iteration+1}_layer{layer+1}',
                            'encoders':[e.record() for e in current],'sample':score})
        state={'initial':[e.record() for e in initial],'current':[e.record() for e in current],
               'history':history,'next_step':position+1,'evaluated':evaluated,'total_coordinates':len(schedule)}
        atomic_json(state_path,state)
        if unit % 8 == 0 or unit == len(groups[layer])-1:
            print(json.dumps({'phase':'fit','point':f'{stage}_t{steps}','branch':branch,
                'coordinate':position+1,'total':len(schedule),'sample_balanced_iou':score['balanced_iou'],
                'candidates':evaluated,'elapsed_this_session_s':time.perf_counter()-started}),flush=True)
    result={'stage':stage,'steps':steps,'branch':branch,'initial':[e.record() for e in initial],
            'final':[e.record() for e in current],'history':history,'evaluated':evaluated,
            'role':'train_main','weights_updated':False,'training_started':False}
    atomic_json(path,result)
    return result
