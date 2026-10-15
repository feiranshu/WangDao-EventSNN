"""Finite local m0 search with a sealed shortlist and no validation feedback."""
import itertools
import json
import numpy as np
import torch
from ..h2pro_channel_calibration_v1.search import Runner,prob,setup
from .common import (PROTOCOL,atomic_json,read_json,parameter_id,anchor,encoders,rank,
                     constraints,sha256,timestamp)


class FitRunner(Runner):
    def __init__(self,data,layers,device):
        super().__init__(data['features'],layers,device)
        np.testing.assert_array_equal(self.teacher.cpu().numpy(),data['teacher'])
        self.labels=torch.tensor(data['labels'],device=device).bool()
        self.hd=torch.tensor(data['hd'],device=device).bool()
        self.weight=torch.tensor(data['weights'],dtype=torch.float64,device=device)
        self.population=data['population']

    @torch.no_grad()
    def grouping(self):
        mean=(self.weight@self.first/self.weight.sum()).cpu().numpy()
        order=np.lexsort((np.arange(64),mean))
        return order.reshape(8,8).tolist(),mean.tolist()

    @torch.no_grad()
    def assess(self,e):
        raw,n1,_=self.forward('r1',16,e);pred=prob(raw)>=.35;vals=[]
        for mask in (torch.ones_like(self.hd),~self.hd,self.hd):
            weight=self.weight*mask
            vals.extend([(weight*(pred & self.labels)).sum(),(weight*(pred & ~self.labels)).sum()])
        vals.append((self.weight*n1).sum()*32)
        numbers=torch.stack(vals).cpu().tolist();out={}
        for i,g in enumerate(('overall','ordinary','high_density')):
            tp,fp=numbers[2*i:2*i+2];total=self.population[g]['targets'];fn=total-tp
            out[g]={'tp':tp,'fp':fp,'fn':fn,'iou':tp/max(tp+fp+fn,1.),
                    'precision':tp/max(tp+fp,1.),'recall':tp/max(total,1.)}
        out.update(parameter_id=parameter_id(e),cost={'equivalent_ac':numbers[-1]})
        return out


def fit(runner,groups,directory):
    marker=directory/'fit_selection.json'
    if marker.exists():
        saved=read_json(marker)
        for name,h in saved['fit_evidence_sha256'].items():
            if sha256(directory/name)!=h:raise ValueError('Sealed fit evidence changed')
        return saved
    folder=directory/'fit_candidates';folder.mkdir(parents=True,exist_ok=True)
    records={}

    def assess(changes):
        changes={str(k):int(v) for k,v in sorted(changes.items(),key=lambda item:int(item[0]))}
        e=encoders(changes,groups);pid=parameter_id(e);p=folder/f'{pid}.json'
        if p.exists():
            record=read_json(p)
            if record['changes']!=changes or record['encoders']!=[x.record() for x in e]:
                raise ValueError('Fit candidate resume mismatch')
        else:
            record={'changes':changes,'encoders':[x.record() for x in e],'metrics':runner.assess(e),
                    'data_scope':'main_fit_sample','weights_updated':False}
            atomic_json(p,record)
        records[pid]=record
        if len(records)%32==0:
            progress={'status':'fit_search','evaluated_unique_parameters':len(records),'training_started':False}
            atomic_json(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
        return record

    reference=assess({})['metrics'];singles=[]
    for group in range(8):
        for offset in PROTOCOL['m0_offsets']:
            singles.append(assess({group:191+offset}))
    options={}
    for group in range(8):
        options[group]=[]
        for sign in (-1,1):
            eligible=[r for r in singles if str(group) in r['changes'] and
                      (r['changes'][str(group)]-191)*sign>0]
            best=sorted(eligible,key=lambda r:rank(r['metrics'],reference,r['changes']))[:2]
            options[group].extend(r['changes'][str(group)] for r in best)
    for a,b in itertools.combinations(range(8),2):
        for va,vb in itertools.product(options[a],options[b]):
            assess({a:va,b:vb})
    finalists=[records[parameter_id(anchor())]]
    for count in (1,2):
        pool=[r for r in records.values() if len(r['changes'])==count]
        finalists+=sorted(pool,key=lambda r:rank(r['metrics'],reference,r['changes']))[:PROTOCOL['shortlist_per_group_count']]
    result={'status':'complete','role':'train_main','scope':'fit_sequences_only','created_at_utc':timestamp(),
            'groups':groups,'evaluated_unique_parameters':len(records),'pair_offsets':{str(k):v for k,v in options.items()},
            'reference':reference,'candidates':finalists,'internal_validation_used_for_search':False,
            'fit_evidence_sha256':{p.relative_to(directory).as_posix():sha256(p) for p in sorted(folder.glob('*.json'))}}
    atomic_json(marker,result)
    return result
