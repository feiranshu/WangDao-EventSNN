"""Paired IoU uncertainty from frozen Step7 scores and frozen H2-pro acceptance."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from compare_ab_same_prefix import ROOT, _source_scores


def run(role, repeats=10000):
    ab=json.loads((ROOT/'reports'/('ab_same_prefix_'+role+'_v2.json')).read_text(encoding='utf-8'))
    prior=json.loads((ROOT/'reports'/('acceptance_'+role+'.json')).read_text(encoding='utf-8'))
    causal=prior['sequences']['causal']
    scores={model:_source_scores(role,model)[0] for model in ('H2','H2-plus')}
    counts={'H2-pro':[], 'H2':[], 'H2-plus':[]}
    high=[]
    for row in causal:
        name=row['name']
        n=row['closed_events']
        high.append(row['high_density'])
        metric=row['event']
        counts['H2-pro'].append([metric['true_positive'],metric['false_positive'],metric['false_negative']])
        for model,threshold in [('H2',.30),('H2-plus',.40)]:
            labels,prob=scores[model][name]
            labels=labels[:n].astype(bool)
            pred=prob[:n]>=threshold
            counts[model].append([np.count_nonzero(pred & labels),
                                  np.count_nonzero(pred & ~labels),
                                  np.count_nonzero(~pred & labels)])
    arrays={model:np.asarray(rows,dtype=np.float64) for model,rows in counts.items()}
    for model,array in arrays.items():
        total=array.sum(axis=0)
        iou=total[0]/total.sum()
        if abs(iou-ab['A'][model]['event']['iou'])>1e-12:
            raise AssertionError('metric population mismatch: '+model)
    pools=[np.flatnonzero(~np.asarray(high,dtype=bool)),np.flatnonzero(high)]
    rng=np.random.default_rng(20261013)
    distribution={model:[] for model in ('H2','H2-plus')}
    for _ in range(repeats):
        chosen=np.concatenate([rng.choice(pool,size=len(pool),replace=True)
                               for pool in pools if len(pool)])
        new=arrays['H2-pro'][chosen].sum(axis=0)
        new_iou=new[0]/new.sum()
        for model in distribution:
            old=arrays[model][chosen].sum(axis=0)
            distribution[model].append(new_iou-old[0]/old.sum())
    return {'role':role,'repeats':repeats,
            'method':'sequence-level paired bootstrap stratified by >=20000 events/s',
            'delta_pro_minus_reference':{
                model:{'point':ab['A']['H2-pro']['event']['iou']-ab['A'][model]['event']['iou'],
                       '95_interval':[float(x) for x in np.quantile(distribution[model],[.025,.5,.975])]}
                for model in distribution}}


if __name__=='__main__':
    output=ROOT/'reports'/'ab_same_prefix_uncertainty.json'
    if output.exists():
        raise FileExistsError(output)
    result={role:run(role) for role in ('validation','test')}
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))
