"""Descriptive paired sequence bootstrap after the frozen acceptance run."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).absolute().parent


def summarize(role: str, repeats: int = 10000):
    source = ROOT/'reports'/('acceptance_'+role+'.json')
    data = json.loads(source.read_text(encoding='utf-8'))
    old, new = data['sequences']['baseline'], data['sequences']['causal']
    if [r['name'] for r in old] != [r['name'] for r in new]:
        raise ValueError('sequence pairing differs')
    high = np.asarray([r['high_density'] for r in old],dtype=bool)
    pools = [np.flatnonzero(~high),np.flatnonzero(high)]
    rng = np.random.default_rng(20261012)
    deltas = []
    def iou(rows,chosen):
        tp=sum(rows[i]['event']['true_positive'] for i in chosen)
        fp=sum(rows[i]['event']['false_positive'] for i in chosen)
        fn=sum(rows[i]['event']['false_negative'] for i in chosen)
        return tp/max(tp+fp+fn,1)
    for _ in range(repeats):
        chosen=np.concatenate([rng.choice(pool,size=len(pool),replace=True)
                               for pool in pools if len(pool)])
        deltas.append(iou(new,chosen)-iou(old,chosen))
    per_sequence = [{'name':b['name'],'high_density':b['high_density'],
                     'delta_iou':b['event']['iou']-a['event']['iou'],
                     'delta_recall':b['event']['recall']-a['event']['recall']}
                    for a,b in zip(old,new)]
    return {'role':role,'repeats':repeats,
            'method':'paired, density-stratified sequence bootstrap; descriptive only',
            'iou_delta_95_interval':[float(v) for v in np.quantile(deltas,[.025,.5,.975])],
            'sequences_improved_iou':sum(x['delta_iou']>0 for x in per_sequence),
            'sequences_worsened_iou':sum(x['delta_iou']<0 for x in per_sequence),
            'worst_sequence_iou':sorted(per_sequence,key=lambda x:x['delta_iou'])[:5],
            'best_sequence_iou':sorted(per_sequence,key=lambda x:x['delta_iou'])[-5:]}


if __name__=='__main__':
    output=ROOT/'reports'/'acceptance_uncertainty.json'
    if output.exists():
        raise FileExistsError(output)
    report={role:summarize(role) for role in ('validation','test')}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({role:report[role]['iou_delta_95_interval'] for role in report}))
