"""Train-main-only C2 separability, density, polarity and movement diagnostics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from run_stage1 import _records, INDEX, sha256, ROOT


def describe(values):
    if not len(values):
        return {'n':0,'mean':None,'p10':None,'median':None,'p90':None}
    return {'n':int(len(values)),'mean':float(np.mean(values)),
            'p10':float(np.quantile(values,0.1)),'median':float(np.median(values)),
            'p90':float(np.quantile(values,0.9))}


def run(cache: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    records = _records(cache,'train_main','C2-LMEC')
    strata = {key:[] for key in ('positive','negative','high_positive','high_negative',
                                 'polarity_0_positive','polarity_1_positive','sparse_positive')}
    motion = []
    for row in records:
        labels = row['labels'][row['indices']].astype(bool)
        values = row['features'][:,2]
        rate = row['full_event_rate']
        strata['positive'].append(values[labels])
        strata['negative'].append(values[~labels])
        if rate>=20000:
            strata['high_positive'].append(values[labels])
            strata['high_negative'].append(values[~labels])
        p = row['features'][:,3]
        strata['polarity_0_positive'].append(values[labels & (p==0)])
        strata['polarity_1_positive'].append(values[labels & (p==255)])
        ids = row['identities']
        t = row['t'].astype(np.float64)
        windows = np.floor((t-float(row['t0']))/100).astype(np.int64)
        sparse = np.zeros(len(t),dtype=bool)
        for identity in np.unique(ids[row['labels'].astype(bool)]):
            previous = None
            for window in np.unique(windows[(ids==identity)&row['labels'].astype(bool)]):
                target = (ids==identity)&row['labels'].astype(bool)&(windows==window)
                sparse[target] = target.sum()<=10
                center = (float(np.mean(row['x'][target])),float(np.mean(row['y'][target])))
                if previous is not None and previous[0]+1==window:
                    motion.append(float(np.hypot(center[0]-previous[1],center[1]-previous[2])))
                previous = (int(window),*center)
        strata['sparse_positive'].append(values[labels & sparse[row['indices']]])
    report = {'role':'train_main','index_sha256':sha256(INDEX),
              'lmec':{key:describe(np.concatenate(parts) if parts else np.empty(0))
                      for key,parts in strata.items()},
              'target_adjacent_100ms_displacement_pixels':describe(np.asarray(motion)),
              'interpretation':'Descriptive only; no target ID or label enters the inference feature.'}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=ROOT/'cache')
    parser.add_argument('--output',type=Path,default=ROOT/'reports'/'feature_diagnostics.json')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps(run(args.cache,args.output)))
