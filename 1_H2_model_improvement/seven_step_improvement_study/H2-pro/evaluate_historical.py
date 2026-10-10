"""Train-dev-only old A4-2 reference on the same complete-window event prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from run_stage1 import (ROOT, DATA, PROPOSAL, SOURCE_INT8, _records, _ap,
                        object_window_continuity, sha256)
from wangdao.metrics import event_metrics, localization_metrics


def run(cache: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    from wangdao.a4_2_runtime import A42Pipeline
    reference = A42Pipeline(PROPOSAL,SOURCE_INT8)
    records = _records(cache,'train_dev','C0-zero')
    labels, probabilities, rows = [], [], []
    for row in records:
        with np.load(DATA/'train'/row['name'],allow_pickle=False) as archive:
            full = archive['ev'].copy()
        closed = full[:len(row['labels'])]
        result = reference.infer(closed)
        prob = result['probabilities']
        label = closed['label']
        event = event_metrics(label,prob,0.4)
        loc = localization_metrics(closed['t'],closed['x'],closed['y'],label,
                                   closed['name'],prob,0.4)
        rows.append({'name':row['name'],'high_density':row['full_event_rate']>=20000,
                     'event':event,'localization':loc,
                     'continuity':object_window_continuity(row,prob,0.4),
                     'candidate_events':int(len(result['candidate_indices']))})
        labels.append(label)
        probabilities.append(prob)
        print(json.dumps({'sequence':row['name'],'events':len(closed),
                          'candidate_events':len(result['candidate_indices'])}),flush=True)
    labels, probabilities = np.concatenate(labels),np.concatenate(probabilities)
    total = event_metrics(labels,probabilities,0.4)
    targets = sum(row['localization']['targets'] for row in rows)
    frames = sum(row['localization']['frame_denominator'] for row in rows)
    false = sum(row['localization']['false_components'] for row in rows)
    output.parent.mkdir(parents=True,exist_ok=True)
    report = {'role':'train_dev','model_id':'A4-2-frozen-historical-noncausal',
              'source_int8_sha256':sha256(SOURCE_INT8),
              'threshold':0.4,'evaluation_subset':'same complete-100ms event prefix as H2-pro',
              'not_independent_test':True,
              'ap':_ap(labels,probabilities),'event':total,
              'localization':{'pd':sum(row['localization']['correct_targets'] for row in rows)/targets
                              if targets else 1.0,
                              'fa':false/(frames*346*260) if frames else 0.0},
              'sequences':rows}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,default=ROOT/'cache')
    p.add_argument('--output',type=Path,default=ROOT/'reports'/'historical_A4-2_train_dev.json')
    a=p.parse_args()
    print(json.dumps({k:v for k,v in run(a.cache,a.output).items() if k!='sequences'}))
