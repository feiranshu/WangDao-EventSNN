"""Apply the documented post-v1 high-density protection amendment to Train-dev."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from run_stage1 import ROOT, INDEX, VARIANTS, sha256
from select_stage1 import _candidates, paired_bootstrap

RULE = ROOT/'selection_protocol_v2.json'
HISTORICAL = ROOT/'reports'/'historical_A4-2_train_dev.json'


def _historical_reference(path: Path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data['role'] != 'train_dev' or data['threshold'] != 0.4 or data['model_id'] != 'A4-2-frozen-historical-noncausal':
        raise ValueError('historical reference identity mismatch')
    high = [row for row in data['sequences'] if row['high_density']]
    ordinary = [row for row in data['sequences'] if not row['high_density']]
    def fa(group):
        false = sum(row['localization']['false_components'] for row in group)
        frames = sum(row['localization']['frame_denominator'] for row in group)
        return false/(frames*346*260) if frames else 0.0
    tp = sum(row['event']['true_positive'] for row in high)
    fn = sum(row['event']['false_negative'] for row in high)
    eligible = sum(row['continuity']['eligible_object_windows'] for row in data['sequences'])
    detected = sum(row['continuity']['detected_object_windows'] for row in data['sequences'])
    return {'iou':data['event']['iou'],'ap':data['ap'],
            'pd':data['localization']['pd'],'fa':data['localization']['fa'],
            'high_density_recall':tp/(tp+fn) if tp+fn else None,
            'high_density_fa':fa(high),'ordinary_density_fa':fa(ordinary),
            'object_window_coverage':detected/eligible if eligible else None}


def _paired_historical_bootstrap(run: Path, variant: str, winner: dict,
                                 historical_path: Path, repeats=10000):
    old = json.loads(historical_path.read_text(encoding='utf-8'))['sequences']
    progress = json.loads((run/variant/'progress.json').read_text(encoding='utf-8'))
    new = next(row['int8']['sequences'] for row in progress if row['epoch']==winner['epoch'])
    if [r['name'] for r in old] != [r['name'] for r in new]:
        raise ValueError('old/new sequence order mismatch')
    high = np.asarray([row['high_density'] for row in old],dtype=bool)
    pools = [np.flatnonzero(~high),np.flatnonzero(high)]
    rng = np.random.default_rng(20261011)
    diffs = []
    threshold = winner['threshold_key']
    def iou(rows,sample,old_format):
        metrics = [rows[i]['event'] if old_format else rows[i][threshold]['event'] for i in sample]
        tp = sum(row['true_positive'] for row in metrics)
        fp = sum(row['false_positive'] for row in metrics)
        fn = sum(row['false_negative'] for row in metrics)
        return tp/max(tp+fp+fn,1)
    for _ in range(repeats):
        sample = np.concatenate([rng.choice(pool,size=len(pool),replace=True)
                                 for pool in pools if len(pool)])
        diffs.append(iou(new,sample,False)-iou(old,sample,True))
    return {'resamples':repeats,'iou_delta_vs_historical_95_interval':
            [float(v) for v in np.quantile(diffs,[0.025,0.5,0.975])]}


def select(run: Path, historical: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    rule = json.loads(RULE.read_text(encoding='utf-8'))
    reference = _historical_reference(historical)
    selected = {}
    for variant in VARIANTS:
        rows = list(_candidates(run,variant))
        eligible = []
        audit = []
        for row in rows:
            m = row['metrics']
            checks = {
                'overall_iou':m['event']['iou']>=reference['iou'],
                'overall_ap':row['ap']>=reference['ap'],
                'overall_pd':m['localization']['pd']>=reference['pd']-0.005,
                'overall_fa':m['localization']['fa']<=reference['fa']*1.10,
                'high_density_recall':m['high_density_recall'] is not None and
                    m['high_density_recall']>=reference['high_density_recall']-0.003,
                'high_density_fa':m['density_strata']['high']['fa']<=reference['high_density_fa']*1.10,
                'ordinary_density_fa':m['density_strata']['ordinary']['fa']<=reference['ordinary_density_fa']*1.10,
                'object_window_coverage':m['object_window_coverage'] is not None and
                    m['object_window_coverage']>=reference['object_window_coverage']-0.005,
            }
            audit.append({'epoch':row['epoch'],'threshold':row['threshold'],
                          'checks':checks,'eligible':all(checks.values())})
            if all(checks.values()):
                eligible.append((m['event']['iou'],row['ap'],m['high_density_recall'],
                                 -m['localization']['fa'],-row['epoch'],-row['threshold'],row))
        winner = max(eligible,key=lambda x:x[:6])[6] if eligible else None
        selected[variant] = {'eligible_count':len(eligible),'winner':winner,'audit':audit,
            'paired_bootstrap_vs_C0':paired_bootstrap(run,variant,winner) if winner else None,
            'paired_bootstrap_vs_historical':
                _paired_historical_bootstrap(run,variant,winner,historical) if winner else None}
    decision = {'source':'Train-dev only; v2 is a post-v1 amendment',
                'selection_protocol_sha256':sha256(RULE),
                'protocol_sha256':sha256(ROOT/'protocol.json'),
                'index_sha256':sha256(INDEX),
                'historical_reference_sha256':sha256(historical),
                'historical_reference':reference,'candidates':selected,
                'frozen_teacher':None,'status':'development selection; reserved-data acceptance pending',
                'interpretation_limit':rule['interpretation_limit']}
    output.write_text(json.dumps(decision,indent=2),encoding='utf-8')
    return decision


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('--historical',type=Path,default=HISTORICAL)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    output=a.output or a.run/'train_dev_selection_v2.json'
    result=select(a.run,a.historical,output)
    print(json.dumps({v:{'epoch':d['winner']['epoch'] if d['winner'] else None,
                         'threshold':d['winner']['threshold'] if d['winner'] else None,
                         'eligible_count':d['eligible_count']}
                      for v,d in result['candidates'].items()}))
