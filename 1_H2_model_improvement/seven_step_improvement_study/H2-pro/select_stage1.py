"""Deterministic Train-dev summary; never opens reserved roles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from run_stage1 import PROTOCOL, VARIANTS, sha256, ROOT


def _candidates(run: Path, variant: str):
    rows = json.loads((run/variant/'progress.json').read_text(encoding='utf-8'))
    for row in rows:
        if 'int8' not in row:
            continue
        for threshold, item in row['int8']['curve'].items():
            yield {'variant':variant,'epoch':row['epoch'],'threshold':float(threshold),
                   'threshold_key':threshold,'metrics':item,'ap':row['int8']['ap'],
                   'checkpoint_sha256':row['checkpoint_sha256'],
                   'int8_sha256':row['int8_sha256']}


def paired_bootstrap(run: Path, variant: str, winner: dict, repeats: int = 10000):
    if variant == 'C0-zero' or winner is None:
        return None
    threshold = winner['threshold_key']
    epoch = winner['epoch']
    def rows(name):
        progress = json.loads((run/name/'progress.json').read_text(encoding='utf-8'))
        return next(item['int8']['sequences'] for item in progress if item['epoch']==epoch)
    reference, treatment = rows('C0-zero'),rows(variant)
    if [row['name'] for row in reference] != [row['name'] for row in treatment]:
        raise ValueError('paired sequence order mismatch')
    high = np.asarray([row['high_density'] for row in reference],dtype=bool)
    pools = [np.flatnonzero(~high),np.flatnonzero(high)]
    rng = np.random.default_rng(20261010)
    delta_iou, delta_high_recall = [], []
    def counts(records, sample):
        metrics = [records[i][threshold]['event'] for i in sample]
        return np.asarray([sum(row[key] for row in metrics) for key in
                           ('true_positive','false_positive','false_negative')],dtype=np.float64)
    for _ in range(repeats):
        sample = np.concatenate([rng.choice(pool,size=len(pool),replace=True)
                                 for pool in pools if len(pool)])
        a,b = counts(reference,sample),counts(treatment,sample)
        delta_iou.append(b[0]/max(b.sum(),1)-a[0]/max(a.sum(),1))
        high_sample = sample[high[sample]]
        if len(high_sample):
            a,b = counts(reference,high_sample),counts(treatment,high_sample)
            delta_high_recall.append(b[0]/max(b[0]+b[2],1)-a[0]/max(a[0]+a[2],1))
    def interval(values):
        return [float(x) for x in np.quantile(values,[0.025,0.5,0.975])] if values else None
    return {'resamples':repeats,'method':'sequence-level paired bootstrap stratified by >=20000 events/s',
            'iou_delta_95_interval':interval(delta_iou),
            'high_density_recall_delta_95_interval':interval(delta_high_recall)}


def select(run: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    rows = {variant:list(_candidates(run,variant)) for variant in VARIANTS}
    c0 = {(row['epoch'],row['threshold_key']):row for row in rows['C0-zero']}
    selected = {}
    for variant in VARIANTS:
        eligible = []
        audit = []
        for row in rows[variant]:
            anchor = c0[(row['epoch'],row['threshold_key'])]
            m, a = row['metrics'],anchor['metrics']
            pd,fa = m['localization']['pd'],m['localization']['fa']
            passes = (variant == 'C0-zero' or
                      (pd >= a['localization']['pd']-0.005 and
                       fa <= a['localization']['fa']*1.10))
            delta = {'iou':m['event']['iou']-a['event']['iou'],
                     'ap':row['ap']-anchor['ap'],
                     'high_density_recall':None if m['high_density_recall'] is None or
                     a['high_density_recall'] is None else
                     m['high_density_recall']-a['high_density_recall'],
                     'object_window_coverage':None if m['object_window_coverage'] is None or
                     a['object_window_coverage'] is None else
                     m['object_window_coverage']-a['object_window_coverage']}
            audit.append({'epoch':row['epoch'],'threshold':row['threshold'],
                          'pd_fa_guard':passes,'delta_from_C0_same_point':delta})
            if passes:
                eligible.append((m['event']['iou'],row['ap'],
                                 m['high_density_recall'] if m['high_density_recall'] is not None else -1,
                                 -fa,-row['epoch'],-row['threshold'],row,delta))
        winner = max(eligible,key=lambda value:value[:6]) if eligible else None
        selected[variant] = {
            'eligible_count':len(eligible),'winner':winner[6] if winner else None,
            'delta_from_C0_same_point':winner[7] if winner else None,
            'paired_bootstrap':paired_bootstrap(run,variant,winner[6] if winner else None),
            'fixed_0_40':[row for row in rows[variant] if row['threshold']==0.4],
            'audit':audit}
    decision = {'source':'Train-dev only','protocol_sha256':sha256(ROOT/'protocol.json'),
                'candidates':selected,'frozen_teacher':None,
                'note':'Candidate ranking is a development result. A teacher requires uncertainty, '
                       'continuity and cost review plus a separately frozen acceptance protocol.'}
    output.write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding='utf-8')
    return decision


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('--output',type=Path)
    a = p.parse_args()
    target = a.output or a.run/'train_dev_selection.json'
    result = select(a.run,target)
    print(json.dumps({v:{'epoch':d['winner']['epoch'] if d['winner'] else None,
                         'threshold':d['winner']['threshold'] if d['winner'] else None}
                      for v,d in result['candidates'].items()}))
