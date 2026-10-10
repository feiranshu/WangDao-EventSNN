"""One locked, append-only reserved-role acceptance of the frozen causal teacher."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from run_stage1 import (ROOT, DATA, INDEX, PROPOSAL, SOURCE_INT8, sha256,
                        _ap, object_window_continuity)
from src.stream_state import CausalPipeline
from wangdao.a4_2_runtime import A42Pipeline
from wangdao.metrics import event_metrics, localization_metrics

FROZEN = ROOT/'frozen'/'H2-pro-Causal-ANN-v1'
LOCK = FROZEN/'acceptance_lock.json'
PROTOCOL = ROOT/'acceptance_protocol.json'


def lock():
    if LOCK.exists():
        raise FileExistsError(LOCK)
    manifest = json.loads((FROZEN/'model_manifest.json').read_text(encoding='utf-8'))
    if manifest['metadata']['variant'] != 'C1-proposal-motion' or manifest['metadata']['threshold'] != 0.35:
        raise ValueError('frozen candidate identity mismatch')
    files = [PROTOCOL, ROOT/'protocol.json', ROOT/'selection_protocol_v2.json',
             ROOT/'acceptance_once.py', ROOT/'src'/'stream_state.py',
             ROOT/'src'/'causal_feature_builder.py', ROOT/'src'/'motion_coherence.py',
             INDEX, PROPOSAL, SOURCE_INT8, FROZEN/'teacher_float.pt',
             FROZEN/'teacher_exact_int8.npz', FROZEN/'model_manifest.json']
    digests = {str(path.absolute()):sha256(path) for path in files}
    if digests[str((FROZEN/'teacher_exact_int8.npz').absolute())] != manifest['teacher_int8_sha256']:
        raise ValueError('teacher hash changed')
    payload = {'protocol_id':'H2-pro-stage1-frozen-acceptance-v1',
               'candidate':manifest['metadata'], 'digests':digests,
               'index_roles':['validation','test'],
               'result_state':'locked before reserved-role evaluation'}
    LOCK.write_text(json.dumps(payload,indent=2),encoding='utf-8')
    return payload


def verify_lock():
    payload = json.loads(LOCK.read_text(encoding='utf-8'))
    for name,digest in payload['digests'].items():
        if sha256(Path(name)) != digest:
            raise ValueError('locked file changed: '+name)
    return payload


def summarize(rows, probabilities, labels, threshold):
    all_p = np.concatenate(probabilities)
    all_y = np.concatenate(labels)
    tp = sum(r['event']['true_positive'] for r in rows)
    fp = sum(r['event']['false_positive'] for r in rows)
    fn = sum(r['event']['false_negative'] for r in rows)
    tn = sum(r['event']['true_negative'] for r in rows)
    targets = sum(r['localization']['targets'] for r in rows)
    correct = sum(r['localization']['correct_targets'] for r in rows)
    false = sum(r['localization']['false_components'] for r in rows)
    frames = sum(r['localization']['frame_denominator'] for r in rows)
    eligible = sum(r['continuity']['eligible_object_windows'] for r in rows)
    detected = sum(r['continuity']['detected_object_windows'] for r in rows)
    density = {}
    for key,group in [('high',[r for r in rows if r['high_density']]),
                      ('ordinary',[r for r in rows if not r['high_density']])]:
        gtp = sum(r['event']['true_positive'] for r in group)
        gfp = sum(r['event']['false_positive'] for r in group)
        gfn = sum(r['event']['false_negative'] for r in group)
        gt = sum(r['localization']['targets'] for r in group)
        gc = sum(r['localization']['correct_targets'] for r in group)
        gf = sum(r['localization']['false_components'] for r in group)
        gd = sum(r['localization']['frame_denominator'] for r in group)
        density[key] = {'sequences':len(group),'iou':gtp/max(gtp+gfp+gfn,1),
                        'recall':gtp/max(gtp+gfn,1),'pd':gc/max(gt,1),
                        'fa':gf/max(gd*346*260,1)}
    return {'threshold':threshold,'events':int(len(all_y)),
            'ap':_ap(all_y,all_p),'event':{'true_positive':tp,'false_positive':fp,
                'false_negative':fn,'true_negative':tn,'iou':tp/max(tp+fp+fn,1),
                'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),
                'f1':2*tp/max(2*tp+fp+fn,1)},
            'localization':{'pd':correct/max(targets,1),
                            'fa':false/max(frames*346*260,1),
                            'targets':targets,'false_components':false,
                            'frame_denominator':frames},
            'density_strata':density,
            'continuity':{'eligible_object_windows':eligible,
                          'detected_object_windows':detected,
                          'coverage':detected/max(eligible,1),
                          'breaks':sum(r['continuity']['breaks'] for r in rows)}}


def evaluate(role: str, output: Path):
    if role not in ('validation','test'):
        raise ValueError('acceptance role must be Validation or Test')
    if output.exists():
        raise FileExistsError(output)
    locked = verify_lock()
    index = json.loads(INDEX.read_text(encoding='utf-8'))
    file_hashes = {Path(item['path']).name:item['sha256'] for item in index['files']}
    baseline = A42Pipeline(PROPOSAL,SOURCE_INT8)
    causal = CausalPipeline(PROPOSAL,FROZEN/'teacher_exact_int8.npz',
                            'C1-proposal-motion',0.35,initial_refiner=False)
    rows = {'baseline':[],'causal':[]}
    probabilities = {'baseline':[],'causal':[]}
    labels = []
    for name in index['roles'][role]:
        path = DATA/('val' if role=='validation' else 'test')/name
        if sha256(path) != file_hashes[name]:
            raise ValueError('reserved sequence hash mismatch: '+name)
        with np.load(path,allow_pickle=False) as archive:
            full = archive['ev'].copy()
        t0 = float(full['t'][0])
        count = int((float(full['t'][-1])-t0)//100)
        boundary = t0+count*100
        closed = full[:int(np.searchsorted(full['t'],boundary,side='left'))]
        base = baseline.infer(closed)
        base_p = base['probabilities']
        causal.reset(t0)
        causal.push(closed,boundary)
        windows = causal.flush_completed_windows()
        _,leftover = causal.finalize()
        if leftover or len(windows)!=count or sum(len(w.event_indices) for w in windows)!=len(closed):
            raise AssertionError('causal closed-prefix coverage mismatch')
        causal_p = np.concatenate([w.probabilities for w in windows])
        row_data = {'labels':closed['label'],'identities':closed['name'],
                    't':closed['t'],'t0':t0}
        high = len(full)/max((float(full['t'][-1])-t0)/1000,1e-9)>=20000
        for key,p,threshold in [('baseline',base_p,0.4),('causal',causal_p,0.35)]:
            rows[key].append({'name':name,'high_density':high,
                              'event':event_metrics(closed['label'],p,threshold),
                              'localization':localization_metrics(
                                  closed['t'],closed['x'],closed['y'],closed['label'],
                                  closed['name'],p,threshold),
                              'continuity':object_window_continuity(row_data,p,threshold),
                              'closed_windows':count,'closed_events':len(closed),
                              'trailing_events':len(full)-len(closed)})
            probabilities[key].append(p)
        labels.append(closed['label'])
        print(json.dumps({'role':role,'sequence':name,'closed_windows':count,
                          'closed_events':len(closed)}),flush=True)
    metrics = {key:summarize(rows[key],probabilities[key],labels,
                             0.4 if key=='baseline' else 0.35) for key in rows}
    result = {'role':role,'lock_sha256':sha256(LOCK),
              'index_sha256':locked['digests'][str(INDEX.absolute())],
              'reference':'same complete-window prefix per sequence',
              'baseline':metrics['baseline'],'causal':metrics['causal'],
              'delta':{'iou':metrics['causal']['event']['iou']-metrics['baseline']['event']['iou'],
                       'ap':metrics['causal']['ap']-metrics['baseline']['ap'],
                       'high_density_recall':metrics['causal']['density_strata']['high']['recall']-
                           metrics['baseline']['density_strata']['high']['recall']},
              'sequences':{key:rows[key] for key in rows}}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('lock','validation','test'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.mode=='lock':
        print(json.dumps({'lock':str(LOCK),'sha256_files':len(lock()['digests'])}))
    else:
        output=args.output or ROOT/'reports'/('acceptance_'+args.mode+'.json')
        result=evaluate(args.mode,output)
        print(json.dumps({'role':args.mode,'output':str(output),'delta':result['delta']}))
