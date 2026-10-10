"""Post-freeze, read-only A/B comparison on the H2-pro closed-window prefix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from acceptance_once import FROZEN, LOCK, summarize
from verify_migrated_lock import verify_migrated_lock as verify_lock
from run_stage1 import ROOT, DATA, INDEX, PROPOSAL, SOURCE_INT8, sha256, object_window_continuity
from src.stream_state import CausalPipeline
from wangdao.a4_2_runtime import load_a4_2_refiner
from wangdao.h2.numpy_model import load_export
from wangdao.h2.representation import _fine_counts_to_patch_features
from wangdao.h2_research import FrozenH2Pipeline, augment_refiner_features
from wangdao.metrics import event_metrics, localization_metrics
from wangdao.target_tracking_metrics import (combine_hota_raw, evaluate_sequence,
                                             hota_metrics, percentile95)

HISTORICAL_B = ROOT.parent/'step7'/'intermediate_H2-plus'/'B_model_complexity'
STEP7 = ROOT.parent/'step7'/'step7_results_20261005'
MODELS = {'H2':0.30,'H2-plus':0.40,'H2-pro':0.35}


def _profile(values, layers):
    value = np.ascontiguousarray(values,dtype=np.uint8)
    dense = effective = 0
    for layer,(weight,scale) in enumerate(layers):
        dense += len(value)*weight.size
        active = np.count_nonzero(value,axis=0).astype(np.int64)
        fanout = np.count_nonzero(weight,axis=1).astype(np.int64)
        effective += int(active @ fanout)
        if layer+1 < len(layers):
            raw = np.matmul(value.astype(np.int32),weight.astype(np.int32))
            floating = raw.astype(np.float64)*scale[None,:].astype(np.float64)/255.0
            value = np.clip(np.rint(np.clip(floating,0,1)*255),0,255).astype(np.uint8)
    return {'dense_macs':int(dense),'effective_macs':int(effective)}


def _static(proposal, refiner, model):
    layers=tuple(proposal)+tuple(refiner)
    return {'B01_parameters':sum(weight.size for weight,_ in layers),
            'B02_tensor_payload_bytes':sum(weight.nbytes+scale.nbytes for weight,scale in layers),
            'persistent_proposal_history_bytes':2*34*44*4 if model!='H2' else 0}


def _tail_proposal_effective(full, t0, closed_windows, layers):
    effective=0
    for window in range(closed_windows,80):
        left=t0+window*100.0
        right=left+100.0
        start=int(np.searchsorted(full['t'],left,side='left'))
        end=int(np.searchsorted(full['t'],right,side='left'))
        piece=full[start:end]
        if not len(piece):
            continue
        fine=np.zeros((1,2,68,88),dtype=np.int32)
        np.add.at(fine[0],(piece['p'],piece['y']//4,piece['x']//4),1)
        patch=_fine_counts_to_patch_features(fine)
        effective+=_profile(patch.reshape(-1,128),layers)['effective_macs']
    return int(effective)


def _source_scores(role, model):
    stem='H2' if model=='H2' else 'A4-2-int8'
    path=STEP7/role/'bootstrap_scores'/(stem+'.npz')
    with np.load(path,allow_pickle=False) as archive:
        names=archive['sequence_files'].tolist()
        seq=archive['sequence'].copy()
        labels=archive['labels'].copy()
        probs=archive['probability'].copy()
    return {name:(labels[seq==i],probs[seq==i]) for i,name in enumerate(names)},sha256(path)


def run(role, output):
    if role not in ('validation','test'):
        raise ValueError(role)
    if output.exists():
        raise FileExistsError(output)
    verify_lock()
    index=json.loads(INDEX.read_text(encoding='utf-8'))
    lookup={Path(item['path']).name:item['sha256'] for item in index['files']}
    h2_export=load_export(PROPOSAL)
    plus_export=load_a4_2_refiner(SOURCE_INT8)
    old_pipeline=FrozenH2Pipeline(PROPOSAL)
    new_pipeline=CausalPipeline(PROPOSAL,FROZEN/'teacher_exact_int8.npz',
                                'C1-proposal-motion',0.35,initial_refiner=False)
    layers={'H2':h2_export.refiner_layers,
            'H2-plus':plus_export.layers,
            'H2-pro':new_pipeline.refiner_layers}
    scores={}
    score_hashes={}
    for model in ('H2','H2-plus'):
        scores[model],score_hashes[model]=_source_scores(role,model)
    old_b={}
    for model,identity in [('H2','h2__former_full_sequence'),
                           ('H2-plus','h2_plus__software_exact_int8')]:
        source=HISTORICAL_B/identity/role/'summary.json'
        record=json.loads(source.read_text(encoding='utf-8'))
        old_b[model]={row['sequence']:row for row in record['per_sequence']}
    rows={model:[] for model in MODELS}
    probabilities={model:[] for model in MODELS}
    labels=[]
    b_rows={model:[] for model in MODELS}
    track_raw={model:[] for model in MODELS}
    track_misses={model:[] for model in MODELS}
    for name in index['roles'][role]:
        path=DATA/('val' if role=='validation' else 'test')/name
        if sha256(path)!=lookup[name]:
            raise ValueError('sequence hash mismatch: '+name)
        with np.load(path,allow_pickle=False) as archive:
            full=archive['ev'].copy()
        t0=float(full['t'][0])
        count=int((float(full['t'][-1])-t0)//100)
        closed_count=int(np.searchsorted(full['t'],t0+count*100,side='left'))
        closed=full[:closed_count]
        new_pipeline.reset(t0)
        new_pipeline.push(closed,t0+count*100)
        windows=new_pipeline.flush_completed_windows()
        _,leftover=new_pipeline.finalize()
        if leftover or len(windows)!=count:
            raise AssertionError('closed window mismatch: '+name)
        new_p=np.concatenate([w.probabilities for w in windows])
        new_features=np.concatenate([w.features for w in windows],axis=0)
        new_candidates=np.concatenate([w.event_indices[w.candidate_mask] for w in windows])
        if len(new_p)!=closed_count:
            raise AssertionError('closed event mismatch: '+name)
        old_prepared=old_pipeline.prepare(full)
        old_candidate_mask=old_prepared.indices<closed_count
        old_candidates=old_prepared.indices[old_candidate_mask]
        if not np.array_equal(old_candidates,new_candidates):
            raise AssertionError('old/new Proposal routing differs: '+name)
        old_features={'H2':old_prepared.features[old_candidate_mask],
                      'H2-plus':augment_refiner_features(old_prepared.features,
                                                       old_prepared.history)[old_candidate_mask]}
        tail_features={'H2':old_prepared.features[~old_candidate_mask],
                       'H2-plus':augment_refiner_features(old_prepared.features,
                                                        old_prepared.history)[~old_candidate_mask]}
        source_b=old_b['H2'][name]
        if source_b['candidates']!=len(old_prepared.indices):
            raise AssertionError('formal B source candidate mismatch: '+name)
        tail_proposal=_tail_proposal_effective(full,t0,count,h2_export.proposal_layers)
        proposal_effective=source_b['proposal_effective_macs']-tail_proposal
        if proposal_effective<0:
            raise AssertionError('negative Proposal effective count')
        proposal_dense=count*336*7680
        high=len(full)/max((float(full['t'][-1])-t0)/1000,1e-9)>=20000
        row_data={'labels':closed['label'],'identities':closed['name'],'t':closed['t'],'t0':t0}
        for model in MODELS:
            if model=='H2-pro':
                p=new_p
                features=new_features
                ref=_profile(features,layers[model])
            else:
                original_label,original_p=scores[model][name]
                if len(original_label)!=len(full) or not np.array_equal(original_label[:closed_count],closed['label']):
                    raise AssertionError('formal A source alignment differs: '+name)
                p=original_p[:closed_count]
                features=old_features[model]
                full_b=old_b[model][name]
                tail=_profile(tail_features[model],layers[model])
                ref={'dense_macs':full_b['refiner_dense_macs']-tail['dense_macs'],
                     'effective_macs':full_b['refiner_effective_macs']-tail['effective_macs']}
                if ref['dense_macs']!=len(features)*sum(w.size for w,_ in layers[model]):
                    raise AssertionError('formal B refiner dense mismatch: '+name)
            threshold=MODELS[model]
            event=event_metrics(closed['label'],p,threshold)
            loc=localization_metrics(closed['t'],closed['x'],closed['y'],closed['label'],
                                     closed['name'],p,threshold)
            rows[model].append({'name':name,'high_density':high,'event':event,
                                'localization':loc,
                                'continuity':object_window_continuity(row_data,p,threshold),
                                'closed_windows':count,'closed_events':closed_count})
            probabilities[model].append(p)
            track=evaluate_sequence(closed,p,threshold,3,20.0,20)
            track_raw[model].append(track['raw'])
            track_misses[model].extend(track['longest_miss_per_track_ms'])
            b_rows[model].append({'name':name,'closed_windows':count,'candidates':len(features),
                                  'proposal_dense_macs':proposal_dense,
                                  'proposal_effective_macs':proposal_effective,
                                  'refiner_dense_macs':ref['dense_macs'],
                                  'refiner_effective_macs':ref['effective_macs']})
        labels.append(closed['label'])
        print(json.dumps({'role':role,'sequence':name,'windows':count,
                          'events':closed_count,'candidates':len(new_candidates)}),flush=True)
    a={}
    b={}
    for model in MODELS:
        a[model]=summarize(rows[model],probabilities[model],labels,MODELS[model])
        a[model]['macro_sequence_recall']=float(np.mean(
            [row['event']['recall'] for row in rows[model]]))
        track=hota_metrics(combine_hota_raw(track_raw[model]))
        a[model]['target_track']={'averaged':track['averaged'],
                                  'longest_continuous_miss_p95_ms':percentile95(track_misses[model])}
        totals={key:sum(row[key] for row in b_rows[model]) for key in
                ('proposal_dense_macs','proposal_effective_macs',
                 'refiner_dense_macs','refiner_effective_macs')}
        b[model]={**_static(h2_export.proposal_layers,layers[model],model),
                  'sequences':len(b_rows[model]),
                  'closed_windows':sum(row['closed_windows'] for row in b_rows[model]),
                  'candidates':sum(row['candidates'] for row in b_rows[model]),
                  'B03_dense_MAC_per_sequence':(totals['proposal_dense_macs']+
                    totals['refiner_dense_macs'])/len(b_rows[model]),
                  'B04a_effective_MAC_per_sequence':(totals['proposal_effective_macs']+
                    totals['refiner_effective_macs'])/len(b_rows[model]),
                  'B04b_effective_AC_per_sequence':0,
                  'B04c_effective_operations_per_sequence':(totals['proposal_effective_macs']+
                    totals['refiner_effective_macs'])/len(b_rows[model]),
                  'totals':totals,'per_sequence':b_rows[model]}
    # The prior A4-2 acceptance replay supplied a truncated sequence to a
    # noncausal model. Its sequence-time column can therefore differ from the
    # historical full-sequence deployment predictions cropped here. Preserve
    # both interpretations explicitly; only the H2-pro replay must coincide.
    prior=json.loads((ROOT/'reports'/('acceptance_'+role+'.json')).read_text(encoding='utf-8'))
    if (abs(a['H2-pro']['event']['iou']-prior['causal']['event']['iou'])>1e-12 or
            abs(a['H2-pro']['ap']-prior['causal']['ap'])>1e-12):
        raise AssertionError('post-freeze H2-pro A result differs from acceptance')
    result={'role':role,'scope':'same first-event-anchored complete 100 ms prefix for all three models',
            'status':'post-freeze read-only comparison; no model or threshold selection',
            'B_scope':'Proposal and conditional Refiner exact-int8 MAC counts; excludes feature construction, routing, connected components, CPU latency, and memory workspace',
            'historical_prediction_policy':'H2 and H2-plus predictions came from the formal full-sequence Step7 score archive, then were cropped; their noncausal sequence-time input remains the original full-sequence value',
            'earlier_truncated_input_A4_2_reference':{
                'iou':prior['baseline']['event']['iou'],'ap':prior['baseline']['ap']},
            'acceptance_lock_sha256':sha256(LOCK),'index_sha256':sha256(INDEX),
            'historical_score_hashes':score_hashes,
            'A':a,'B':b}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('role',choices=('validation','test'))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    output=args.output or ROOT/'reports'/('ab_same_prefix_'+args.role+'.json')
    result=run(args.role,output)
    print(json.dumps({'role':args.role,'output':str(output),
                      'IoU':{m:result['A'][m]['event']['iou'] for m in MODELS},
                      'B03':{m:result['B'][m]['B03_dense_MAC_per_sequence'] for m in MODELS}}))
