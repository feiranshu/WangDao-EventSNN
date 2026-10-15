"""Full-coverage FRED inference with independent neighborhood-only adaptation."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

import diagnose_fred_inputs as prior
import evaluate_fred_frozen as frozen
from wangdao.fred_statistics_v1 import neighborhood as adapter
from wangdao.h2_pro.causal_feature_builder import base_features

ROOT = frozen.ROOT
SOURCE = prior.OUT
BASELINE = prior.BASELINE
DATA = prior.DATA
OUT = ROOT/'results/runs/fred_statistics_adaptation_v1'
DENOMINATOR = 16
GROUPS = ('weak_target','background','ann_tp','ann_fn','ann_fp')
SEGMENTS = ('all','opening_5s','after_5s','statistics_updates','query_only')
PROTOCOL = {
    'version':'fred_statistics_adaptation_v1', 'threshold':.35,
    'models':list(frozen.MODELS),'training_started':False,'parameter_updates':0,
    'changed_feature_columns':list(range(4,28)),
    'unchanged_feature_columns':adapter.UNCHANGED_COLUMNS.tolist(),
    'statistics_denominator':DENOMINATOR,'selection_seed':42,
    'selection':'same nested source-ID hash as prior density16; labels unused',
    'queries':'all original Proposal candidates; noncandidates receive the original background decision',
    'self_event':'current query contributes once to its same-polarity current cell, even if not selected for state',
    'statistics_context':'selected source events from previous/current original closed 100 ms window',
    'geometry':'original whole-frame letterbox; no tiling or coordinate/time/polarity change',
    'unchanged':'original full-event Proposal, gate, heatmap/context/history, motion, model weights and encoding',
    'tail':'all events fed; original incomplete tail is not published',
    'backend':'CPU unchanged count_reference; fixed samples checked with spike_add',
    'bits':'bits0-7 identical meanings to original; bit8 updates statistics (NOT a retention flag)',
    'baseline':'saved original three-model predictions/cost; full ANN replay and SNN sampled replay',
    'cost':'full closed-input neural equivalents and executed dense reference reported separately',
    'extra_audits':'original ANN replay plus Proposal activity recount and sampled equivalence checks are excluded from single-model costs',
    'selection_limit':'1/16 fixed from prior FRED diagnostic; these two already-examined recordings are development evidence, not independent test data',
    'label_limit':'unchanged box-derived labels and eval_mask, including known uncertain no-box negatives',
}


def score(bits,totals):
    valid = ((bits&16)!=0)&((bits&64)!=0)
    label,candidate,opening,updates = (bits&32)!=0,(bits&1)!=0,(bits&128)!=0,(bits&256)!=0
    masks = {'all':valid,'opening_5s':valid&opening,'after_5s':valid&~opening,
             'statistics_updates':valid&updates,'query_only':valid&~updates}
    row = {}
    for mi,m in enumerate(frozen.MODELS):
        row[m] = {}
        for segment,take in masks.items():
            c = frozen.counts(label,(bits&(2<<mi))!=0,candidate,take)
            key=(m,segment)
            if key not in totals:totals[key]=frozen.zero_counts()
            frozen.add_counts(totals[key],c)
            row[m][segment]=c
    return row


def report_metrics(totals):
    return {m:{s:frozen.summarize(totals[(m,s)]) for s in SEGMENTS} for m in frozen.MODELS}


def inference(models,features,model,backend='count_reference'):
    if model==frozen.MODELS[0]:
        return frozen.forward(features,models.ann,'ann',backend=backend)
    return models.snn(features,model,backend)


def run_sequence(models,output,seq,sequence_number,identity):
    name=seq['source_sequence']; directory=output/name
    directory.mkdir(exist_ok=True)
    done=directory/'result.json'
    if done.exists():
        result=frozen.read_json(done)
        if result['identity']!=identity:raise ValueError('Completed record identity changed')
        for filename,h in result['artifact_sha256'].items():
            if frozen.sha256(directory/filename)!=h:raise ValueError('Completed artifact changed')
        print(json.dumps({'stage':'reuse_sequence','sequence':name}),flush=True)
        return result
    raw=np.load(str(SOURCE/'source'/name/'events.npy'),mmap_mode='r',allow_pickle=False)
    old=np.load(str(BASELINE/name/'predictions.npy'),mmap_mode='r',allow_pickle=False)
    baseline=frozen.read_json(BASELINE/name/'result.json')
    assert frozen.sha256(BASELINE/name/'predictions.npy')==baseline['predictions_sha256']
    assert frozen.sha256(BASELINE/name/'windows.jsonl')==baseline['windows_sha256']
    old_windows=[json.loads(x) for x in (BASELINE/name/'windows.jsonl').read_text().splitlines()]
    assert len(raw)==len(old)==seq['output_events']
    pipe=models.pipeline();pipe.reset(baseline['t0_ms'])
    previous_context=np.empty(0,dtype=frozen.EVENT_DTYPE)
    partial=directory/'predictions.partial.npy';logpath=directory/'windows.partial.jsonl'
    for p in (partial,logpath):
        if p.exists():p.rename(p.with_name(p.name+f'.interrupted_{time.time_ns()}'))
    saved=np.lib.format.open_memmap(str(partial),mode='w+',dtype=np.uint16,shape=(len(raw),))
    for lo in range(0,len(raw),500_000):
        hi=min(lo+500_000,len(raw))
        saved[lo:hi]=(old[lo:hi].astype(np.uint16)&(16+32+128))|(
            adapter.update_mask(np.arange(lo,hi),DENOMINATOR,sequence_number).astype(np.uint16)*256)
    totals={};base_totals={};proposal={};refiners={m:{} for m in frozen.MODELS}
    hist=np.zeros((len(GROUPS),36,256),np.int64)
    changed_columns=np.zeros(36,np.int64)
    checks={m:0 for m in frozen.MODELS[1:]}
    identity_checks=state_selected=covered=candidates=0
    sample_records=[];started=time.perf_counter()
    with logpath.open('w',encoding='utf-8') as log:
        for wi,old_w in enumerate(old_windows):
            lo,n=old_w['first_event_index'],old_w['events'];hi=lo+n
            assert lo==covered
            packet,_=prior.transform(raw[lo:hi],1)
            keep=adapter.update_mask(np.arange(lo,hi),DENOMINATOR,sequence_number)
            context=np.concatenate((previous_context,packet[keep]))
            # The unchanged pipeline supplies the original candidates/context and a full baseline audit.
            full_history=np.concatenate((pipe.previous_events,packet)) if wi%100==0 else None
            history_start=len(pipe.previous_events)
            pipe.push(packet,old_w['right_boundary_ms'])
            ready=pipe.flush_completed_windows();assert len(ready)==1
            w=ready[0];assert len(w.predictions)==n
            np.testing.assert_array_equal(w.candidate_mask,(old[lo:hi]&1)!=0)
            np.testing.assert_array_equal(w.predictions,(old[lo:hi]&2)!=0)
            pc=frozen.proposal_cost(pipe,w);assert pc==old_w['proposal']
            selected=np.flatnonzero(w.candidate_mask);nc=len(selected)
            adapted=adapter.replace_counts(w.features,adapter.count_columns(context,packet[selected],keep[selected],pipe.t0))
            np.testing.assert_array_equal(adapted[:,adapter.UNCHANGED_COLUMNS],w.features[:,adapter.UNCHANGED_COLUMNS])
            changed_columns+=np.count_nonzero(adapted!=w.features,axis=0)
            code=saved[lo:hi].copy()|64
            code|=w.candidate_mask.astype(np.uint16)
            costs={m:{} for m in frozen.MODELS}
            for mi,m in enumerate(frozen.MODELS):
                for start in range(0,max(nc,1),32768):
                    x=adapted[start:start+32768]
                    logits,cost=inference(models,x,m)
                    code[selected[start:start+len(x)]]|=(frozen.probabilities(logits)>=.35).astype(np.uint16)*(2<<mi)
                    prior.sum_values(costs[m],asdict(cost))
            if nc and wi%100==0:
                pick=np.unique(np.linspace(0,nc-1,min(16,nc),dtype=np.int64))
                # Denominator-one identity on real data, using full context and unchanged original features.
                exact=adapter.count_columns(full_history,packet[selected[pick]],np.ones(len(pick),bool),pipe.t0)
                np.testing.assert_array_equal(exact,w.features[pick,4:28])
                # Selected update queries are exactly equivalent to original extraction on the sparse context.
                updating=np.flatnonzero(keep[selected])
                if len(updating):
                    subset=updating[np.unique(np.linspace(0,len(updating)-1,min(16,len(updating)),dtype=np.int64))]
                    original_sparse=base_features(context,len(previous_context),pipe.t0)
                    positions=np.searchsorted(np.flatnonzero(keep),selected[subset])
                    np.testing.assert_array_equal(adapted[subset,4:28],original_sparse[positions,4:28])
                for m in frozen.MODELS[1:]:
                    before,_=inference(models,w.features[pick],m)
                    mi=frozen.MODELS.index(m)
                    np.testing.assert_array_equal(frozen.probabilities(before)>=.35,(old[lo+selected[pick]]&(2<<mi))!=0)
                    dense,dc=inference(models,adapted[pick],m)
                    explicit,ec=inference(models,adapted[pick],m,'spike_add')
                    np.testing.assert_array_equal(dense,explicit)
                    assert dc.rate_network_ac==ec.executed_synaptic_ac
                    checks[m]+=len(pick)
                before,_=inference(models,w.features[pick],frozen.MODELS[0])
                np.testing.assert_array_equal(frozen.probabilities(before),w.probabilities[selected[pick]])
                identity_checks+=len(pick)
                for j in pick:
                    sample_records.append((int(lo+selected[j]),wi,w.features[j].copy(),adapted[j].copy(),bool(keep[selected[j]])))
            saved[lo:hi]=code
            row=score(code,totals)
            baseline_code=old[lo:hi].astype(np.uint16)|(keep.astype(np.uint16)*256)
            score(baseline_code,base_totals)
            prior.feature_histograms(hist,adapted,raw['label'][lo+selected].astype(bool),raw['valid'][lo+selected],(code[selected]&2)!=0)
            prior.sum_values(proposal,pc)
            for m in frozen.MODELS:prior.sum_values(refiners[m],costs[m])
            previous_context=packet[keep].copy()
            state_selected+=int(keep.sum());covered=hi;candidates+=nc
            log.write(json.dumps({'window':wi,'first_source_index':lo,'source_events':n,
                'right_boundary_ms':old_w['right_boundary_ms'],'candidate_events':nc,
                'statistics_update_events':int(keep.sum()),'models':row,'proposal':pc,'refiner_cost':costs})+'\n')
            if wi%50==0 or wi==len(old_windows)-1:
                log.flush()
                progress={'stage':'inference','sequence':name,'closed_windows':wi+1,
                    'total_windows':len(old_windows),'source_events':covered,'candidate_events':candidates,
                    'statistics_update_events':state_selected,'seconds':time.perf_counter()-started,'training_started':False}
                frozen.atomic_json(output/'progress.json',progress)
                print(json.dumps(progress),flush=True)
        tail,_=prior.transform(raw[covered:],1)
        end=np.nextafter(seq['source_last_timestamp_us']/1000.,np.inf)
        pipe.push(tail,end)
        remaining,leftover=pipe.finalize()
        assert not remaining and leftover==len(raw)-covered
    saved.flush();del saved
    partial.rename(directory/'predictions.npy');logpath.rename(directory/'windows.jsonl')
    np.savez_compressed(directory/'features.npz',histogram=hist,groups=np.array(GROUPS),changed_columns=changed_columns,
        sample_ids=np.array([x[0] for x in sample_records],np.int64),sample_windows=np.array([x[1] for x in sample_records],np.int32),
        sample_original=np.array([x[2] for x in sample_records],np.uint8).reshape(-1,36),
        sample_adapted=np.array([x[3] for x in sample_records],np.uint8).reshape(-1,36),
        sample_updates=np.array([x[4] for x in sample_records],bool))
    result={'status':'complete','sequence':name,'identity':identity,'source_events':len(raw),
        'closed_source_events':covered,'source_tail_events':len(raw)-covered,'closed_windows':len(old_windows),
        'candidate_events':candidates,'statistics_update_events':state_selected,
        'query_only_closed_events':covered-state_selected,'discarded_query_events':0,
        'models':report_metrics(totals),'paired_baseline':report_metrics(base_totals),
        'cost':{m:frozen.cost_report(refiners[m],proposal,m,candidates) for m in frozen.MODELS},
        'baseline_replayed_events':covered,'explicit_snn_checks':checks,
        'denominator_one_identity_checks':identity_checks,'changed_feature_values_by_column':changed_columns.tolist(),
        'additional_baseline_ann_refiner_audit_mac':candidates*4480,
        'seconds':time.perf_counter()-started,'training_started':False,
        'artifact_sha256':{n:frozen.sha256(directory/n) for n in ('predictions.npy','windows.jsonl','features.npz')}}
    for mi,m in enumerate(frozen.MODELS):
        assert result['cost'][m]['refiner_mac']==candidates*[4480,2432,2304][mi]
    frozen.atomic_json(done,result)
    print(json.dumps({'stage':'sequence_complete','sequence':name,'seconds':result['seconds'],
        'iou':{m:result['models'][m]['all']['iou'] for m in frozen.MODELS}}),flush=True)
    return result


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if not (output/'preservation_before.json').exists():raise ValueError('Create the original-file preservation snapshot first')
    models=frozen.FrozenModels();index=frozen.read_json(DATA/'dataset/index.json')
    audit=frozen.read_json(SOURCE/'source_audit.json')
    assert audit['status']=='passed' and audit['audited_events']==48_516_008
    cache={r['sequence']:r['cache_sha256'] for r in audit['sequences']}
    for name,h in cache.items():assert frozen.sha256(SOURCE/'source'/name/'events.npy')==h
    identity={'protocol':PROTOCOL,'frozen_models':models.identity(DATA/'dataset'),
        'source_cache_sha256':cache,'baseline_comparison_sha256':frozen.sha256(BASELINE/'comparison.json'),
        'source_diagnosis_sha256':frozen.sha256(SOURCE/'comparison.json'),
        'runner_sha256':frozen.sha256(Path(__file__)),'adapter_sha256':frozen.sha256(Path(adapter.__file__)),
        'prior_input_helpers_sha256':frozen.sha256(Path(prior.__file__))}
    assert identity['frozen_models']==frozen.read_json(SOURCE/'experiment_identity.json')['frozen_models']
    path=output/'experiment_identity.json'
    if path.exists():
        if frozen.read_json(path)!=identity:raise ValueError('Experiment identity changed; use a new run directory')
    else:frozen.atomic_json(path,identity)
    rows=[run_sequence(models,output,s,i,identity) for i,s in enumerate(index['sequences'])]
    frozen.atomic_json(output/'completed.json',{'status':'complete','sequences':rows,
        'completed_at':datetime.now(timezone.utc).isoformat(),'training_started':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    run(p.parse_args().output)
