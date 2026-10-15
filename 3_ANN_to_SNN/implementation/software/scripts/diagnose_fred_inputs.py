"""Frozen-model input-only FRED controls. Never trains or edits model code.

Single factors: fixed 2x2 spatial tiling (2x geometric scale), or nested,
label-independent event retention 1/4 and 1/16 with original geometry.
Retention metrics are paired against baseline on exactly the same source IDs;
full-source rejection accounting is reported separately.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import numpy as np

import evaluate_fred_frozen as frozen
import prepare_fred_diagnosis as source

ROOT, OUT, BASELINE, DATA = source.ROOT, source.OUT, source.BASELINE, source.DATA
ARMS = {
    'baseline':dict(tiles=1,retain_denominator=1),
    'scale2':dict(tiles=2,retain_denominator=1),
    'density4':dict(tiles=1,retain_denominator=4),
    'density16':dict(tiles=1,retain_denominator=16),
}
GROUPS=('weak_target','background','ann_tp','ann_fn','ann_fp')
PROTOCOL = {
    'version':'fred_input_diagnosis_v1', 'models':list(frozen.MODELS),
    'threshold':.35,'training_started':False,'parameter_updates':0,
    'arms':ARMS, 'selection_seed':42,
    'geometry':'fixed equal source partitions, uniform letterbox within each; no label-dependent crops',
    'density':'nested source-event-index hash retention, no time/polarity/label changes',
    'window_origin':'original first event and identical 100 ms source-time boundaries for every arm/view',
    'metrics':['retained events; paired old baseline on the same event IDs',
               'all source events, discarded events counted as predicted background'],
    'weak_labels':'unchanged original source-space labels and eval_mask',
    'cost':'all retained closed events including masked labels; every spatial view counted',
    'backend':'original CPU ANN and count_reference SNN; sampled explicit spike_add check',
    'bits':'original evaluation bits0-7; bit8 retained in this arm; bit9 reserved',
    'limitations':['tiling changes boundary context as an intrinsic geometry consequence',
                   'event thinning is an input diagnostic, not full-stream recall improvement',
                   'two source recordings and one deterministic selection seed'],
}


def keep_events(ids, denominator, sequence_number=0):
    x=np.asarray(ids,dtype=np.uint64)+np.uint64(42+sequence_number*1000000007)
    with np.errstate(over='ignore'):
        x=(x^(x>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        x=(x^(x>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
    x=x^(x>>np.uint64(31))
    return (x & np.uint64(denominator-1))==0


def transform(raw, tiles=1):
    width,height=1280//tiles,720//tiles
    ix=raw['x'].astype(np.int64)//width
    iy=raw['y'].astype(np.int64)//height
    view=iy*tiles+ix
    scale=min(346/width,260/height)
    packet=np.empty(len(raw),dtype=frozen.EVENT_DTYPE)
    packet['x']=np.floor((raw['x'].astype(np.float64)-ix*width+.5)*scale+(346-width*scale)/2)
    packet['y']=np.floor((raw['y'].astype(np.float64)-iy*height+.5)*scale+(260-height*scale)/2)
    packet['t']=raw['t'].astype(np.float64)/1000.
    packet['p']=raw['p']
    return packet,view


def groups_for_bits(bits):
    valid=((bits&16)!=0)&((bits&64)!=0)
    retained=(bits&256)!=0
    opening=(bits&128)!=0
    return {'all':valid,'after_5s':valid&~opening,'opening_5s':valid&opening},retained


def update_metrics(target,bits,old_bits):
    masks,retained=groups_for_bits(bits)
    label=(bits&32)!=0
    for model_i,model in enumerate(frozen.MODELS):
        predicted=(bits&(2<<model_i))!=0
        old_predicted=(old_bits&(2<<model_i))!=0
        candidate=(bits&1)!=0
        old_candidate=(old_bits&1)!=0
        for segment,valid in masks.items():
            for scope,p,c,take in [
                ('retained',predicted,candidate,valid&retained),
                ('paired_baseline',old_predicted,old_candidate,valid&retained),
                ('full_source',predicted,candidate,valid)]:
                key=(model,scope,segment)
                if key not in target:target[key]=frozen.zero_counts()
                frozen.add_counts(target[key],frozen.counts(label,p,c,take))


def metric_report(totals):
    return {m:{scope:{segment:frozen.summarize(totals[(m,scope,segment)])
            for segment in ('all','after_5s','opening_5s')}
            for scope in ('retained','paired_baseline','full_source')} for m in frozen.MODELS}


def feature_histograms(hist,x,labels,valid,predicted):
    masks=(valid&labels,valid&~labels,valid&labels&predicted,valid&labels&~predicted,
           valid&~labels&predicted)
    for g,take in enumerate(masks):
        selected=x[take]
        if not len(selected):continue
        for col in range(36):
            hist[g,col]+=np.bincount(selected[:,col],minlength=256).astype(np.int64)


def density_profile(raw,packet,t0_ms):
    # Event-weighted collision profiles: same 2 ms bin and polarity, raw vs mapped pixel.
    if not len(raw):return None
    tb=np.floor((packet['t']-t0_ms)/2.).astype(np.int64)
    result={}
    for name,x,y,width,height in [('source',raw['x'],raw['y'],1280,720),
                                  ('mapped',packet['x'],packet['y'],346,260)]:
        key=(((tb*height+y)*width+x)*2+raw['p']).astype(np.int64)
        _,counts=np.unique(key,return_counts=True)
        hist=np.bincount(np.minimum(counts,4095),weights=counts,minlength=4096).astype(np.int64)
        result[name]=hist
    return result


def sum_values(target,values):
    for k,v in values.items():target[k]=target.get(k,0)+int(v)


def execute_sequence(models,output,arm,seq,sequence_number,identity):
    name=seq['source_sequence']
    directory=output/arm/name
    directory.mkdir(parents=True,exist_ok=True)
    done=directory/'result.json'
    if done.exists():
        result=frozen.read_json(done)
        assert result['identity']==identity
        for fn,h in result['artifact_sha256'].items():assert frozen.sha256(directory/fn)==h
        print(json.dumps({'stage':'reuse_sequence','arm':arm,'sequence':name}),flush=True)
        return result
    raw=np.load(str(output/'source'/name/'events.npy'),mmap_mode='r',allow_pickle=False)
    old=np.load(str(BASELINE/name/'predictions.npy'),mmap_mode='r',allow_pickle=False)
    previous=frozen.read_json(BASELINE/name/'result.json')
    assert frozen.sha256(BASELINE/name/'predictions.npy')==previous['predictions_sha256']
    assert frozen.sha256(BASELINE/name/'windows.jsonl')==previous['windows_sha256']
    old_windows=[json.loads(line) for line in (BASELINE/name/'windows.jsonl').read_text().splitlines()]
    assert len(raw)==len(old)==seq['output_events']
    config=ARMS[arm]
    pipes=[models.pipeline() for _ in range(config['tiles']**2)]
    for pipe in pipes:pipe.reset(previous['t0_ms'])
    partial=directory/'predictions.partial.npy'
    windows_partial=directory/'windows.partial.jsonl'
    for p in (partial,windows_partial):
        if p.exists():p.rename(p.with_name(p.name+f'.interrupted_{time.time_ns()}'))
    saved=np.lib.format.open_memmap(str(partial),mode='w+',dtype=np.uint16,shape=(len(raw),))
    for start in range(0,len(raw),1_000_000):
        saved[start:start+1_000_000]=old[start:start+1_000_000].astype(np.uint16)&(16+32+128)
    totals={}
    proposal=dict(dense_mac=0,zero_skip_equivalent_mac=0,output_scalars=0)
    refiners={m:{} for m in frozen.MODELS}
    checks={m:0 for m in frozen.MODELS[1:]}
    histogram=np.zeros((len(GROUPS),36,256),np.int64)
    pixel_density={k:np.zeros(4096,np.int64) for k in ('source','mapped')}
    candidates=retained=covered=nonempty_views=0
    baseline_replay=0
    started=time.perf_counter()
    with windows_partial.open('w',encoding='utf-8') as log:
        for wi,old_window in enumerate(old_windows):
            lo,n=old_window['first_event_index'],old_window['events']
            hi=lo+n
            assert lo==covered
            ids=np.arange(lo,hi,dtype=np.int64)
            keep=keep_events(ids,config['retain_denominator'],sequence_number)
            selected_ids=ids[keep]
            block=raw[lo:hi][keep]
            packet,view=transform(block,config['tiles'])
            code=(old[lo:hi].astype(np.uint16)&(16+32+128))|64
            code[keep]|=256
            costs={m:{} for m in frozen.MODELS}
            pc_total=dict(dense_mac=0,zero_skip_equivalent_mac=0,output_scalars=0)
            right=old_window['right_boundary_ms']
            window_candidate=0
            # Density is measured on the whole source-time window, independent of labels.
            if arm=='baseline':
                density=density_profile(block,packet,previous['t0_ms'])
                if density:
                    for k,v in density.items():pixel_density[k]+=v
            for view_i,pipe in enumerate(pipes):
                take=view==view_i
                local_ids=selected_ids[take]-lo
                pipe.push(packet[take],right)
                ready=pipe.flush_completed_windows()
                assert len(ready)==1
                w=ready[0]
                assert len(w.predictions)==len(local_ids)
                nc=len(w.features)
                candidate_ids=local_ids[w.candidate_mask]
                code[local_ids]|=w.candidate_mask.astype(np.uint16)
                code[local_ids]|=w.predictions.astype(np.uint16)*2
                pc=frozen.proposal_cost(pipe,w)
                sum_values(pc_total,pc)
                window_candidate+=nc
                nonempty_views+=int(len(local_ids)>0)
                sum_values(costs[frozen.MODELS[0]],asdict(frozen.Cost(executed_synaptic_mac=nc*4480,
                    rate_network_mac=nc*4480,scalar_scale_mul=nc*100,scalar_normalization_div=nc*100,
                    quantizer_scale_mul=nc*96,quantizer_round=nc*96,quantizer_comparisons=nc*96*4)))
                labels=raw['label'][lo+candidate_ids].astype(bool)
                valid=raw['valid'][lo+candidate_ids]
                feature_histograms(histogram,w.features,labels,valid,w.predictions[w.candidate_mask])
                for mi,m in enumerate(frozen.MODELS[1:],1):
                    if arm=='baseline':
                        code[candidate_ids]|=old[lo+candidate_ids].astype(np.uint16)&(2<<mi)
                        sum_values(costs[m],old_window['refiner_cost'][m])
                    else:
                        for begin in range(0,nc,32768):
                            x=w.features[begin:begin+32768]
                            logits,cost=models.snn(x,m)
                            p=frozen.probabilities(logits)
                            if not np.isfinite(p).all():raise FloatingPointError('Nonfinite frozen output')
                            code[candidate_ids[begin:begin+len(x)]]|=(p>=.35).astype(np.uint16)*(2<<mi)
                            sum_values(costs[m],asdict(cost))
                        if not nc:
                            _,cost=models.snn(np.empty((0,36),np.uint8),m)
                            sum_values(costs[m],asdict(cost))
                    if nc and wi%100==0:
                        probe_ids=np.unique(np.linspace(0,nc-1,min(16,nc),dtype=np.int64))
                        x=w.features[probe_ids]
                        dense,dc=models.snn(x,m)
                        explicit,ec=models.snn(x,m,'spike_add')
                        np.testing.assert_array_equal(dense,explicit)
                        assert dc.rate_network_ac==ec.executed_synaptic_ac
                        expected=(code[candidate_ids[probe_ids]]&(2<<mi))!=0
                        np.testing.assert_array_equal(frozen.probabilities(dense)>=.35,expected)
                        checks[m]+=len(probe_ids)
            if arm=='baseline':
                np.testing.assert_array_equal((code&255).astype(np.uint8),old[lo:hi])
                assert pc_total==old_window['proposal']
                baseline_replay+=n
            saved[lo:hi]=code
            update_metrics(totals,code,old[lo:hi])
            sum_values(proposal,pc_total)
            for m in frozen.MODELS:sum_values(refiners[m],costs[m])
            retained+=len(block);candidates+=window_candidate;covered=hi
            current_totals={}
            update_metrics(current_totals,code,old[lo:hi])
            log.write(json.dumps({'window':wi,'right_boundary_ms':right,'first_source_index':lo,
                'source_events':n,'retained_events':len(block),'candidate_events':window_candidate,
                'proposal':pc_total,'refiner_cost':costs,'models':metric_report(current_totals)})+'\n')
            if wi%50==0 or wi==len(old_windows)-1:
                log.flush()
                progress={'stage':'inference','arm':arm,'sequence':name,'closed_windows':wi+1,
                    'total_windows':len(old_windows),'closed_source_events':covered,
                    'retained_events':retained,'seconds':time.perf_counter()-started,'training_started':False}
                frozen.atomic_json(output/f'progress_{arm}.json',progress)
                print(json.dumps(progress),flush=True)
        # Feed the incomplete tail on the same time axis, without publishing it.
        ids=np.arange(covered,len(raw),dtype=np.int64)
        keep=keep_events(ids,config['retain_denominator'],sequence_number)
        tail=raw[covered:][keep]
        packet,view=transform(tail,config['tiles'])
        saved[covered:][keep]|=256
        tail_count=0
        for i,pipe in enumerate(pipes):
            pipe.push(packet[view==i],previous['last_source_ms'])
            ready,left=pipe.finalize()
            assert not ready
            tail_count+=left
        assert tail_count==len(tail)
    saved.flush()
    saved._mmap.close()
    del saved
    os.replace(partial,directory/'predictions.npy')
    os.replace(windows_partial,directory/'windows.jsonl')
    np.savez_compressed(directory/'features.npz',histogram=histogram,groups=np.array(GROUPS),
                        source_pixel_density=pixel_density['source'],mapped_pixel_density=pixel_density['mapped'])
    result={'status':'complete','arm':arm,'sequence':name,'identity':identity,
        'source_events':len(raw),'closed_source_events':covered,'retained_closed_events':retained,
        'retained_tail_events':tail_count,'source_tail_events':len(raw)-covered,
        'closed_windows':len(old_windows),'spatial_views':len(pipes),'nonempty_view_windows':nonempty_views,
        'candidate_events':candidates,'models':metric_report(totals),
        'cost':{m:frozen.cost_report(refiners[m],proposal,m,candidates) for m in frozen.MODELS},
        'baseline_replayed_events':baseline_replay,'explicit_checks_per_model':checks,
        'seconds':time.perf_counter()-started,'training_started':False,
        'artifact_sha256':{f:frozen.sha256(directory/f) for f in ['predictions.npy','windows.jsonl','features.npz']}}
    frozen.atomic_json(done,result)
    print(json.dumps({'stage':'sequence_complete','arm':arm,'sequence':name,
        'iou':{m:result['models'][m]['retained']['all']['iou'] for m in frozen.MODELS},
        'seconds':result['seconds']}),flush=True)
    return result


def run(output,arms):
    models=frozen.FrozenModels()
    index=frozen.read_json(DATA/'dataset/index.json')
    audit=frozen.read_json(output/'source_audit.json')
    assert audit['status']=='passed' and audit['audited_events']==48_516_008
    cache_hashes={r['sequence']:r['cache_sha256'] for r in audit['sequences']}
    for name,digest in cache_hashes.items():
        assert frozen.sha256(output/'source'/name/'events.npy')==digest
    identity={'protocol':PROTOCOL,'frozen_models':models.identity(DATA/'dataset'),
        'script_sha256':frozen.sha256(Path(__file__)),
        'source_cache_sha256':cache_hashes,
        'source_identity_sha256':frozen.sha256(output/'source_identity.json'),
        'baseline_comparison_sha256':frozen.sha256(BASELINE/'comparison.json')}
    p=output/'experiment_identity.json'
    if p.exists() and frozen.read_json(p)!=identity:raise ValueError('Different experiment identity')
    frozen.atomic_json(p,identity)
    for arm in arms:
        rows=[execute_sequence(models,output,arm,s,i,identity) for i,s in enumerate(index['sequences'])]
        frozen.atomic_json(output/arm/'completed.json',{'status':'complete','arm':arm,'sequences':rows,
            'completed_at':datetime.now(timezone.utc).isoformat(),'training_started':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--arms',nargs='+',choices=ARMS,default=list(ARMS))
    args=parser.parse_args()
    run(args.output,args.arms)
