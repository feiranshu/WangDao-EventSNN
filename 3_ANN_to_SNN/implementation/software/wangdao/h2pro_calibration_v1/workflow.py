"""Exact checks and complete Dev evaluation after calibration is frozen."""
from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from ..h2pro_snn.foundation import read_json, sha256, verify_baseline
from ..h2pro_snn.diagnostics import aggregate_counts, error_counts
from ..h2pro_direct_v1.core import forward as original_forward
from ..h2pro_direct_v1.workflow import cache_manifest, load_sequence, checks as original_checks
from .common import identity, lock_directory, atomic_json, protocol, source_hashes
from .core import Encoder, Cost, accumulate, forward, probabilities, load_layers


def checks():
    original=original_checks(); layers=load_layers()
    a=np.tile(np.arange(256,dtype=np.uint8)[:,None],(1,3))
    w=np.array([[127,-128,0],[-7,9,11],[5,-1,-3]],np.int8)
    count=0
    for t in protocol()['steps']:
        for theta in protocol()['threshold_grid']:
            for q in protocol()['initial_phase_quarters']:
                e=Encoder(theta,theta*q//4)
                reference,rc=accumulate(a,w,t,e,'count_reference')
                actual,ac=accumulate(a,w,t,e,'spike_add')
                if not np.array_equal(reference,actual) or rc.unit_spikes!=ac.unit_spikes:
                    raise AssertionError('Calibrated IF differs from analytic reference')
                if ac.executed_synaptic_mac or ac.executed_synaptic_ac!=ac.unit_spikes*w.shape[1]:
                    raise AssertionError('Binary AC counting mismatch')
                count+=1
    x=np.random.default_rng(42).integers(0,256,(257,36),dtype=np.uint8)
    for stage in ('ann','r1','r2'):
        for t in (4,8,16,255):
            old,_=original_forward(x,layers,stage,t,'count_reference')
            for backend in ('count_reference','spike_add'):
                new,_=forward(x,layers,stage,t,backend=backend)
                if not np.array_equal(old,new):raise AssertionError('Uncalibrated direct_v1 contract changed')
    return {'status':'passed','training_started':False,'weights_updated':False,
            'baseline':original['baseline'],'exhaustive_encoder_configurations':count,
            'original_default_exact':True,'sources_utf8_lf':source_hashes()}


def evaluate_point(cache,manifest,layers,stage,steps,mode,encoders,output,chunk_rows):
    target=Path(output)/f'{stage}_t{steps}_{mode}'
    target.mkdir(exist_ok=True)
    signature={'stage':stage,'steps':steps,'mode':mode,'encoders':[asdict(e) for e in encoders]}
    rows=[]
    for row in manifest['sequences']:
        receipt=target/(Path(row['sequence']).stem+'.json')
        if receipt.exists():
            saved=read_json(receipt)
            if (saved['cache_sha256']!=row['cache_sha256'] or saved['configuration']!=signature
                    or saved['sequence']!=row['sequence'] or not saved['reference_logits_exact']):
                raise ValueError('Evaluation resume receipt mismatch')
            rows.append(saved); continue
        a=load_sequence(cache,row)
        parts=[]; total=Cost(); digest=hashlib.sha256()
        inference_seconds=0.; reference_seconds=0.
        for lo in range(0,len(a['features']),chunk_rows):
            x=a['features'][lo:lo+chunk_rows]
            start=time.perf_counter()
            raw,cost=forward(x,layers,stage,steps,encoders,'spike_add')
            inference_seconds+=time.perf_counter()-start
            start=time.perf_counter()
            reference,rc=forward(x,layers,stage,steps,encoders,'count_reference')
            reference_seconds+=time.perf_counter()-start
            if (not np.array_equal(raw,reference) or cost.unit_spikes!=rc.unit_spikes
                    or cost.executed_synaptic_ac!=rc.rate_network_ac):
                raise AssertionError('Full-Dev explicit spikes differ from reference')
            parts.append(raw); total.merge(cost); digest.update(raw.astype('<f4').tobytes())
        raw=np.concatenate(parts) if parts else np.empty((0,4),np.float32)
        p=np.zeros(len(a['label']),np.float32); p[a['candidate']]=probabilities(raw)
        if stage=='ann' and (not np.array_equal(raw,a['logits']) or not np.array_equal(p,a['teacher'])):
            raise AssertionError('Original frozen ANN replay is no longer exact')
        delta=raw.astype(np.float64)-a['logits'].astype(np.float64)
        saved={'configuration':signature,'sequence':row['sequence'],'cache_sha256':row['cache_sha256'],
            'high_density':row['high_density'],'counts':error_counts(a['label'],p,a['candidate'],.35),
            'cost':asdict(total),'proposal_synaptic_mac':row['proposal_synaptic_mac'],
            'reference_logits_exact':True,'logits_sha256':digest.hexdigest(),
            'teacher_exact':np.array_equal(raw,a['logits']) and np.array_equal(p,a['teacher']),
            'prediction_disagreements':int(np.count_nonzero((p>=.35)!=(a['teacher']>=.35))),
            'raw_logit_abs_error_max':float(np.abs(delta).max()) if delta.size else 0.,
            'raw_logit_squared_error_sum':float(np.square(delta).sum()),'raw_logit_count':int(delta.size),
            'explicit_inference_seconds_including_counting':inference_seconds,
            'reference_check_seconds':reference_seconds}
        atomic_json(receipt,saved); rows.append(saved)
        print(json.dumps({'stage':stage,'T':steps,'mode':mode,'sequence':row['sequence'],
            'explicit_seconds':inference_seconds,'reference_logits_exact':True}),flush=True)
    cost={k:sum(r['cost'][k] for r in rows) for k in asdict(Cost())}
    proposal=sum(r['proposal_synaptic_mac'] for r in rows)
    n=sum(r['counts']['candidate_events'] for r in rows)
    result={**signature,'backend':'spike_add','threshold':.35,
        'overall':aggregate_counts([r['counts'] for r in rows]),
        'ordinary':aggregate_counts([r['counts'] for r in rows if not r['high_density']]),
        'high_density':aggregate_counts([r['counts'] for r in rows if r['high_density']]),
        'refiner_cost':cost,'proposal_synaptic_mac':proposal,
        'executed_neural_mac':proposal+cost['executed_synaptic_mac'],
        'executed_neural_ac':cost['executed_synaptic_ac'],
        'mac_reduction_fraction':1-(proposal+cost['executed_synaptic_mac'])/(proposal+n*4480),
        'synaptic_mac_plus_ac_ratio':(proposal+cost['executed_synaptic_mac']+cost['executed_synaptic_ac'])/(proposal+n*4480),
        'reference_logits_exact':all(r['reference_logits_exact'] for r in rows),
        'prediction_disagreements':sum(r['prediction_disagreements'] for r in rows),
        'sequences':rows,'end_to_end_complete':False,'energy_measured':False,
        'unaccounted':['Proposal scalar quantization/scaling','original feature construction',
                       'heatmaps/routing/components','prototype probability readout','memory traffic'],
        'timing_scope':'CPU cached-feature forward including diagnostics/counting; excludes input loading and reference verification; not end-to-end hardware latency.'}
    return result


def write_table(output,report):
    fields=['stage','T','mode','iou_pct','precision_pct','recall_pct','high_density_iou_pct',
            'high_density_recall_pct','MAC_G','AC_G','unit_spikes','ac_reduction_vs_uncalibrated_pct']
    with (Path(output)/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields); writer.writeheader()
        for r in report['results']:
            writer.writerow({'stage':r['stage'],'T':r['steps'],'mode':r['mode'],
                **{m+'_pct':100*r['overall'][m] for m in ('iou','precision','recall')},
                'high_density_iou_pct':100*r['high_density']['iou'],
                'high_density_recall_pct':100*r['high_density']['recall'],
                'MAC_G':r['executed_neural_mac']/1e9,'AC_G':r['executed_neural_ac']/1e9,
                'unit_spikes':r['refiner_cost']['unit_spikes'],
                'ac_reduction_vs_uncalibrated_pct':100*r.get('ac_reduction_vs_uncalibrated',0.)})


def evaluate(cache,calibration_directory,output,chunk_rows=8192):
    if type(chunk_rows) is not int or chunk_rows<1:raise ValueError('Positive chunk size required')
    calibration_path=Path(calibration_directory)/'calibration.json'
    calibration=read_json(calibration_path)
    expected=identity('encoder_calibration')
    if (calibration.get('status')!='complete' or calibration.get('data_role')!='train_main'
            or calibration.get('labels_used') is not False or calibration.get('dev_used_for_selection') is not False
            or any(calibration['identity'].get(k)!=v for k,v in expected.items())):
        raise ValueError('Freeze matching Train-main calibration before Dev evaluation')
    requested={(s,t) for s in protocol()['stages'] for t in protocol()['steps']}
    if {(p['stage'],p['steps']) for p in calibration['points']}!=requested or len(calibration['points'])!=len(requested):
        raise ValueError('Incomplete calibration selection')
    manifest=cache_manifest(cache); layers=load_layers()
    spec={**identity('calibrated_development_evaluation'),'calibration_sha256':sha256(calibration_path),
          'dev_cache_manifest_sha256':sha256(Path(cache)/'manifest.json'),'chunk_rows':chunk_rows}
    with lock_directory(output,spec) as directory:
        final=directory/'comparison.json'
        if final.exists():
            report=read_json(final)
            if report['identity']!=spec:raise ValueError('Evaluation report identity mismatch')
            return report
        atomic_json(directory/'checks.json',checks())
        plan: list[tuple[str,int,str,tuple[Encoder,...]]] = [('ann',255,'ann',(Encoder(),Encoder()))]
        for p in calibration['points']:
            for mode in ('uncalibrated','accuracy','low_spike'):
                plan.append((p['stage'],p['steps'],mode,tuple(Encoder(**e) for e in p[mode]['encoders'])))
        results=[]; originals={}
        for stage,t,mode,encoders in plan:
            atomic_json(directory/'progress.json',{'status':'evaluating','stage':stage,'T':t,'mode':mode,
                'completed_points':len(results),'total_points':len(plan),'training_started':False})
            result=evaluate_point(cache,manifest,layers,stage,t,mode,encoders,directory,chunk_rows)
            baseline=results[0] if results else result
            result['delta_to_frozen_ann_pp']={g:{m:100*(result[g][m]-baseline[g][m]) for m in ('iou','precision','recall')}
                                             for g in ('overall','ordinary','high_density')}
            if mode=='uncalibrated':originals[(stage,t)]=result
            if stage!='ann':
                uncal=originals[(stage,t)]
                result['delta_to_uncalibrated_pp']={g:{m:100*(result[g][m]-uncal[g][m]) for m in ('iou','precision','recall')}
                                                  for g in ('overall','ordinary','high_density')}
                result['ac_reduction_vs_uncalibrated']=1-result['executed_neural_ac']/uncal['executed_neural_ac'] if uncal['executed_neural_ac'] else 0.
                result['spike_reduction_vs_uncalibrated']=1-result['refiner_cost']['unit_spikes']/uncal['refiner_cost']['unit_spikes'] if uncal['refiner_cost']['unit_spikes'] else 0.
            atomic_json(directory/f'{stage}_t{t}_{mode}.json',result); results.append(result)
        report={'version':spec['version'],'status':'complete','identity':spec,'protocol':protocol(),
            'training_started':False,'weights_updated':False,'data_role':'train_dev',
            'complete_train_dev':True,'reserved_sequences_read':0,'decision_threshold_retuned':False,
            'dev_feedback_to_calibration':False,'calibration_sample':calibration['sample'],
            'results':results,'frozen_baseline_after':verify_baseline(),
            'cost_scope':'Proposal + Refiner synaptic MAC/AC. Refiner scalar/IF counts separate; incomplete end-to-end accounting. No energy measurement.'}
        atomic_json(final,report); write_table(directory,report)
        atomic_json(directory/'progress.json',{'status':'complete','training_started':False,
            'completed_points':len(results),'comparison':str(final)})
        return report
