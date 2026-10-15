"""Main selection seal, complete Dev validation, preserved original experiments."""
from __future__ import annotations
import json
from pathlib import Path
import time
import numpy as np
import torch

from ..h2pro_distill_v1.evaluation import summarize, delta
from ..h2pro_snn.diagnostics import error_counts
from ..h2pro_calibration_v1.core import Encoder, forward as shared_forward
from .common import (DEFAULT_CACHE, DEFAULT_RUN, IMPLEMENTATION, PRIOR, PROTOCOL, atomic_json,
                     identity, load_role, lock_directory, output_path, parameter_id, read_json,
                     save_arrays, sha256, timestamp, tree_hashes, verify_baseline)
from .core import Channels, cost, defaults, forward, load_layers, probabilities
from .search import Runner, TaskRunner, fit_branch, objective, sample, setup

GROUPS=('overall','ordinary','high_density')


def checkpoint_encoders(records):
    return tuple(Channels.read(e) for e in records)


def historical_parameters(stage,steps):
    prior=read_json(PRIOR)
    if prior['data_role'] != 'train_main' or prior['dev_used_for_selection'] or prior['weights_updated']:
        raise ValueError('Require immutable Main-only shared calibration')
    point=next(p for p in prior['points'] if p['stage']==stage and p['steps']==steps)
    return {name:tuple(Channels.shared(width,e['threshold'],e['initial_membrane'])
                       for width,e in zip((64,32),point[name]['encoders']))
            for name in ('uncalibrated','mse_reference','task_rescore','task','task_matched_ac')}


def preserve_inputs(directory):
    marker=directory/'preserved_inputs.json'
    if marker.exists():
        saved=read_json(marker)
    else:
        names=['h2pro_distill_r2t8_v1_s42','h2pro_distill_r2t8_v1_s42_epoch5_snapshot_20261014',
               'h2pro_hardcase_v1_s42','h2pro_fixed_diagnosis_v1','h2pro_task_calibration_v1_fit',
               'h2pro_task_calibration_v1_dev','h2pro_calibration_v1_fit_fine','h2pro_calibration_v1_dev',
               'h2pro_direct_v1_spike_add']
        saved={}
        for name in names:
            path=IMPLEMENTATION/'results/runs'/name
            if path.exists():saved.update(tree_hashes(path))
        atomic_json(marker,saved)
    for name,digest in saved.items():
        if sha256(IMPLEMENTATION/name) != digest:
            raise ValueError('Historical experiment changed: '+name)
    return len(saved)


@torch.no_grad()
def evaluate_config(sequences,layers,device,stage,steps,encoders,chunk_rows=32768,explicit=False,return_raw=False):
    rows=[]; parts=[]; spike_sums=np.zeros(2,np.int64); n=0; verified=0
    started=time.perf_counter()
    for s in sequences:
        outputs=[]; ns=np.zeros(2,np.int64)
        for lo in range(0,len(s['features']),chunk_rows):
            features=s['features'][lo:lo+chunk_rows]
            runner=Runner(features,layers,device)
            # Every reference ANN output must still equal the original cache.
            np.testing.assert_array_equal(runner.teacher.cpu().numpy(),s['teacher'][lo:lo+len(features)])
            raw,n1,n2=runner.forward(stage,steps,encoders)
            out=raw.cpu().numpy(); outputs.append(out)
            ns+=np.array([int(n1.sum()),int(n2.sum())],np.int64)
        raw=np.concatenate(outputs) if outputs else np.empty((0,4),np.float32)
        if explicit and len(raw):
            ids=np.unique(np.linspace(0,len(raw)-1,min(64,len(raw)),dtype=np.int64))
            golden,c=forward(s['features'][ids],layers,stage,steps,encoders,'spike_add')
            reference,refcost=forward(s['features'][ids],layers,stage,steps,encoders,'count_reference')
            np.testing.assert_array_equal(golden,raw[ids]);np.testing.assert_array_equal(golden,reference)
            assert c['equivalent_ac']==refcost['equivalent_ac']==c['executed_explicit_synaptic_ac']
            verified+=len(ids)
        full=np.zeros(s['row']['events'],np.float32);full[s['candidate']]=probabilities(raw)
        rows.append({'sequence':s['row']['sequence'],'high_density':s['row']['high_density'],
                     'counts':error_counts(s['labels'],full,s['candidate'],.35),'hidden_spikes':ns.tolist()})
        spike_sums+=ns;n+=len(raw)
        if return_raw:parts.append(raw)
    c=cost(n,stage,steps,encoders,spike_sums)
    c['proposal_sparse_mac']=sum(s['row']['proposal_sparse_mac'] for s in sequences)
    c['proposal_dense_mac']=sum(s['row']['proposal_dense_mac'] for s in sequences)
    c['mac_sparse_proposal']=c['refiner_equivalent_mac']+c['proposal_sparse_mac']
    c['mac_dense_proposal']=c['refiner_equivalent_mac']+c['proposal_dense_mac']
    c['synaptic_mac_plus_decoder_multiplications']=c['mac_sparse_proposal']+c['base_decode_scale_mul']+c['extra_threshold_decode_mul']
    c['unaccounted']=['feature extraction','routing','prototype readout','physical instruction costs','memory traffic']
    result={g:summarize([r['counts'] for r in rows if g=='overall' or r['high_density']==(g=='high_density')]) for g in GROUPS}
    result.update(stage=stage,steps=steps,encoders=[e.record() for e in encoders],parameter_id=parameter_id(encoders),
                  threshold=.35,role=sequences[0]['row']['role'],sequences=rows,cost=c,
                  verified_explicit_rows=verified,seconds=time.perf_counter()-started,
                  backend='FP64 exact integer-count reference; explicit spike-add checks separately counted')
    return result,np.concatenate(parts) if parts else None


def select(candidates,ann,policy,role):
    if role != 'train_main' or policy not in ('iou','recall_preserving'):
        raise ValueError('Parameter selection restricted to Main and fixed policies')
    eligible=list(candidates)
    if policy=='recall_preserving':
        eligible=[c for c in eligible if all(c[g]['tp']>=ann[g]['tp'] for g in ('overall','high_density'))]
    if not eligible:return {'status':'infeasible','policy':policy,'parameter_id':None}
    best=min(eligible,key=lambda c:(-c['overall']['iou'],-c['high_density']['recall'],
                -c['overall']['recall'],c['cost']['equivalent_ac'],c['parameter_id']))
    return {'status':'selected','policy':policy,'parameter_id':best['parameter_id'],
            'encoders':best['encoders'],'main':{g:best[g] for g in GROUPS},'main_cost':best['cost']}


def calibrate(cache,directory,device,chunk_rows=32768):
    if (directory/'selection.json').exists():
        selected=read_json(directory/'selection.json')
        for name,h in selected['main_evidence_sha256'].items():
            if sha256(directory/name)!=h:raise ValueError('Sealed Main evidence changed')
        return selected
    atomic_json(directory/'progress.json',{'status':'loading_main','training_started':False})
    seqs=load_role(cache,'train_main');layers=load_layers()
    data=sample(seqs);runner=TaskRunner(data,layers,device)
    grouping,diagnostics=runner.grouping()
    atomic_json(directory/'sample.json',{'role':'train_main','protocol':PROTOCOL['sample'],
        'sequences':data['receipts'],'sampled_candidates':len(data['features']),
        'sampled_targets':int(data['labels'].sum()),'population_candidates':sum(len(s['features']) for s in seqs),
        'groups':grouping,'original_hidden_diagnostics':diagnostics})
    ann,_=evaluate_config(seqs,layers,device,'ann',8,defaults(),chunk_rows)
    atomic_json(directory/'main_ann.json',ann)
    selections={};points=[]
    for stage,steps in PROTOCOL['points']:
        key=f'{stage}_t{steps}';pointdir=directory/'fit'/key;pointdir.mkdir(parents=True,exist_ok=True)
        prior=historical_parameters(stage,steps)
        assessed={name:runner.assess(stage,steps,e) for name,e in prior.items()}
        start_name=min(prior,key=lambda name:objective(assessed[name],prior[name]))
        fits={}
        for branch in PROTOCOL['branches']:
            atomic_json(directory/'progress.json',{'status':'calibrating','point':key,'branch':branch,
                        'cursor_file':f'fit/{key}/{branch}_cursor.json','training_started':False})
            if branch=='shared':initial=prior[start_name]
            elif branch.startswith('group'):initial=checkpoint_encoders(fits['shared']['final'])
            else:initial=checkpoint_encoders(fits['group_m0' if branch.endswith('m0') else 'group_joint']['final'])
            fits[branch]=fit_branch(runner,stage,steps,branch,initial,grouping,pointdir)
        pools={}
        for branch,fit in fits.items():
            pools[branch]={parameter_id(checkpoint_encoders(s['encoders'])):s['encoders'] for s in fit['history']}
        # Scalar branch also includes every immutable historical shared control.
        pools['shared'].update({parameter_id(e):[x.record() for x in e] for e in prior.values()})
        all_params={pid:e for pool in pools.values() for pid,e in pool.items()}
        scored={};maindir=directory/'main'/key;maindir.mkdir(parents=True,exist_ok=True)
        for index,(pid,params) in enumerate(sorted(all_params.items()),1):
            atomic_json(directory/'progress.json',{'status':'full_main_validation','point':key,
                         'candidate':index,'candidates':len(all_params),'training_started':False})
            receipt=maindir/f'{pid}.json'
            if receipt.exists():
                result=read_json(receipt)
                if result['encoders'] != params or result['role'] != 'train_main':raise ValueError('Full-Main receipt mismatch')
            else:
                result,_=evaluate_config(seqs,layers,device,stage,steps,checkpoint_encoders(params),chunk_rows)
                atomic_json(receipt,result)
            scored[pid]=result
            print(json.dumps({'phase':'full_main','point':key,'candidate':index,'total':len(all_params),
                              'iou':result['overall']['iou'],'hd_recall':result['high_density']['recall']}),flush=True)
        choices={branch:{policy:select([scored[k] for k in pool],ann,policy,'train_main')
                         for policy in PROTOCOL['full_main_selection']} for branch,pool in pools.items()}
        point={'stage':stage,'steps':steps,'choices':choices,'grouping':grouping,'start_name':start_name,
               'old_shared_controls':{name:parameter_id(e) for name,e in prior.items()},
               'pool_members':{b:sorted(p) for b,p in pools.items()},
               'fit_candidates':{b:fit['evaluated'] for b,fit in fits.items()}}
        atomic_json(directory/f'{key}_main_selection.json',point)
        selections[key]=point;points.append(point)
    main_files={p.relative_to(directory).as_posix():sha256(p) for folder in ('main','fit')
                for p in sorted((directory/folder).rglob('*.json'))}
    for name in ('sample.json','main_ann.json',*[f'{p}_main_selection.json' for p in selections]):
        main_files[name]=sha256(directory/name)
    selected={'status':'complete','selection_role':'train_main','created_at_utc':timestamp(),
              'points':selections,'main_evidence_sha256':main_files,'training_started':False,
              'weights_updated':False,'dev_loaded_before_seal':False}
    atomic_json(directory/'selection.json',selected)
    atomic_json(directory/'progress.json',{'status':'main_selection_sealed','training_started':False})
    return selected


def compare_historical(ann,rows):
    """Check historical controls before presenting any new calibrated metrics."""
    old=read_json(IMPLEMENTATION/'results/runs/h2pro_task_calibration_v1_dev/comparison.json')['results']
    if isinstance(old,dict):old=list(old.values())
    matches=0
    for row in rows:
        if row['name'].startswith('historical/'):
            mode=row['name'].split('/')[1]
            expected=next(v for v in old if v['stage']==row['stage'] and v['steps']==row['steps'] and v['mode']==mode)
            for g in GROUPS:
                for metric in ('tp','fp','fn','iou','precision','recall'):
                    assert row[g][metric]==expected[g][metric],(row['name'],g,metric)
            assert row['cost']['mac_dense_proposal']==expected['executed_neural_mac']
            assert row['cost']['equivalent_ac']==expected['executed_neural_ac']
            matches+=1
    source=read_json(IMPLEMENTATION/'results/runs/h2pro_distill_r2t8_v1_s42/training.json')
    for g in GROUPS:
        for metric in ('tp','fp','fn','iou','precision','recall'):
            assert ann[g][metric]==source['frozen_ann'][g][metric]
    return matches


def evaluate(cache,directory,device,chunk_rows=32768):
    path=directory/'comparison.json'
    if path.exists():
        result=read_json(path)
        if result['selection_sha256']!=sha256(directory/'selection.json'):
            raise ValueError('Dev result selection seal mismatch')
        for name,h in result['dev_evidence_sha256'].items():
            if sha256(directory/name)!=h:raise ValueError('Dev evidence changed')
        return result
    selected=read_json(directory/'selection.json')
    if selected['selection_role']!='train_main' or selected['dev_loaded_before_seal']:
        raise ValueError('Require sealed Main-only choices')
    started=timestamp()
    if selected['created_at_utc']>started:raise ValueError('Invalid Main/Dev ordering')
    atomic_json(directory/'progress.json',{'status':'validating_dev','selection_sha256':sha256(directory/'selection.json'),
                                         'training_started':False})
    seqs=load_role(cache,'train_dev');layers=load_layers()
    devdir=directory/'dev';devdir.mkdir(parents=True,exist_ok=True)
    ann,raw=evaluate_config(seqs,layers,device,'ann',8,defaults(),chunk_rows,explicit=True,return_raw=True)
    atomic_json(devdir/'ann.json',ann);save_arrays(devdir/'ann_predictions.npz',raw=raw)
    results=[];seen={}
    for point in selected['points'].values():
        stage,steps=point['stage'],point['steps']
        candidates=[(f'historical/{name}',e,'fixed_prior') for name,e in historical_parameters(stage,steps).items()]
        candidates += [(f'{branch}/{policy}',checkpoint_encoders(choice['encoders']),policy)
                       for branch,choices in point['choices'].items() for policy,choice in choices.items()
                       if choice['status']=='selected']
        for name,encoders,policy in candidates:
            pid=parameter_id(encoders);key=f'{stage}_t{steps}_{pid}';receipt=devdir/f'{key}.json'
            prediction=devdir/f'{key}_predictions.npz'
            if key in seen:result=seen[key]
            elif receipt.exists():
                result=read_json(receipt)
                if (result['encoders']!=[e.record() for e in encoders] or result['selection_sha256']!=sha256(directory/'selection.json')
                        or sha256(prediction)!=result['predictions_sha256']):raise ValueError('Dev resume mismatch')
            else:
                result,raw=evaluate_config(seqs,layers,device,stage,steps,encoders,chunk_rows,explicit=True,return_raw=True)
                save_arrays(prediction,raw=raw)
                result.update(selection_sha256=sha256(directory/'selection.json'),predictions_sha256=sha256(prediction),
                              delta_to_ann_pp=delta(result,ann))
                atomic_json(receipt,result)
            seen[key]=result
            results.append(dict(result,name=name,policy=policy))
            print(json.dumps({'phase':'dev','point':f'{stage}_t{steps}','name':name,
                               'iou':result['overall']['iou'],'recall':result['overall']['recall'],
                               'hd_recall':result['high_density']['recall'],'AC':result['cost']['equivalent_ac']}),flush=True)
    replayed=compare_historical(ann,results)
    original=read_json(IMPLEMENTATION/'results/runs/h2pro_distill_r2t8_v1_s42/epochs/epoch_088.json')['evaluation']
    hard=read_json(IMPLEMENTATION/'results/runs/h2pro_hardcase_v1_s42/hard_weighted/training.json')
    hard_best=hard['epochs'][hard['best_added_epoch']-1]['evaluation']
    memory=({'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
             'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)} if device.type=='cuda' else {})
    result={'status':'complete','version':PROTOCOL['version'],'protocol':PROTOCOL,'selection_sha256':sha256(directory/'selection.json'),
            'main_selection_utc':selected['created_at_utc'],'dev_started_utc':started,'data_role':'train_dev',
            'training_started':False,'weights_updated':False,'dev_feedback_to_calibration':False,
            'ann':ann,'results':results,'reference_trained_weights_only_for_comparison':{'epoch88':original,'direction5_best_added19':hard_best},
            'historical_controls_exact':replayed,'explicit_unique_configs':len(seen)+1,
            'memory':memory,'baseline_after':verify_baseline(),'preserved_input_files':preserve_inputs(directory),
            'dev_evidence_sha256':{p.relative_to(directory).as_posix():sha256(p) for p in sorted(devdir.iterdir()) if p.is_file()}}
    atomic_json(path,result)
    atomic_json(directory/'progress.json',{'status':'complete','training_started':False,'weights_updated':False})
    return result


def run(action,cache=DEFAULT_CACHE,directory=DEFAULT_RUN,device='cuda',chunk_rows=32768):
    cache=Path(cache).resolve();directory=output_path(directory)
    if directory==cache or cache in directory.parents or directory in cache.parents:
        raise ValueError('Use a separate new output directory')
    d,engine=setup(device);spec=identity(cache,engine)
    with lock_directory(directory,spec):
        preserve_inputs(directory)
        if action in ('calibrate','all'):calibrate(cache,directory,d,chunk_rows)
        if action in ('evaluate','all'):return evaluate(cache,directory,d,chunk_rows)
        return read_json(directory/'progress.json') if (directory/'progress.json').exists() else {'status':'not_started'}
