"""Fit shortlist -> complete Main internal validation -> seal -> complete Dev."""
import json
from pathlib import Path
import numpy as np
import torch

from ..h2pro_channel_calibration_v1.workflow import evaluate_config
from ..h2pro_channel_calibration_v1.core import load_layers,probabilities,Channels
from ..h2pro_distill_v1.evaluation import summarize,delta
from .common import (DEFAULT_CACHE,DEFAULT_RUN,OLD_RUN,PROTOCOL,IMPLEMENTATION,atomic_json,
    read_json,sha256,timestamp,load_role,parameter_id,save_arrays,identity,output_path,
    lock_directory,preserve,anchor,split_sequences,sample_fit,encoders,cost,constraints,rank)
from .search import FitRunner,fit,setup

GROUPS=('overall','ordinary','high_density')


def verify_hashes(directory,hashes):
    for name,digest in hashes.items():
        if sha256(directory/name)!=digest:raise ValueError('Sealed evidence changed: '+name)


def partition(record,seqs,names):
    wanted=set(names);rows=[r for r in record['sequences'] if r['sequence'] in wanted]
    source=[s['row'] for s in seqs if s['row']['sequence'] in wanted]
    if len(rows)!=len(wanted):raise ValueError('Incomplete partition')
    result={g:summarize([r['counts'] for r in rows if g=='overall' or r['high_density']==(g=='high_density')]) for g in GROUPS}
    n=sum(r['candidate_events'] for r in source)
    e=tuple(Channels.read(x) for x in record['encoders'])
    spikes=np.asarray([r['hidden_spikes'] for r in rows],np.int64).sum(0)
    c=cost(n,record['stage'],16,e,spikes)
    c.update(proposal_sparse_mac=sum(r['proposal_sparse_mac'] for r in source),
             proposal_dense_mac=sum(r['proposal_dense_mac'] for r in source))
    c['mac_sparse_proposal']=c['proposal_sparse_mac']+c['refiner_equivalent_mac']
    c['mac_dense_proposal']=c['proposal_dense_mac']+c['refiner_equivalent_mac']
    result.update(cost=c,parameter_id=record['parameter_id'],sequences=sorted(wanted))
    return result


def choose(records):
    reference=next(r for r in records if not r['changes'])
    for r in records:
        r['eligibility_failures']={scope:constraints(r['partitions'][scope],reference['partitions'][scope])
                                   for scope in ('fit','validation','all')}
        r['eligible']=not any(r['eligibility_failures'].values())
    def order(r):
        return rank(r['partitions']['validation'],reference['partitions']['validation'],r['changes'],r['eligible'])
    choices={'anchor':reference}
    for number in (1,2):
        pool=[r for r in records if len(r['changes'])==number]
        if pool:choices[f'changed_{number}_group']=min(pool,key=order)
    choices['primary']=min([r for r in records if r['eligible']],key=order)
    return {key:{'parameter_id':r['parameter_id'],'changes':r['changes'],'encoders':r['encoders'],
                 'eligible':r['eligible'],'eligibility_failures':r['eligibility_failures'],
                 'main':r['partitions']} for key,r in choices.items()}


def calibrate(cache,directory,device,chunk_rows=32768):
    seal=directory/'selection.json'
    if seal.exists():
        saved=read_json(seal);verify_hashes(directory,saved['main_evidence_sha256']);return saved
    seqs=load_role(cache,'train_main')
    split=split_sequences([s['row'] for s in seqs])
    split_path=directory/'split.json'
    if split_path.exists() and read_json(split_path)!=split:raise ValueError('Main split changed')
    atomic_json(split_path,split)
    fit_seqs=[s for s in seqs if s['row']['sequence'] in set(split['fit'])]
    layers=load_layers();shortlist_path=directory/'fit_selection.json'
    if not shortlist_path.exists():
        atomic_json(directory/'progress.json',{'status':'loading_fit_sample','training_started':False})
        data=sample_fit(fit_seqs);runner=FitRunner(data,layers,device);groups,means=runner.grouping()
        sample_record={'protocol':PROTOCOL['sample'],'scope':'main_fit_sequences_only',
                       'sampled_candidates':len(data['labels']),'sampled_positives':int(data['labels'].sum()),
                       'sampled_background':int((data['labels']==0).sum()),'population':data['population'],
                       'receipts':data['receipts'],'groups':groups,'weighted_hidden1_mean':means}
        if (directory/'sample.json').exists() and read_json(directory/'sample.json')!=sample_record:
            raise ValueError('Fit sample changed on resume')
        atomic_json(directory/'sample.json',sample_record)
        shortlist=fit(runner,groups,directory)
        del runner,data
    else:
        shortlist=read_json(shortlist_path);verify_hashes(directory,shortlist['fit_evidence_sha256'])
        groups=shortlist['groups']
    validation_started=timestamp()
    folder=directory/'main';folder.mkdir(parents=True,exist_ok=True);scored=[]
    all_names=[s['row']['sequence'] for s in seqs]
    for i,item in enumerate(shortlist['candidates'],1):
        e=encoders(item['changes'],groups);pid=parameter_id(e);path=folder/f'{pid}.json';pred=folder/f'{pid}_probabilities.npz'
        if path.exists():
            result=read_json(path)
            if (result['changes']!=item['changes'] or result['encoders']!=[x.record() for x in e]
                    or result['fit_selection_sha256']!=sha256(shortlist_path)
                    or result['predictions_sha256']!=sha256(pred)):
                raise ValueError('Complete Main resume mismatch')
        else:
            result,raw=evaluate_config(seqs,layers,device,'r1',16,e,chunk_rows,return_raw=True)
            save_arrays(pred,probability=probabilities(raw))
            result.update(changes=item['changes'],fit_selection_sha256=sha256(shortlist_path),predictions_sha256=sha256(pred))
            result['partitions']={scope:partition(result,seqs,names) for scope,names in
                                  [('fit',split['fit']),('validation',split['validation']),('all',all_names)]}
            atomic_json(path,result)
        scored.append(result)
        progress={'status':'full_main','candidate':i,'total':len(shortlist['candidates']),
                  'iou':result['overall']['iou'],'internal_validation_iou':result['partitions']['validation']['overall']['iou'],
                  'training_started':False}
        atomic_json(directory/'progress.json',progress);print(json.dumps(progress),flush=True)
    reference=next(r for r in scored if not r['changes'])
    old=read_json(OLD_RUN/'main/r1_t16'/f"{reference['parameter_id']}.json")
    for g in GROUPS:assert reference[g]==old[g],('Main anchor replay',g)
    assert reference['cost']==old['cost'],'Main anchor cost replay'
    choices=choose(scored)
    hashes={p.relative_to(directory).as_posix():sha256(p) for folder_name in ('main','fit_candidates')
            for p in sorted((directory/folder_name).iterdir()) if p.is_file()}
    for name in ('split.json','sample.json','fit_selection.json'):hashes[name]=sha256(directory/name)
    result={'status':'complete','selection_role':'train_main','created_at_utc':timestamp(),
            'fit_shortlist_sealed_at_utc':shortlist['created_at_utc'],'internal_validation_started_utc':validation_started,
            'choices':choices,'split':split,'main_evidence_sha256':hashes,
            'fit_search_candidates':shortlist['evaluated_unique_parameters'],'full_main_candidates':len(scored),
            'anchor_exact_replay':True,'dev_loaded_before_seal':False,'training_started':False,'weights_updated':False}
    atomic_json(seal,result);atomic_json(directory/'progress.json',{'status':'main_sealed','training_started':False})
    return result


def evaluate(cache,directory,device,chunk_rows=32768):
    seal=directory/'selection.json'
    if not seal.exists():raise ValueError('Seal Main choices before Dev validation')
    selected=read_json(seal);verify_hashes(directory,selected['main_evidence_sha256'])
    if selected['selection_role']!='train_main' or selected['dev_loaded_before_seal']:
        raise ValueError('Invalid Main selection')
    final=directory/'comparison.json'
    if final.exists():
        result=read_json(final)
        if result['selection_sha256']!=sha256(seal):raise ValueError('Selection identity changed')
        verify_hashes(directory,result['dev_evidence_sha256']);return result
    started=timestamp();seqs=load_role(cache,'train_dev');layers=load_layers()
    folder=directory/'dev';folder.mkdir(parents=True,exist_ok=True)
    ann_path=folder/'ann.json'
    if ann_path.exists():
        ann=read_json(ann_path)
        if ann['predictions_sha256']!=sha256(folder/'ann_predictions.npz'):raise ValueError('ANN predictions changed')
    else:
        ann,raw=evaluate_config(seqs,layers,device,'ann',16,anchor(),chunk_rows,explicit=True,return_raw=True)
        save_arrays(folder/'ann_predictions.npz',raw=raw)
        ann['predictions_sha256']=sha256(folder/'ann_predictions.npz');atomic_json(ann_path,ann)
    rows=[];seen={}
    for name,choice in selected['choices'].items():
        pid=choice['parameter_id'];e=tuple(Channels.read(x) for x in choice['encoders'])
        receipt=folder/f'{pid}.json';prediction=folder/f'{pid}_predictions.npz'
        if pid in seen:record=seen[pid]
        elif receipt.exists():
            record=read_json(receipt)
            if (record['encoders']!=choice['encoders'] or record['selection_sha256']!=sha256(seal)
                    or record['predictions_sha256']!=sha256(prediction)):
                raise ValueError('Dev resume mismatch')
        else:
            record,raw=evaluate_config(seqs,layers,device,'r1',16,e,chunk_rows,explicit=True,return_raw=True)
            save_arrays(prediction,raw=raw)
            record.update(selection_sha256=sha256(seal),predictions_sha256=sha256(prediction))
            atomic_json(receipt,record)
        seen[pid]=record
        rows.append(dict(record,name=name,changes=choice['changes'],main_eligible=choice['eligible'],
                         main_eligibility_failures=choice['eligibility_failures'],delta_ann_pp=delta(record,ann)))
        print(json.dumps({'phase':'dev','name':name,'iou':record['overall']['iou'],
                          'recall':record['overall']['recall'],'hd_recall':record['high_density']['recall'],
                          'AC':record['cost']['equivalent_ac'],'main_eligible':choice['eligible']}),flush=True)
    previous=read_json(OLD_RUN/'comparison.json')
    old_anchor=next(r for r in previous['results'] if r['stage']=='r1' and r['name']=='historical/task_rescore')
    current_anchor=next(r for r in rows if r['name']=='anchor')
    for g in GROUPS:
        assert ann[g]==previous['ann'][g],('Dev ANN replay',g)
        assert current_anchor[g]==old_anchor[g],('Dev anchor replay',g)
    assert current_anchor['cost']==old_anchor['cost']
    for r in rows:r['delta_anchor_pp']=delta(r,current_anchor)
    references={'previous_direction2_group_joint':next(r for r in previous['results'] if r['stage']=='r1' and r['name']=='group_joint/iou'),
                **previous['reference_trained_weights_only_for_comparison']}
    result={'status':'complete','version':PROTOCOL['version'],'protocol':PROTOCOL,'selection_sha256':sha256(seal),
            'main_selection_utc':selected['created_at_utc'],'dev_started_utc':started,'dev_feedback_to_calibration':False,
            'ann':ann,'results':rows,'references':references,'training_started':False,'weights_updated':False,
            'preserved_input_files':preserve(directory),'old_anchor_and_ann_exact':True,
            'explicit_unique_configs':len(seen)+1,
            'memory':({'peak_allocated_bytes':torch.cuda.max_memory_allocated(device),
                       'peak_reserved_bytes':torch.cuda.max_memory_reserved(device)} if device.type=='cuda' else {}),
            'dev_evidence_sha256':{p.relative_to(directory).as_posix():sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}}
    atomic_json(final,result);atomic_json(directory/'progress.json',{'status':'complete','training_started':False,'weights_updated':False})
    return result


def run(action,cache=DEFAULT_CACHE,directory=DEFAULT_RUN,device='cuda',chunk_rows=32768):
    if action not in ('all','calibrate','evaluate'):raise ValueError('Unknown action')
    cache=Path(cache).resolve();directory=output_path(directory)
    if directory==cache or cache in directory.parents or directory in cache.parents:
        raise ValueError('Use an independent run directory')
    d,engine=setup(device);spec=identity(cache,engine)
    with lock_directory(directory,spec):
        preserve(directory)
        if action in ('all','calibrate'):calibrate(cache,directory,d,chunk_rows)
        if action in ('all','evaluate'):return evaluate(cache,directory,d,chunk_rows)
        return read_json(directory/'progress.json')
