"""Stage-balanced paired continuations with atomic minibatch recovery."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import time
import traceback

import numpy as np
import torch

from ..h2pro_distill_v1.data import ResidentData
from ..h2pro_distill_v1.evaluation import delta, evaluate, export
from ..h2pro_distill_v1.model import R2T8
from ..h2pro_distill_v1.runtime import (setup_device, memory, checkpoint, restore,
                                      fresh_totals as original_totals, add_totals as original_add)
from ..h2pro_longtrain_v1.storage import PauseControl, PauseRequested
from .common import (ARMS, MILESTONES, SOURCE_EPOCH, Config, DEFAULT_CACHE, DEFAULT_RUN, SOURCE_RUN,
                     DEFAULT_DIAGNOSIS, atomic_json, lock_directory, inputs, independent_output,
                     read_json, sha256, arm_identity)
from .evaluation import fit_main, finish_milestone
from .objective import update


def fresh_totals():
    return dict(original_totals(),plain_loss_sums=[0.,0.,0.],category_counts=[0,0,0])


def initial_state(origin, baseline):
    return {'cursor': {'epoch': 1,'batch':0,'phase':'train'}, 'global_step': origin['state']['global_step'],
            'origin_global_step': origin['state']['global_step'], 'source_epoch': SOURCE_EPOCH,
            'totals': fresh_totals(), 'history': [], 'best_epoch': 0,
            'best_score': [baseline['overall']['iou'],baseline['overall']['recall']]}


def initialized_payload(origin, identity, baseline):
    return {'identity': identity,'model':origin['model'],'optimizer':origin['optimizer'],'rng':origin['rng'],
            'state':initial_state(origin,baseline),'resumable':True}


def attach_hard_categories(data, suite, ready):
    path = suite/'main_hard_categories.npz'
    if sha256(path) != ready['hard_categories_sha256']:
        raise ValueError('Fixed Main hard mask changed')
    with np.load(path,allow_pickle=False) as z:
        category=z['category']
    if category.shape != (data.size('train_main'),) or category.dtype != np.uint8 or not np.isin(category,[0,1,2]).all():
        raise ValueError('Invalid full Main category map')
    y=data.roles['train_main']['labels'].cpu().numpy()
    if ((category==1)&(y!=1)).any() or ((category==2)&(y!=0)).any():
        raise ValueError('Hard positives/negatives are not aligned to Main labels')
    data.roles['train_main']['hard_category']=torch.from_numpy(category.copy()).to(data.device)
    data.stats=dict(data.stats,hard_category_gpu_bytes=int(category.nbytes),hard_category_role='train_main',
                    fixed_category_counts=np.bincount(category,minlength=3).tolist())


def status(suite):
    workflow=read_json(suite/'suite_status.json').get('status') if (suite/'suite_status.json').exists() else 'not_prepared'
    return {'suite':str(suite),'workflow_status':workflow,'pause_requested':(suite/'PAUSE').exists(),
            'arms':{arm:read_json(suite/arm/'progress.json') if (suite/arm/'progress.json').exists()
                                    else {'status':'not_prepared'} for arm in ARMS},
            'completed_comparisons':[e for e in MILESTONES if (suite/f'comparison_added_{e:03}.json').exists()]}


def emit(directory, arm, state, data, target, status_name):
    report={'status':status_name,'arm':arm,'completed_additional_epochs':len(state['history']),
            'source_epoch':SOURCE_EPOCH,'effective_completed_epoch':SOURCE_EPOCH+len(state['history']),
            'target_additional_epochs':target,'cursor':state['cursor'],'global_step':state['global_step'],
            'additional_updates':state['global_step']-state['origin_global_step'],
            'best_added_epoch':state['best_epoch'],'totals':state['totals'], 'data':data.stats,
            'resumable_checkpoint':str(directory/'last.pt'),'updated_unix_s':time.time(),**memory(data.device)}
    atomic_json(directory/'progress.json',report)
    with (directory/'progress.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(report,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('data','totals')},allow_nan=False),flush=True)
    return report


def validate_cursor(state, data, config):
    c=state['cursor']; n=data.size('train_main')
    batches=(n+config.batch_size-1)//config.batch_size
    if (c['epoch']!=len(state['history'])+1 or c['phase'] not in ('train','evaluate')
            or not 0<=c['batch']<=batches
            or (c['phase']=='evaluate' and c['batch']!=batches)
            or state['totals']['events']!=min(c['batch']*config.batch_size,n)
            or sum(state['totals']['category_counts'])!=state['totals']['events']
            or state['global_step']-state['origin_global_step']!=len(state['history'])*batches+c['batch']):
        raise ValueError('Checkpoint cursor/update/sample coverage is inconsistent')


def train_arm(data, suite, arm, identity, ready, config, target, checkpoint_batches, pause=lambda:None,
              *, _test_pause_after=None):
    directory=suite/arm
    with lock_directory(directory,arm_identity(identity,arm,ready['hard_categories_sha256'])):
        model=R2T8().to(data.device)
        optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=0.,foreach=False)
        aid=arm_identity(identity,arm,ready['hard_categories_sha256'])
        state=restore(directory/'last.pt',model,optimizer,aid)
        validate_cursor(state,data,config)
        if any(float(s['step'])!=state['global_step'] for s in optimizer.state.values()):
            raise ValueError('Adam steps differ from committed update cursor')
        if data.device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(data.device)
        def save():
            checkpoint(directory/'last.pt',model,optimizer,state,aid)
        def history():
            atomic_json(directory/'training.json',{'identity':aid,'source_epoch':SOURCE_EPOCH,
                'initial_student':ready['initial_student'],'frozen_ann':ready['frozen_ann'],
                'best_added_epoch':state['best_epoch'],'epochs':state['history']})
        if len(state['history'])>=target:
            # A kill may occur after last.pt commits but before its JSON mirror.
            history()
            return {'status':'already_complete','completed_additional_epochs':len(state['history'])}
        batches=(data.size('train_main')+config.batch_size-1)//config.batch_size
        try:
            while state['cursor']['epoch']<=target:
                pause()
                c=state['cursor']; added=c['epoch']; absolute=SOURCE_EPOCH+added
                emit(directory,arm,state,data,target,'running')
                order=data.order(absolute,config.seed)
                if c['phase']=='train':
                    for batch_number in range(c['batch'],batches):
                        pause(); started=time.perf_counter()
                        ids=order[batch_number*config.batch_size:(batch_number+1)*config.batch_size]
                        batch=data.batch('train_main',ids)
                        result=update(model,optimizer,batch,config,arm)
                        original_add(state['totals'],result,time.perf_counter()-started)
                        state['totals']['plain_loss_sums']=[a+b*result['events'] for a,b in
                            zip(state['totals']['plain_loss_sums'],result['plain_losses'])]
                        counts=torch.bincount(batch['hard_category'].long(),minlength=3).cpu().tolist()
                        state['totals']['category_counts']=[a+b for a,b in zip(state['totals']['category_counts'],counts)]
                        c['batch']=batch_number+1; state['global_step']+=1
                        if c['batch']%checkpoint_batches==0:
                            save(); emit(directory,arm,state,data,target,'running')
                        if _test_pause_after is not None and state['global_step']-state['origin_global_step']==_test_pause_after:
                            raise PauseRequested()
                    expected=torch.bincount(data.roles['train_main']['hard_category'].long(),minlength=3).cpu().tolist()
                    if state['totals']['events']!=data.size('train_main') or state['totals']['category_counts']!=expected:
                        raise AssertionError('Epoch failed complete hard/ordinary sample coverage')
                    c['phase']='evaluate'; save()
                pause()
                is_milestone=added in MILESTONES or added==target
                selected=fit_main(model,data,config.batch_size,directory/'milestones',added,pause) if is_milestone else None
                validation_started=datetime.now(timezone.utc).isoformat()
                result,raw,spikes=evaluate(model,data,config.batch_size,pause)
                result['delta_to_ann_pp']=delta(result,ready['frozen_ann'])
                result['delta_to_epoch88_pp']=delta(result,ready['initial_student'])
                if selected is not None:
                    finish_milestone(data,raw,spikes,result,selected,directory/'milestones',added,
                                     ready['frozen_ann'],validation_started)
                totals=state['totals']; n=totals['events']
                entry={'added_epoch':added,'effective_epoch':absolute,'global_step':state['global_step'],
                    'training':dict(totals,mean_losses=[v/n for v in totals['loss_sums']],
                                   plain_mean_losses=[v/n for v in totals['plain_loss_sums']]),
                    'evaluation':result,'memory':memory(data.device)}
                rank=[result['overall']['iou'],result['overall']['recall']]
                state['history'].append(entry)
                state['cursor']={'epoch':added+1,'batch':0,'phase':'train'}; state['totals']=fresh_totals()
                epoch_dir=directory/'epochs'; epoch_dir.mkdir(exist_ok=True)
                atomic_json(epoch_dir/f'added_{added:03}.json',entry)
                export(epoch_dir/f'added_{added:03}_int8.npz',model)
                if rank>state['best_score']:
                    state['best_score']=rank; state['best_epoch']=added
                    checkpoint(directory/'best.pt',model,optimizer,state,aid)
                    export(directory/'best_int8.npz',model)
                if is_milestone:
                    checkpoint(directory/'milestones'/f'added_{added:03}.pt',model,optimizer,state,aid)
                save(); history(); emit(directory,arm,state,data,target,'epoch_complete')
            history()
            return emit(directory,arm,state,data,target,'complete')
        except PauseRequested:
            save(); history()
            return emit(directory,arm,state,data,target,'paused')
        except Exception as error:
            stamp=time.time_ns()
            atomic_json(directory/f'failure_{stamp}.json',{'error':repr(error),'traceback':traceback.format_exc(),
                'cursor':state['cursor'],'global_step':state['global_step'],'resumable_checkpoint':str(directory/'last.pt'),
                'last_sha256':sha256(directory/'last.pt')})
            emit(directory,arm,state,data,target,'failed')
            raise


def compare(suite, added, ready):
    paths={arm:suite/arm/'milestones'/f'added_{added:03}_dev.json' for arm in ARMS}
    if not all(p.exists() for p in paths.values()):
        raise ValueError('Both arms must complete this milestone before comparison')
    result={'additional_epochs_per_arm':added,'effective_epoch':SOURCE_EPOCH+added,
            'frozen_ann':ready['frozen_ann'],'source_epoch88':ready['initial_student'],
            'arms':{arm:read_json(path) for arm,path in paths.items()},
            'source_sha256':{arm:sha256(path) for arm,path in paths.items()},'weighted_minus_control_pp':{}}
    controls=result['arms']['control']['operating_points']
    weighted=result['arms']['hard_weighted']['operating_points']
    for policy in controls.keys() & weighted.keys():
        result['weighted_minus_control_pp'][policy]=delta(weighted[policy],controls[policy])
    for arm in ARMS:
        history=read_json(suite/arm/'training.json')['epochs']
        result['arms'][arm]['learning_trend']=[{'added_epoch':e['added_epoch'],
            'objective_losses':e['training']['mean_losses'],'unweighted_losses':e['training']['plain_mean_losses'],
            'iou':e['evaluation']['overall']['iou'],'precision':e['evaluation']['overall']['precision'],
            'recall':e['evaluation']['overall']['recall'],'hd_recall':e['evaluation']['high_density']['recall'],
            'cost':e['evaluation']['cost'],'memory':e['memory']} for e in history if e['added_epoch']<=added]
    path=suite/f'comparison_added_{added:03}.json'
    if path.exists():
        if read_json(path)!=result:
            raise ValueError('Existing milestone comparison changed')
    else:
        atomic_json(path,result)
    print(json.dumps({'stage':'paired_comparison','added_epochs':added,'path':str(path),
                      'weighted_minus_control_pp':result['weighted_minus_control_pp']}),flush=True)
    return result


def train_suite(cache=DEFAULT_CACHE,suite=DEFAULT_RUN,source=SOURCE_RUN,diagnosis=DEFAULT_DIAGNOSIS,
                additional_epochs=20,checkpoint_batches=10,device_name='cuda'):
    if type(additional_epochs) is not int or additional_epochs<1 or type(checkpoint_batches) is not int or checkpoint_batches<1:
        raise ValueError('Positive additional total epochs and checkpoint interval required')
    device=setup_device(device_name)
    identity,_,config=inputs(cache,source,diagnosis,device)
    suite=independent_output(suite,cache,source,diagnosis)
    with lock_directory(suite,identity):
        ready=read_json(suite/'ready.json')
        if ready.get('identity')!=identity or ready.get('status')!='passed':
            raise ValueError('Run matching check before training')
        for name,expected in ready.get('immutable_files_sha256',{}).items():
            if sha256(suite/name)!=expected:
                raise ValueError('Prepared immutable initialization changed: '+name)
        # Consume only a stale request at invocation start; never between arms.
        (suite/'PAUSE').unlink(missing_ok=True)
        targets=sorted({e for e in MILESTONES if e<=additional_epochs}|{additional_epochs})
        with PauseControl(suite) as control:
            def pause():
                if control.pending():
                    raise PauseRequested()
            try:
                atomic_json(suite/'suite_status.json',{'status':'running','target_additional_epochs':additional_epochs})
                data=ResidentData(cache,device); attach_hard_categories(data,suite,ready)
                for target in targets:
                    for arm in ARMS:
                        pause()
                        result=train_arm(data,suite,arm,identity,ready,config,target,checkpoint_batches,pause)
                        if result['status']=='paused':
                            raise PauseRequested()
                    pause(); compare(suite,target,ready)
                outcome={**status(suite),'status':'complete','workflow_status':'complete','target_additional_epochs':additional_epochs}
            except PauseRequested:
                outcome={**status(suite),'status':'paused','workflow_status':'paused','target_additional_epochs':additional_epochs}
            except Exception:
                atomic_json(suite/'suite_status.json',{'status':'failed',**status(suite)})
                raise
        atomic_json(suite/'suite_status.json',outcome)
        return outcome
