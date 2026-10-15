"""User-started paired training with identical source-event update budgets."""
import gc
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .common import (ARMS,Config,atomic_json,atomic_npz,digest,read,run_identity,local_output,sha)
from .storage import (process_lock,PauseControl,PauseRequested,rng_state,set_rng,atomic_checkpoint,load_checkpoint)
from .model import Refiner,optimizer_for,update
from .data import ResidentData,verify_cache
from .evaluation import evaluate,export,differences


def setup_device(name):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if name not in ('cuda','cpu'):raise ValueError('Choose cuda or cpu explicitly')
    if name=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; select gazelle-train Python')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    return torch.device('cuda:0' if name=='cuda' else 'cpu')


def engine(device):
    return dict(torch=str(torch.__version__),numpy=np.__version__,cuda=str(torch.version.cuda),device=str(device),
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        training='original FP32 float ANN; AdamW FP32',int8_evaluation='exact FP64 integer dot and scale decode',deterministic=True)


def memory(device):
    return dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0)


def totals():return dict(candidate_events=0,source_events=0,loss_sum=0.,gradient_norm_max=0.,max_abs_logit=0.,train_seconds=0.,empty_candidate_batches=0)


def initial_state(initial_result):
    m=initial_result['int8']['overall']
    return dict(cursor=dict(epoch=1,batch=0,phase='train'),global_step=0,history=[],totals=totals(),best_epoch=0,
                best_score=[m['iou'],m['recall']])


def checkpoint(path,model,optimizer,state,identity,arm):
    atomic_checkpoint(path,dict(identity=identity,arm=arm,model=model.state_dict(),optimizer=optimizer.state_dict(),
        state=state,rng=rng_state(),resumable=True))


def restore(path,model,optimizer,identity,arm):
    saved=load_checkpoint(path)
    if saved.get('identity')!=identity or saved.get('arm')!=arm or saved.get('resumable') is not True:
        raise ValueError('Checkpoint identity/arm mismatch; previous experiments cannot be reused here')
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);set_rng(saved['rng'])
    return saved['state']


def validate_state(state,data,config):
    c=state['cursor'];batches=data.batches(config)
    if (c['epoch']!=len(state['history'])+1 or c['phase'] not in ('train','evaluate')
            or not 0<=c['batch']<=batches or (c['phase']=='evaluate' and c['batch']!=batches)
            or state['global_step']!=(c['epoch']-1)*batches+c['batch']
            or state['totals']['source_events']!=min(c['batch']*config.source_batch_size,data.source_count)):
        raise ValueError('Checkpoint cursor or source-event coverage inconsistent')


def emit(directory,status,state,data,device,target_epochs):
    row=dict(status=status,completed_epochs=len(state['history']),target_total_epochs=target_epochs,
        cursor=state['cursor'],global_step=state['global_step'],best_epoch=state['best_epoch'],
        totals=state['totals'],data=data.stats,memory=memory(device),resumable_checkpoint=str(directory/'last.pt'),updated_unix_s=time.time())
    atomic_json(directory/'progress.json',row)
    with (directory/'progress.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in row.items() if k!='data'},allow_nan=False),flush=True)


def run_epochs(directory,arm,model,optimizer,state,identity,data,config,initial_result,control,
               epochs=5,checkpoint_batches=10,_test_pause_after=None,_test_after_artifacts=None):
    """Production loop also exercised with synthetic data in regression tests."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);validate_state(state,data,config)
    def commit():checkpoint(directory/'last.pt',model,optimizer,state,identity,arm)
    def history():atomic_json(directory/'training.json',dict(identity=identity,arm=arm,initial=initial_result,
                       epochs=state['history'],best_epoch=state['best_epoch'],global_step=state['global_step']))
    try:
        while state['cursor']['epoch']<=epochs:
            control.check();c=state['cursor'];epoch=c['epoch'];order=data.order(epoch,config.seed)
            emit(directory,'running',state,data,data.device,epochs)
            if c['phase']=='train':
                for batch in range(c['batch'],data.batches(config)):
                    control.check();start=batch*config.source_batch_size
                    source=order[start:start+config.source_batch_size];x,y=data.batch(source)
                    started=time.perf_counter();result=update(model,optimizer,x,y)
                    t=state['totals'];n=result['candidate_events']
                    t['candidate_events']+=n;t['source_events']+=len(source);t['loss_sum']+=result['loss']*n
                    t['gradient_norm_max']=max(t['gradient_norm_max'],result['gradient_norm'])
                    t['max_abs_logit']=max(t['max_abs_logit'],result['max_abs_logit'])
                    t['empty_candidate_batches']+=int(n==0);t['train_seconds']+=time.perf_counter()-started
                    c['batch']=batch+1;state['global_step']+=1
                    if c['batch']%checkpoint_batches==0:
                        commit();emit(directory,'running',state,data,data.device,epochs)
                    if _test_pause_after==state['global_step']:raise PauseRequested()
                if (state['totals']['candidate_events']!=data.candidate_count
                        or state['totals']['source_events']!=data.source_count):raise AssertionError('Incomplete epoch coverage')
                c['phase']='evaluate';commit()
            control.check()
            evaluation,predictions,_=evaluate(model,data,config.evaluation_batch_size,control.check)
            entry=dict(epoch=epoch,global_step=state['global_step'],training=dict(state['totals']),
                       evaluation=evaluation,delta_to_initial_pp=differences(evaluation,initial_result),memory=memory(data.device))
            entry['training']['mean_loss']=entry['training']['loss_sum']/max(1,data.candidate_count)
            m=evaluation['int8']['overall'];rank=[m['iou'],m['recall']]
            state['history'].append(entry);state['cursor']=dict(epoch=epoch+1,batch=0,phase='train');state['totals']=totals()
            ed=directory/'epochs';ed.mkdir(exist_ok=True);stem=f'epoch_{epoch:03}'
            atomic_json(ed/(stem+'.json'),entry);atomic_npz(ed/(stem+'_dev.npz'),**predictions)
            export(ed/(stem+'_int8.npz'),model,arm,epoch)
            if rank>state['best_score']:
                state['best_epoch']=epoch;state['best_score']=rank
                checkpoint(directory/'best.pt',model,optimizer,state,identity,arm)
                export(directory/'best_int8.npz',model,arm,epoch)
            checkpoint(ed/(stem+'.pt'),model,optimizer,state,identity,arm)
            if _test_after_artifacts:_test_after_artifacts(epoch)
            commit();history();emit(directory,'epoch_complete',state,data,data.device,epochs)
        history();emit(directory,'complete',state,data,data.device,epochs)
    except PauseRequested:
        commit();history();emit(directory,'paused',state,data,data.device,epochs)
        return 'paused'
    except Exception as error:
        stamp=time.time_ns()
        atomic_json(directory/f'failure_{stamp}.json',dict(error=repr(error),traceback=traceback.format_exc(),
            cursor=state['cursor'],global_step=state['global_step'],resumable_checkpoint=str(directory/'last.pt'),
            checkpoint_sha256=sha(directory/'last.pt') if (directory/'last.pt').exists() else None))
        emit(directory,'failed',state,data,data.device,epochs)
        raise
    return 'complete'


def suite_status(suite):
    suite=local_output(suite);result={}
    for arm in ARMS:
        p=suite/arm/'progress.json'
        result[arm]=read(p) if p.exists() else dict(status='ready_not_trained' if (suite/'ready.json').exists() else 'not_prepared')
    return dict(suite=str(suite),arms=result)


def train(cache,suite,config=Config(),epochs=5,checkpoint_batches=10,device_name='cuda'):
    if type(epochs) is not int or epochs<1 or type(checkpoint_batches) is not int or checkpoint_batches<1:
        raise ValueError('Positive total epochs/checkpoint interval required')
    config.validate();suite=local_output(suite);device=setup_device(device_name)
    manifest=verify_cache(cache);identity=run_identity(cache,config,engine(device))
    with process_lock(suite):
        ready=read(suite/'ready.json')
        if ready['identity']!=identity or ready['status']!='passed':raise ValueError('Run matching preflight before training')
        if sha(suite/'initial.pt')!=ready['initial_checkpoint_sha256']:raise ValueError('Shared initial state changed')
        # An explicit train invocation resumes the entire suite, clearing only
        # its own prior pause request. No old model/run artifacts are touched.
        (suite/'PAUSE').unlink(missing_ok=True)
        with PauseControl(suite) as control:
            try:
                for arm in ARMS:
                    control.check();directory=suite/arm;directory.mkdir(exist_ok=True)
                    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
                    data=ResidentData(cache,arm,device,manifest)
                    model=Refiner().to(device);optimizer=optimizer_for(model,config)
                    if (directory/'last.pt').exists():state=restore(directory/'last.pt',model,optimizer,identity,arm)
                    else:
                        saved=load_checkpoint(suite/'initial.pt')
                        if saved['identity']!=identity:raise ValueError('Initial state identity changed')
                        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer']);set_rng(saved['rng'])
                        state=initial_state(ready['arms'][arm]['initial'])
                        checkpoint(directory/'best.pt',model,optimizer,state,identity,arm)
                        export(directory/'best_int8.npz',model,arm,0)
                        checkpoint(directory/'last.pt',model,optimizer,state,identity,arm)
                    status=run_epochs(directory,arm,model,optimizer,state,identity,data,config,
                        ready['arms'][arm]['initial'],control,epochs,checkpoint_batches)
                    del model,optimizer,data;gc.collect()
                    if device.type=='cuda':torch.cuda.empty_cache()
                    if status=='paused':break
            except PauseRequested:
                atomic_json(suite/'suite_progress.json',dict(status='paused_between_arms',target_epochs=epochs))
    result=suite_status(suite);atomic_json(suite/'suite_progress.json',result)
    return result
