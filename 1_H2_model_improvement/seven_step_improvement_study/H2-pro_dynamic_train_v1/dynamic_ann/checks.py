"""Real inference/backward without optimizer updates; synthetic-only training tests."""
import contextlib
import gc
import math
from pathlib import Path
import time
import numpy as np
import torch
from .common import (ARMS,Config,ROOT,PRIOR,FLOAT_SOURCE,atomic_json,atomic_npz,read,sha,run_identity,local_output,preserve_check)
from .data import ResidentData,verify_cache
from .model import Refiner,optimizer_for,assert_initial_export,supervised,require_finite,exact_forward,exact_layers_on_device
from .runtime import setup_device,engine,memory
from .evaluation import evaluate,export
from .storage import process_lock,seed_all,rng_state,atomic_checkpoint,load_checkpoint
from .synthetic import resume_check,equal_tree
from .preservation import preservation_report
from wangdao.h2.numpy_model import quantized_mlp


def hardcases(model,data,raw,prob):
    y=data.roles['train_dev']['labels'].cpu().numpy();hd=np.zeros(len(y),bool)
    for seq in data.sequences['train_dev']:
        if seq['row']['metadata']['high_density']:hd[seq['lo']:seq['hi']]=True
    def take(mask):
        ids=np.flatnonzero(mask)
        return ids[np.linspace(0,len(ids)-1,min(128,len(ids)),dtype=int)] if len(ids) else ids
    groups={'false_negative':take((y==1)&(prob<.35)),'false_positive':take((y==0)&(prob>=.35)),
            'high_density_targets':take(hd&(y==1)),
            'threshold_nearest':np.argsort(np.abs(prob-.35),kind='stable')[:128],
            'largest_logits':np.argsort(np.max(np.abs(raw),axis=1),kind='stable')[-128:]}
    ids=np.unique(np.concatenate(list(groups.values())))
    x=data.roles['train_dev']['features'][torch.as_tensor(ids,device=data.device)].cpu().numpy()
    expected=quantized_mlp(x,model.export_layers());np.testing.assert_array_equal(raw[ids],expected)
    return dict(status='passed',candidates=len(ids),groups={k:len(v) for k,v in groups.items()},int32_reference_exact=True)


def check(cache,suite,config=Config(),device_name='cuda'):
    config.validate();suite=local_output(suite);device=setup_device(device_name);manifest=verify_cache(cache)
    identity=run_identity(cache,config,engine(device));started=time.perf_counter()
    with process_lock(suite):
        if (suite/'ready.json').exists():
            old=read(suite/'ready.json')
            if old['identity']!=identity or old['status']!='passed' or sha(suite/'initial.pt')!=old['initial_checkpoint_sha256']:
                raise ValueError('Existing preflight identity changed; use a new run directory')
            return old
        if any((suite/a/'last.pt').exists() for a in ARMS):raise ValueError('Do not run a fresh preflight over existing training')
        atomic_json(suite/'identity.json',identity,immutable=True)
        seed_all(config.seed);initial=Refiner().to(device);assert_initial_export(initial)
        optimizer=optimizer_for(initial,config)
        shared=dict(identity=identity,model=initial.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),
                    source_float_sha256=sha(FLOAT_SOURCE),real_data_optimizer_steps=0)
        if (suite/'initial.pt').exists():equal_tree(load_checkpoint(suite/'initial.pt'),shared)
        else:atomic_checkpoint(suite/'initial.pt',shared)
        del initial,optimizer,shared;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
        arms={};prior=read(PRIOR/'train_dev_summary.json')
        for arm,policy in ARMS.items():
            if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
            data=ResidentData(cache,arm,device,manifest);model=Refiner().to(device);assert_initial_export(model)
            before={k:v.detach().clone() for k,v in model.state_dict().items()}
            result,probs,raw=evaluate(model,data,config.evaluation_batch_size,keep_raw=True)
            # All predictions, not just aggregate metrics, must reproduce the
            # frozen-weight window experiment before any training is allowed.
            for seq in data.sequences['train_dev']:
                np.testing.assert_array_equal(probs['int8'][seq['lo']:seq['hi']],seq['initial_probability'][seq['candidate_ids']])
            for group,old_group in (('overall','metrics'),('high_density','high_density')):
                if result['int8'][group]!=prior['policies'][policy][old_group]:raise AssertionError('Epoch-zero metric differs')
            for cost in ('mac_sparse_proposal','mac_dense_proposal'):
                if result['int8']['cost'][cost]!=prior['policies'][policy]['cost'][cost]:raise AssertionError('Epoch-zero cost differs')
            hard=hardcases(model,data,raw['int8'],probs['int8'])
            order=data.order(1,config.seed);x,y=data.batch(order[:config.source_batch_size])
            model.zero_grad(set_to_none=True);fb=time.perf_counter();output,loss=supervised(model,x,y);loss.backward()
            gradients=[p.grad for p in model.parameters()]
            if any(g is None or not bool(torch.count_nonzero(g)) for g in gradients):raise AssertionError('Missing working Refiner gradient')
            require_finite(gradients,'preflight gradients')
            for key,value in model.state_dict().items():
                if not torch.equal(value,before[key]):raise AssertionError('No-update preflight changed weights')
            if device.type=='cuda':torch.cuda.synchronize(device)
            elapsed=time.perf_counter()-fb
            # Exercise the upper bound if every source event in a batch becomes
            # a candidate. Backward only: no real-data optimizer.step anywhere.
            worst_n=min(config.source_batch_size,len(data.roles['train_dev']['features']))
            model.zero_grad(set_to_none=True)
            worst_x=data.roles['train_dev']['features'][:worst_n]
            worst_y=data.roles['train_dev']['labels'][:worst_n]
            worst_raw,worst_loss=supervised(model,worst_x,worst_y);worst_loss.backward()
            require_finite([p.grad for p in model.parameters()],'maximum-batch preflight gradients')
            for key,value in model.state_dict().items():
                if not torch.equal(value,before[key]):raise AssertionError('Maximum-batch check changed weights')
            if device.type=='cuda':torch.cuda.synchronize(device)
            arms[arm]=dict(initial=result,hardcases=hard,data=data.stats,memory=memory(device),
                backward_candidates=len(x),backward_seconds=elapsed,gradient_norms=[float(g.double().norm()) for g in gradients],
                maximum_candidate_batch_backward=worst_n,
                real_data_optimizer_steps=0,full_dev_epoch0_probability_exact=True,source_batches_per_epoch=data.batches(config))
            directory=suite/arm;directory.mkdir(exist_ok=True)
            export(directory/'initial_int8.npz',model,arm,0)
            atomic_npz(directory/'initial_dev_predictions.npz',**probs)
            atomic_json(directory/'initial_metrics.json',result)
            del model,data,before,order,x,y,output,loss,gradients,raw,probs,worst_x,worst_y,worst_raw,worst_loss;gc.collect()
            if device.type=='cuda':torch.cuda.empty_cache()
        steps={v['source_batches_per_epoch'] for v in arms.values()}
        if len(steps)!=1:raise AssertionError('Paired update budgets differ')
        synlog=suite/'synthetic_resume.log'
        with synlog.open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log):synthetic=resume_check(device)
        preservation=preservation_report()
        ready=dict(status='passed',identity=identity,formal_training_started=False,real_data_optimizer_steps=0,
            real_data_backward_without_update=True,reserved_sequences_read=False,initial_checkpoint_sha256=sha(suite/'initial.pt'),
            original_float_and_initial_export_exact=True,shared_initial_adam_empty=True,shared_initial_rng=True,
            steps_per_epoch=steps.pop(),requested_initial_epochs=5,arms=arms,synthetic_resume=synthetic,
            preservation=preservation,elapsed_seconds=time.perf_counter()-started)
        atomic_json(suite/'ready.json',ready,immutable=True)
    return ready
