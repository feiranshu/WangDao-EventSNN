"""Prepare frozen Main categories and two identical checkpoints; no real updates."""
from __future__ import annotations

import time

import numpy as np
import torch

from ..h2pro_distill_v1.data import ResidentData
from ..h2pro_distill_v1.evaluation import evaluate, export
from ..h2pro_distill_v1.model import R2T8, losses as original_losses, require_finite
from ..h2pro_distill_v1.runtime import setup_device, memory
from ..h2pro_longtrain_v1.storage import save_atomic, load
from .common import (ARMS, DEFAULT_CACHE, DEFAULT_RUN, SOURCE_RUN, DEFAULT_DIAGNOSIS,
    atomic_json, read_json, lock_directory, inputs, independent_output, arm_identity,
    sha256, digest, save_arrays)
from .evaluation import infer_main
from .objective import categories, weighted_losses
from .runtime import initialized_payload


def check(cache=DEFAULT_CACHE,suite=DEFAULT_RUN,source=SOURCE_RUN,diagnosis=DEFAULT_DIAGNOSIS,device_name='cuda'):
    started=time.perf_counter(); device=setup_device(device_name)
    identity,origin,config=inputs(cache,source,diagnosis,device)
    suite=independent_output(suite,cache,source,diagnosis)
    with lock_directory(suite,identity):
        if (suite/'ready.json').exists():
            ready=read_json(suite/'ready.json')
            if ready['identity']!=identity or ready['status']!='passed':
                raise ValueError('Different preparation identity')
            for name,d in ready['immutable_files_sha256'].items():
                if sha256(suite/name)!=d:
                    raise ValueError('Prepared initialization/mask was changed')
            return ready
        data=ResidentData(cache,device)
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model=R2T8().to(device); model.load_state_dict(origin['model'],strict=True)
        model_id=digest(model.state_dict())
        if model_id!=identity['source_model_digest']:
            raise AssertionError('FP32 source checkpoint changed on device transfer')
        main_raw,main_spikes=infer_main(model,data,config.batch_size)
        with np.load(diagnosis/'train_main_epoch_088_predictions.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(main_raw,z['raw'])
            np.testing.assert_array_equal(main_spikes,z['spikes'])
        teacher=data.roles['train_main']['teacher'].cpu().numpy()
        with np.load(diagnosis/'train_main_ann_predictions.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(teacher,z['raw'])
        y=data.roles['train_main']['labels'].cpu().numpy()
        category=categories(teacher,main_raw,y,role='train_main')
        mask_path=suite/'main_hard_categories.npz'
        save_arrays(mask_path,category=category)
        counts=np.bincount(category,minlength=3).tolist()
        rows=[]
        for seq in data.sequences['train_main']:
            c=category[seq['lo']:seq['hi']]
            rows.append({'sequence':seq['row']['sequence'],'high_density':seq['row']['high_density'],
                         'category_counts':np.bincount(c,minlength=3).tolist()})
        mining={'role':'train_main','source_model':'epoch88','teacher_threshold':.35,'student_threshold':.35,
                'categories':{'0':'ordinary','1':'hard_positive','2':'hard_negative'},'counts':counts,
                'total_candidates':len(category),'hard_fraction':float((category!=0).mean()),
                'sequences':rows,'used_dev_labels':False,'refreshed_during_training':False,
                'hard_weight':2.,'normal_weight':1.,'sha256':sha256(mask_path)}
        atomic_json(suite/'hard_mining.json',mining)
        baseline,raw,spikes=evaluate(model,data,config.batch_size)
        training=read_json(source/'training.json')
        old=training['epochs'][87]['evaluation']
        for key in ('overall','high_density','ordinary','cost'):
            if baseline[key]!=old[key]:
                raise AssertionError('Prepared epoch88 no longer matches historical Dev')
        # Real data: two forward/backward checks, no optimizer.step. Both reset
        # gradients; master weights and scales are compared bit for bit afterward.
        order=data.order(89,config.seed)
        ids=order[:config.batch_size]
        batch=data.batch('train_main',ids)
        cat=torch.from_numpy(category.copy()).to(device)[ids]
        backward={}
        for arm in ARMS:
            model.zero_grad(set_to_none=True)
            output,_=model(batch['features'])
            terms=(original_losses(output,batch['labels'],batch['teacher'],config) if arm=='control' else
                   weighted_losses(output,batch['labels'],batch['teacher'],1.+(cat!=0).float(),config))
            require_finite(terms,'preflight losses'); terms[0].backward()
            grads=[p.grad for p in model.parameters()]
            if any(g is None or not bool(torch.count_nonzero(g)) for g in grads):
                raise AssertionError('Missing preflight gradient')
            require_finite([g for g in grads if g is not None],'preflight gradients')
            backward[arm]={'losses':[float(t.detach()) for t in terms],
                           'gradient_norms':[float(g.double().norm()) for g in grads if g is not None]}
        model.zero_grad(set_to_none=True)
        if digest(model.state_dict())!=model_id:
            raise AssertionError('No-update preflight changed source weights')
        mask_hash=sha256(mask_path)
        initial_hashes={}
        for arm in ARMS:
            aid=arm_identity(identity,arm,mask_hash)
            directory=suite/arm
            with lock_directory(directory,aid):
                payload=initialized_payload(origin,aid,baseline)
                initial=directory/'initial.pt'
                if initial.exists():
                    if digest(load(initial))!=digest(payload):
                        raise ValueError('Existing initial checkpoint differs')
                else:
                    save_atomic(initial,payload)
                if (directory/'last.pt').exists() and digest(load(directory/'last.pt'))!=digest(payload):
                    raise ValueError('Cannot re-prepare an arm after updates')
                for name in ('last.pt','best.pt'):
                    if not (directory/name).exists():
                        save_atomic(directory/name,payload)
                export(directory/'initial_int8.npz',model)
                export(directory/'best_int8.npz',model)
                atomic_json(directory/'progress.json',{'status':'ready_not_trained','arm':arm,
                    'completed_additional_epochs':0,'source_epoch':88,'effective_completed_epoch':88,
                    'global_step':origin['state']['global_step'],'additional_updates':0,'formal_training_started':False,
                    'resumable_checkpoint':str(directory/'last.pt')})
                for field in ('model','optimizer','rng'):
                    if digest(load(initial)[field])!=digest(origin[field]):
                        raise AssertionError('Initialization differs from source '+field)
                initial_hashes[arm]={'model':digest(payload['model']),'optimizer':digest(payload['optimizer']),
                                     'rng':digest(payload['rng'])}
        if initial_hashes['control']!=initial_hashes['hard_weighted']:
            raise AssertionError('Paired initial model/Adam/RNG differ')
        immutable=['main_hard_categories.npz','hard_mining.json']+[f'{a}/{n}' for a in ARMS
                    for n in ('initial.pt','initial_int8.npz')]
        ready={'status':'passed','identity':identity,'formal_training_started':False,'real_data_optimizer_steps':0,
               'real_data_backward_checks':backward,'initial_student':baseline,'frozen_ann':training['frozen_ann'],
               'hard_categories_sha256':mask_hash,'hard_mining':mining,'identical_initial_states':initial_hashes,
               'initial_adam_step':origin['state']['global_step'],'full_main_predictions_exact':len(main_raw),
               'full_dev_metrics_and_cost_exact':True,'explicit_dev_candidates_verified':baseline['explicit_verified_candidates'],
               'immutable_files_sha256':{n:sha256(suite/n) for n in immutable},'data':data.stats,
               'memory':memory(device),'seconds':time.perf_counter()-started,'reserved_splits_read':False}
        atomic_json(suite/'ready.json',ready)
        atomic_json(suite/'suite_status.json',{'status':'ready_not_trained','additional_epochs_per_arm':0})
        return ready
