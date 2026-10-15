"""Check exact source replay and recovery; real data never takes optimizer steps."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import time

import numpy as np
import torch

from ..h2pro_r1_train_v1.checks import hardcase_check
from ..h2pro_r1_train_v1.data import ResidentData
from ..h2pro_r1_train_v1.model import losses, require_finite
from .model import R1Multi as R1T16, joint_losses, update
from .evaluation import evaluate, export, load_export, atomic_npz
from ..h2pro_r1_train_v1.runtime import setup_device, memory, checkpoint, restore
from ..h2pro_r1_train_v1.common import STAGE, STEPS, ENCODERS
from ..h2pro_calibration_v1.core import forward
from ..h2pro_longtrain_v1.storage import save_atomic, load, all_rng, set_rng
from .common import (
    ARMS, DEFAULT_CACHE, DEFAULT_RUN, SOURCE_RUN, SOURCE_EPOCH,
    atomic_json, lock_directory, inputs, independent_output, arm_identity, arm_config,
    initialized_payload, verify_prepared, sha256, digest,
    read_json, PREFIX_REFERENCE, GATE_REFERENCE, STEPS as PREFIX_STEPS, EXIT_POLICY,
)
from ..h2pro_calibration_v1.core import probabilities
from ..h2pro_adaptive_t_v2.core import incremental, choose, ledger


def initialize_arms(suite, identity, origin, baseline, model):
    hashes = {}
    for arm in ARMS:
        aid = arm_identity(identity, arm); directory = suite / arm
        payload = initialized_payload(origin, aid, baseline)
        with lock_directory(directory, aid):
            for name in ('initial.pt', 'last.pt', 'best.pt', 'best_adaptive.pt'):
                path = directory / name
                if path.exists():
                    if digest(load(path)) != digest(payload):
                        raise ValueError('Cannot reset an existing changed checkpoint: ' + name)
                else:
                    save_atomic(path, payload)
            for name in ('initial_int8.npz', 'best_int8.npz', 'best_adaptive_int8.npz'):
                path = directory / name
                if path.exists():
                    for (w, s), (expected_w, expected_s) in zip(load_export(path), model.export_layers()):
                        np.testing.assert_array_equal(w, expected_w); np.testing.assert_array_equal(s, expected_s)
                else:
                    export(path, model)
            restored = load(directory / 'initial.pt')
            # Adam tensor states AND every hyperparameter must match the source.
            opt = deepcopy(restored['optimizer'])
            for group, base_group in zip(opt['param_groups'], origin['optimizer']['param_groups']):
                assert group['lr'] == aid['config']['learning_rate']
                assert group == base_group
            assert digest(opt) == digest(origin['optimizer'])
            for field in ('model', 'rng'):
                assert digest(restored[field]) == digest(origin[field])
            hashes[arm] = {'model': digest(restored['model']),
                           'adam_moments_steps': digest(restored['optimizer']['state']),
                           'optimizer': digest(opt), 'rng': digest(restored['rng'])}
            atomic_json(directory / 'progress.json', {
                'status': 'ready_not_trained', 'arm': arm, 'learning_rate': aid['config']['learning_rate'],
                'completed_additional_epochs': 0, 'source_epoch': SOURCE_EPOCH,
                'effective_completed_epoch': SOURCE_EPOCH, 'global_step': origin['state']['global_step'],
                'additional_updates': 0, 'formal_training_started': False,
                'resumable_checkpoint': str(directory / 'last.pt')})
    assert hashes['control'] == hashes['multit']
    return hashes


def synthetic_resume(device, directory, origin, identity, baseline, config):
    """Two random-data steps per trajectory, with inherited nonzero Adam state."""
    directory.mkdir(exist_ok=True)
    generator = torch.Generator().manual_seed(159)
    batch = {'features': torch.randint(0, 256, (96, 36), dtype=torch.uint8, generator=generator).to(device),
             'labels': torch.randint(0, 2, (96,), dtype=torch.long, generator=generator).to(device),
             'teacher': torch.randn(96, 4, generator=generator).to(device)}
    results = {}
    for arm in ARMS:
        aid = arm_identity(identity, arm); cfg = arm_config(config, arm)
        payload = initialized_payload(origin, aid, baseline)
        def fresh():
            model = R1T16().to(device); model.load_state_dict(payload['model'])
            opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=0., foreach=False)
            # load_state_dict may retain already-on-device tensor storage,
            # especially CPU Adam step counters. Each synthetic trajectory
            # must receive independent moments and counters.
            opt.load_state_dict(deepcopy(payload['optimizer'])); set_rng(payload['rng'])
            return model, opt
        full, full_opt = fresh()
        for _ in range(2):
            update(full, full_opt, batch, cfg, arm)
        expected = {k: digest(v) for k, v in
                    {'model': full.state_dict(), 'optimizer': full_opt.state_dict(), 'rng': all_rng()}.items()}
        partial, opt = fresh()
        update(partial, opt, batch, cfg, arm)
        path = directory / f'{arm}_random_data.pt'
        checkpoint(path, partial, opt, {'synthetic_only': True}, aid)
        resumed, opt2 = fresh()
        restore(path, resumed, opt2, aid)
        update(resumed, opt2, batch, cfg, arm)
        actual = {k: digest(v) for k, v in
                  {'model': resumed.state_dict(), 'optimizer': opt2.state_dict(), 'rng': all_rng()}.items()}
        assert actual == expected, {'arm': arm, 'actual': actual, 'expected': expected}
        assert all(float(v['step']) == origin['state']['global_step'] + 2 for v in opt2.state.values())
        results[arm] = {'model_adam_rng_bit_exact': True, 'learning_rate': cfg.learning_rate,
                        'synthetic_optimizer_steps': 4, 'real_data_optimizer_steps': 0}
    return results


def prefix_hardcases(model,data,raw,spikes):
    labels=data.roles['train_dev']['labels'].cpu().numpy()
    hd=np.zeros(len(labels),bool)
    for seq in data.sequences['train_dev']:
        if seq['row']['high_density']:hd[seq['lo']:seq['hi']]=True
    reports={};collected=[]
    for i,t in enumerate(PREFIX_STEPS):
        p=probabilities(raw[i])
        def spaced(ids):
            return ids[np.linspace(0,len(ids)-1,min(128,len(ids)),dtype=int)] if len(ids) else ids
        groups={'false_negatives':spaced(np.flatnonzero((labels==1)&(p<.35))),
                'false_positives':spaced(np.flatnonzero((labels==0)&(p>=.35))),
                'hd_targets':spaced(np.flatnonzero(hd&(labels==1))),
                'decision_boundary':np.argsort(abs(p-.35),kind='stable')[:128],
                'largest_logits':np.argsort(abs(raw[i]).max(axis=1),kind='stable')[-128:]}
        ids=np.unique(np.concatenate(list(groups.values())));collected.append(ids)
        x=data.roles['train_dev']['features'][torch.as_tensor(ids,device=data.device)].cpu().numpy()
        expected,cost=forward(x,model.export_layers(),'r1',t,ENCODERS,'spike_add')
        np.testing.assert_array_equal(expected,raw[i,ids])
        assert cost.executed_synaptic_ac==int(spikes[i,ids].sum())*32
        reports[str(t)]=dict(unique_candidates=len(ids),groups={k:len(v) for k,v in groups.items()})
    ids=np.unique(np.concatenate(collected))
    x=data.roles['train_dev']['features'][torch.as_tensor(ids,device=data.device)].cpu().numpy()
    y,n,ix,measured=incremental(x,model.export_layers(),EXIT_POLICY)
    expected_ix=choose(np.stack([probabilities(v[ids]) for v in raw]),EXIT_POLICY)
    np.testing.assert_array_equal(ix,expected_ix);np.testing.assert_array_equal(y,raw[ix,ids])
    np.testing.assert_array_equal(n,spikes[ix,ids])
    cost=ledger(ix,n,EXIT_POLICY)
    assert all(cost[k]==v for k,v in measured.items())
    return dict(prefixes=reports,adaptive_unique_candidates=len(ids),selection_changes=0,optimizer_steps=0)


def real_backward_no_update(data,origin,config,arm):
    probe=R1T16().to(data.device);probe.load_state_dict(origin['model'])
    opt=torch.optim.Adam(probe.parameters(),lr=config.learning_rate,weight_decay=0.,foreach=False)
    opt.load_state_dict(deepcopy(origin['optimizer']))
    model_before=digest(probe.state_dict());adam_before=digest(opt.state_dict())
    order=data.order(160,config.seed);batch=data.batch('train_main',order[:config.batch_size])
    if data.device.type=='cuda':torch.cuda.reset_peak_memory_stats(data.device)
    if arm=='control':
        raw,_=probe(batch['features']);terms=torch.stack(losses(raw,batch['labels'],batch['teacher'],config))
        prefix={'16':terms.detach().cpu().tolist()}
    else:
        raw,_=probe.prefixes(batch['features']);terms,per=joint_losses(raw,batch['labels'],batch['teacher'],config)
        prefix={str(t):v for t,v in zip(PREFIX_STEPS,per.detach().cpu().tolist())}
    require_finite([raw,terms],'preflight');terms[0].backward()
    gradients=[p.grad for p in probe.parameters()]
    assert all(g is not None and bool(torch.count_nonzero(g)) for g in gradients)
    require_finite(gradients,'preflight gradients')
    if data.device.type=='cuda':torch.cuda.synchronize(data.device)
    report=dict(losses=terms.detach().cpu().tolist(),prefix_losses=prefix,
                gradient_norms=[float(g.double().norm()) for g in gradients],gradient_digest=digest(gradients),
                batch_digest=digest(batch),optimizer_steps=0,memory=memory(data.device))
    assert digest(probe.state_dict())==model_before and digest(opt.state_dict())==adam_before
    return report


def check(cache=DEFAULT_CACHE,suite=DEFAULT_RUN,source=SOURCE_RUN,device_name='cuda'):
    started=time.perf_counter();device=setup_device(device_name)
    identity,origin,config,old_ready=inputs(cache,source,device)
    suite=independent_output(suite,cache,source)
    with lock_directory(suite,identity):
        if (suite/'ready.json').exists():return verify_prepared(suite,identity)
        origin_digest=digest(origin)
        data=ResidentData(cache,device)
        model=R1T16().to(device);model.load_state_dict(origin['model']);model.verify_frozen_scales()
        assert digest(model.state_dict())==identity['source_model_digest']
        baseline,raw,spikes=evaluate(model,data,config.batch_size)
        original=origin['state']['history'][-1]['evaluation']
        for key in ('overall','ordinary','high_density','cost'):
            assert baseline[key]==original[key],key
        reference=read_json(PREFIX_REFERENCE/'comparison.json')
        reference_adaptive=read_json(GATE_REFERENCE/'comparison.json')['adaptive']['accuracy']
        for key,current,old in [(str(t),baseline['fixed'][str(t)],reference['fixed'][str(t)]) for t in PREFIX_STEPS]+[
                ('adaptive',baseline['adaptive'],reference_adaptive)]:
            for group in ('overall','ordinary','high_density','sequences'):
                assert current[group]==old[group],(key,group)
            for costkey in ('inference_equivalent_mac_sparse_proposal','inference_equivalent_mac_dense_proposal',
                            'inference_equivalent_ac','average_steps','exit_counts'):
                assert current['cost'][costkey]==old['cost'][costkey],(key,costkey)
        for seq in data.sequences['train_dev']:
            path=PREFIX_REFERENCE/'traces/train_dev'/(Path(seq['row']['sequence']).stem+'.npz')
            receipt=read_json(path.with_suffix('.json'))
            assert sha256(path)==receipt['trace_sha256']
            assert seq['row']['cache_sha256']==receipt['cache_sha256']
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(raw[:,seq['lo']:seq['hi']],z['raw'])
                np.testing.assert_array_equal(spikes[:,seq['lo']:seq['hi']],z['spikes'])
        hard=prefix_hardcases(model,data,raw,spikes)
        backward={a:real_backward_no_update(data,origin,config,a) for a in ARMS}
        assert backward['control']['batch_digest']==backward['multit']['batch_digest']
        assert backward['control']['gradient_digest']!=backward['multit']['gradient_digest']
        assert digest(model.state_dict())==identity['source_model_digest']
        smoke=synthetic_resume(device,suite/'synthetic_checks',origin,identity,baseline,config)
        assert digest(origin)==origin_digest,'Synthetic checks mutated source tensors'
        hashes=initialize_arms(suite,identity,origin,baseline,model)
        atomic_npz(suite/'initial_dev_predictions.npz',raw=raw,spikes=spikes)
        immutable=[f'{a}/{n}' for a in ARMS for n in ('identity.json','initial.pt','initial_int8.npz')]+['initial_dev_predictions.npz']
        ready=dict(status='passed',identity=identity,formal_training_started=False,real_data_optimizer_steps=0,
            reserved_splits_read=False,initial_student=baseline,frozen_ann=old_ready['frozen_ann'],
            original_r1_before_training=old_ready['initial_student'],initial_adam_step=origin['state']['global_step'],
            identical_model_adam_rng=hashes,full_dev_candidates_exact_per_prefix=raw.shape[1],
            all_three_prefixes_and_fixed_policy_exact=True,hardcases=hard,
            real_data_backward_checks_no_update=backward,memory_by_arm={a:backward[a]['memory'] for a in ARMS},
            synthetic_resume=smoke,data=data.stats,immutable_files_sha256={n:sha256(suite/n) for n in immutable},
            check_seconds=time.perf_counter()-started)
        atomic_json(suite/'ready.json',ready)
        atomic_json(suite/'suite_status.json',dict(status='ready_not_trained',target_additional_epochs=5,additional_epochs_per_arm=0))
        return ready

