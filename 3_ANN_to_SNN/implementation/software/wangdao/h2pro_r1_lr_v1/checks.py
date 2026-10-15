"""Check exact source replay and recovery; real data never takes optimizer steps."""
from __future__ import annotations

from copy import deepcopy
import time

import numpy as np
import torch

from ..h2pro_r1_train_v1.checks import hardcase_check
from ..h2pro_r1_train_v1.data import ResidentData
from ..h2pro_r1_train_v1.model import R1T16, losses, require_finite, update
from ..h2pro_r1_train_v1.evaluation import evaluate, export, load_export
from ..h2pro_r1_train_v1.runtime import setup_device, memory, checkpoint, restore
from ..h2pro_r1_train_v1.common import STAGE, STEPS, ENCODERS
from ..h2pro_calibration_v1.core import forward
from ..h2pro_longtrain_v1.storage import save_atomic, load, all_rng, set_rng
from .common import (
    ARMS, DEFAULT_CACHE, DEFAULT_RUN, SOURCE_RUN, SOURCE_EPOCH,
    atomic_json, lock_directory, inputs, independent_output, arm_identity, arm_config,
    initialized_payload, verify_prepared, sha256, digest,
)


def initialize_arms(suite, identity, origin, baseline, model):
    hashes = {}
    for arm in ARMS:
        aid = arm_identity(identity, arm); directory = suite / arm
        payload = initialized_payload(origin, aid, baseline)
        with lock_directory(directory, aid):
            for name in ('initial.pt', 'last.pt', 'best.pt'):
                path = directory / name
                if path.exists():
                    if digest(load(path)) != digest(payload):
                        raise ValueError('Cannot reset an existing changed checkpoint: ' + name)
                else:
                    save_atomic(path, payload)
            for name in ('initial_int8.npz', 'best_int8.npz'):
                path = directory / name
                if path.exists():
                    for (w, s), (expected_w, expected_s) in zip(load_export(path), model.export_layers()):
                        np.testing.assert_array_equal(w, expected_w); np.testing.assert_array_equal(s, expected_s)
                else:
                    export(path, model)
            restored = load(directory / 'initial.pt')
            # Adam tensor states and every other hyperparameter must match;
            # only the explicitly assigned learning rate is allowed to differ.
            opt = deepcopy(restored['optimizer'])
            for group, base_group in zip(opt['param_groups'], origin['optimizer']['param_groups']):
                assert group['lr'] == aid['config']['learning_rate']
                group['lr'] = base_group['lr']
            assert digest(opt) == digest(origin['optimizer'])
            for field in ('model', 'rng'):
                assert digest(restored[field]) == digest(origin[field])
            hashes[arm] = {'model': digest(restored['model']),
                           'adam_moments_steps': digest(restored['optimizer']['state']),
                           'optimizer_except_lr': digest(opt), 'rng': digest(restored['rng'])}
            atomic_json(directory / 'progress.json', {
                'status': 'ready_not_trained', 'arm': arm, 'learning_rate': aid['config']['learning_rate'],
                'completed_additional_epochs': 0, 'source_epoch': SOURCE_EPOCH,
                'effective_completed_epoch': SOURCE_EPOCH, 'global_step': origin['state']['global_step'],
                'additional_updates': 0, 'formal_training_started': False,
                'resumable_checkpoint': str(directory / 'last.pt')})
    assert hashes['control'] == hashes['low_lr']
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
            update(full, full_opt, batch, cfg)
        expected = {k: digest(v) for k, v in
                    {'model': full.state_dict(), 'optimizer': full_opt.state_dict(), 'rng': all_rng()}.items()}
        partial, opt = fresh()
        update(partial, opt, batch, cfg)
        path = directory / f'{arm}_random_data.pt'
        checkpoint(path, partial, opt, {'synthetic_only': True}, aid)
        resumed, opt2 = fresh()
        restore(path, resumed, opt2, aid)
        update(resumed, opt2, batch, cfg)
        actual = {k: digest(v) for k, v in
                  {'model': resumed.state_dict(), 'optimizer': opt2.state_dict(), 'rng': all_rng()}.items()}
        assert actual == expected, {'arm': arm, 'actual': actual, 'expected': expected}
        assert all(float(v['step']) == origin['state']['global_step'] + 2 for v in opt2.state.values())
        results[arm] = {'model_adam_rng_bit_exact': True, 'learning_rate': cfg.learning_rate,
                        'synthetic_optimizer_steps': 4, 'real_data_optimizer_steps': 0}
    return results


def check(cache=DEFAULT_CACHE, suite=DEFAULT_RUN, source=SOURCE_RUN, device_name='cuda'):
    started = time.perf_counter(); device = setup_device(device_name)
    identity, origin, config, old_ready = inputs(cache, source, device)
    suite = independent_output(suite, cache, source)
    with lock_directory(suite, identity):
        if (suite / 'ready.json').exists():
            return verify_prepared(suite, identity)
        data = ResidentData(cache, device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model = R1T16().to(device); model.load_state_dict(origin['model'], strict=True)
        model.verify_frozen_scales()
        model_digest = digest(model.state_dict())
        assert model_digest == identity['source_model_digest']
        baseline, raw, spikes = evaluate(model, data, config.batch_size)
        source_result = origin['state']['history'][-1]['evaluation']
        for key in ('overall', 'ordinary', 'high_density', 'cost'):
            assert baseline[key] == source_result[key], 'Source Dev metrics or cost changed: ' + key
        layers = model.export_layers(); reference_ac = 0
        for lo in range(0, len(raw), config.batch_size):
            x = data.roles['train_dev']['features'][lo:lo + config.batch_size].cpu().numpy()
            expected, cost = forward(x, layers, STAGE, STEPS, ENCODERS, 'count_reference')
            np.testing.assert_array_equal(raw[lo:lo + len(x)], expected)
            reference_ac += cost.rate_network_ac
        assert reference_ac == baseline['cost']['inference_equivalent_ac']
        hardcases = hardcase_check(model, data, raw, spikes)
        order = data.order(SOURCE_EPOCH + 1, config.seed)
        batch = data.batch('train_main', order[:config.batch_size])
        backward = {}
        for arm in ARMS:
            model.zero_grad(set_to_none=True)
            output, _ = model(batch['features'])
            terms = losses(output, batch['labels'], batch['teacher'], arm_config(config, arm))
            require_finite(terms, 'preflight losses'); terms[0].backward()
            gradients = [p.grad for p in model.parameters()]
            assert all(g is not None and bool(torch.count_nonzero(g)) for g in gradients)
            require_finite(gradients, 'preflight gradients')
            backward[arm] = {'losses': [float(t.detach()) for t in terms],
                             'gradient_norms': [float(g.double().norm()) for g in gradients],
                             'gradient_digest': digest(gradients)}
        model.zero_grad(set_to_none=True)
        assert backward['control'] == backward['low_lr']
        assert digest(model.state_dict()) == model_digest
        smoke = synthetic_resume(device, suite / 'synthetic_checks', origin, identity, source_result, config)
        initial_hashes = initialize_arms(suite, identity, origin, source_result, model)
        immutable = [f'{arm}/{name}' for arm in ARMS for name in ('identity.json', 'initial.pt', 'initial_int8.npz')]
        ready = {
            'status': 'passed', 'identity': identity, 'formal_training_started': False,
            'real_data_optimizer_steps': 0, 'reserved_splits_read': False,
            'initial_student': source_result, 'frozen_ann': old_ready['frozen_ann'],
            'original_r1_before_training': old_ready['initial_student'],
            'identical_initial_states_except_lr': initial_hashes, 'initial_adam_step': origin['state']['global_step'],
            'full_dev_candidates_exact': len(raw), 'full_dev_metrics_and_cost_exact': True,
            'explicit_dev_candidates_verified': baseline['explicit_verified_candidates'],
            'hardcases': hardcases, 'real_data_backward_checks_no_update': backward,
            'synthetic_resume': smoke, 'data': data.stats, 'memory': memory(device),
            'immutable_files_sha256': {name: sha256(suite / name) for name in immutable},
            'check_seconds': time.perf_counter() - started,
        }
        atomic_json(suite / 'ready.json', ready)
        atomic_json(suite / 'suite_status.json', {'status': 'ready_not_trained',
                    'target_additional_epochs': 50, 'additional_epochs_per_arm': 0})
        return ready
