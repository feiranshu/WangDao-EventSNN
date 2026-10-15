"""Preflight: real-data inference/backward only; synthetic-only update tests."""
from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import time
from typing import cast

import numpy as np
import torch

from ..h2pro_calibration_v1.core import forward, load_layers, probabilities
from ..h2pro_longtrain_v1.storage import load
from .common import (Config, atomic_json, lock_directory, run_identity, read_json,
                     ANCHOR_DEV, STAGE, STEPS, ENCODERS, THRESHOLD)
from .data import ResidentData
from .evaluation import evaluate, score, export, load_export
from .model import CountSTE, R1T16, losses, require_finite, update
from .runtime import checkpoint, engine, initial_state, memory, restore, setup_device


def synthetic_resume(device, output, config=Config()):
    """Check Adam, RNG, master weights and export across a committed interruption."""
    # Deliberate synthetic perturbation crosses quantization bins in a few steps.
    # This is NOT a real-data LR search or a change to the experiment's 1e-4 LR.
    config = replace(config, learning_rate=.01)
    rng = np.random.default_rng(412)
    x = rng.integers(0, 256, (193, 36), dtype=np.uint8)
    teacher, _ = forward(x, load_layers(), 'ann')
    batch = {'features': torch.from_numpy(x).to(device),
             'labels': torch.from_numpy(rng.integers(0, 2, len(x), dtype=np.uint8)).to(device),
             'teacher': torch.from_numpy(teacher).to(device)}
    a, b = R1T16().to(device), R1T16().to(device)
    oa = torch.optim.Adam(a.parameters(), lr=config.learning_rate, foreach=False)
    ob = torch.optim.Adam(b.parameters(), lr=config.learning_rate, foreach=False)
    for _ in range(4):
        update(a, oa, batch, config)
    for _ in range(2):
        update(b, ob, batch, config)
    identity = {'synthetic_check': True, 'config': asdict(config)}
    state = initial_state()
    state['global_step'] = 2
    with tempfile.TemporaryDirectory(dir=output) as temporary:
        path = Path(temporary)/'synthetic.pt'
        checkpoint(path, b, ob, state, identity)
        c = R1T16().to(device)
        oc = torch.optim.Adam(c.parameters(), lr=config.learning_rate, foreach=False)
        recovered = restore(path, c, oc, identity)
        if recovered['global_step'] != 2:
            raise AssertionError('Checkpoint step changed')
        for _ in range(2):
            update(c, oc, batch, config)
        for key, tensor in a.state_dict().items():
            if not torch.equal(tensor, c.state_dict()[key]):
                raise AssertionError('Resumed student is not bit-exact')
        for sa, sc in zip(oa.state.values(), oc.state.values()):
            for key in sa:
                if not torch.equal(sa[key], sc[key]):
                    raise AssertionError('Adam moments/steps changed on resume')
        raw, spikes = c(batch['features'])
        explicit, cost = forward(x, c.export_layers(), STAGE, STEPS, ENCODERS, 'spike_add')
        np.testing.assert_array_equal(raw.detach().cpu().numpy(), explicit)
        if cost.executed_synaptic_ac != int(spikes[:, 0].sum()*32):
            raise AssertionError('Updated student AC differs')
        changed = sum(not np.array_equal(w, base[0]) for (w, _), base in zip(c.export_layers(), load_layers()))
        if changed != 3:
            raise AssertionError('Synthetic export test must actually change all int8 matrices')
        export(Path(temporary)/'student.npz', c)
        saved_layers = load_export(Path(temporary)/'student.npz')
        for actual, expected in zip(saved_layers, c.export_layers()):
            for aa, ee in zip(actual, expected):
                np.testing.assert_array_equal(aa, ee)
    return {'status': 'passed', 'data': '193 synthetic features/random labels only',
            'synthetic_perturbation_learning_rate': config.learning_rate,
            'uninterrupted_vs_resumed_steps': 4, 'adam_and_weights_bit_exact': True,
            'updated_export_explicit_spike_exact': True, 'changed_int8_matrices': changed}


def hardcase_check(model, data, raw, spikes):
    """Additional Dev inference-only checks of errors, HD targets and boundaries."""
    y = data.roles['train_dev']['labels'].cpu().numpy()
    p = probabilities(raw)
    hd = np.zeros(len(raw), bool)
    for seq in data.sequences['train_dev']:
        if seq['row']['high_density']:
            hd[seq['lo']:seq['hi']] = True
    def spaced(ids):
        return ids[np.linspace(0, len(ids)-1, min(128, len(ids)), dtype=int)] if len(ids) else ids
    groups = {
        'false_negatives': spaced(np.flatnonzero((y == 1) & (p < THRESHOLD))),
        'false_positives': spaced(np.flatnonzero((y == 0) & (p >= THRESHOLD))),
        'hd_targets': spaced(np.flatnonzero(hd & (y == 1))),
        'decision_boundary': np.argsort(np.abs(p-THRESHOLD), kind='stable')[:128],
        'largest_logits': np.argsort(np.max(np.abs(raw), axis=1), kind='stable')[-128:],
    }
    ids = np.unique(np.concatenate(list(groups.values())))
    x = data.roles['train_dev']['features'][torch.as_tensor(ids, device=data.device)].cpu().numpy()
    expected, cost = forward(x, model.export_layers(), STAGE, STEPS, ENCODERS, 'spike_add')
    np.testing.assert_array_equal(raw[ids], expected)
    if cost.executed_synaptic_ac != int(spikes[ids, 0].sum()*32):
        raise AssertionError('Hardcase spike accounting mismatch')
    return {'status': 'passed', 'unique_candidates': len(ids),
            'groups': {key: len(value) for key, value in groups.items()},
            'backward_calls': 0, 'optimizer_steps': 0, 'selection_changes': 0}


def check(cache, directory, config=Config(), device_name='cuda'):
    device = setup_device(device_name)
    data = ResidentData(cache, device)
    identity = run_identity(cache, config, engine(device))
    started = time.perf_counter()
    with lock_directory(directory, identity) as directory:
        if (directory/'last.pt').exists():
            raise ValueError('Training has started; use another run directory for a fresh preflight')
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model = R1T16().to(device)
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for (w, s), (ow, os) in zip(model.export_layers(), load_layers()):
            np.testing.assert_array_equal(w, ow)
            np.testing.assert_array_equal(s, os)
        activation = torch.arange(256, dtype=torch.float64, device=device)
        np.testing.assert_array_equal(cast(torch.Tensor, CountSTE.apply(activation)).cpu().numpy(),
                                      (np.arange(256)*STEPS+ENCODERS[0].initial_membrane)//255)
        result, raw, spikes = evaluate(model, data, config.batch_size)
        reference_cost = 0
        for lo in range(0, len(raw), config.batch_size):
            x = data.roles['train_dev']['features'][lo:lo+config.batch_size].cpu().numpy()
            expected, cost = forward(x, load_layers(), STAGE, STEPS, ENCODERS, 'count_reference')
            np.testing.assert_array_equal(raw[lo:lo+len(x)], expected)
            reference_cost += cost.rate_network_ac
        if result['cost']['inference_equivalent_ac'] != reference_cost:
            raise AssertionError('Full-development AC recount failed')
        original = read_json(ANCHOR_DEV)
        for group in ('overall', 'high_density', 'ordinary'):
            for key, value in original[group].items():
                if result[group][key] != value:
                    raise AssertionError(f'Original task_rescore metric changed: {group}/{key}')
        if (result['cost']['inference_equivalent_ac'] != original['executed_neural_ac']
                or result['cost']['inference_equivalent_mac_dense_proposal'] != original['executed_neural_mac']):
            raise AssertionError('Original task_rescore inference cost changed')
        hardcases = hardcase_check(model, data, raw, spikes)
        teacher = data.roles['train_dev']['teacher'].cpu().numpy()
        frozen = score(data, teacher, np.zeros((len(teacher), 2), np.int64))
        frozen['cost'] = {'mac_sparse_proposal': result['cost']['frozen_ann_sparse_mac'],
                          'mac_dense_proposal': result['cost']['frozen_ann_dense_mac'], 'ac': 0}
        # A real-data forward/backward verifies working gradients and 4060 memory.
        # No optimizer is created for this model and no real-data update occurs.
        order = data.order(1, config.seed)
        batch = data.batch('train_main', order[:config.batch_size])
        forward_start = time.perf_counter()
        output, _ = model(batch['features'])
        terms = losses(output, batch['labels'], batch['teacher'], config)
        terms[0].backward()
        gradients = [p.grad for p in model.parameters()]
        if any(g is None or not bool(torch.count_nonzero(g)) for g in gradients):
            raise AssertionError('Missing/nonworking straight-through gradient')
        require_finite([g for g in gradients if g is not None], 'preflight gradients')
        gradnorm = [float(g.double().norm()) for g in gradients if g is not None]
        for k, v in model.state_dict().items():
            if not torch.equal(v, before[k]):
                raise AssertionError('Real-data preflight changed student weights/scales')
        model.zero_grad(set_to_none=True)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        backward_seconds = time.perf_counter()-forward_start
        resume = synthetic_resume(device, directory, config)
        ready = {'status': 'passed', 'identity': identity, 'formal_training_started': False,
                 'real_data_optimizer_steps': 0, 'reserved_sequences_read': 0,
                 'real_data_backward_no_update': True, 'gradient_norm_per_matrix': gradnorm,
                 'one_batch_forward_backward_seconds': backward_seconds,
                 'all_dev_candidates_exact_original_r1t16': len(raw),
                 'original_task_rescore_metrics_exact': True, 'hardcases': hardcases,
                 'uint8_encoder_exhaustive': True, 'full_dev_ac_exact': True,
                 'synthetic_resume': resume, 'frozen_ann': frozen, 'initial_student': result,
                 'data': data.stats, 'memory': memory(device),
                 'check_seconds': time.perf_counter()-started}
        atomic_json(directory/'ready.json', ready)
        export(directory/'initial_student_int8.npz', model)
        atomic_json(directory/'progress.json', {'status': 'ready_not_trained', 'completed_epochs': 0,
                                              'formal_training_started': False})
        return ready
