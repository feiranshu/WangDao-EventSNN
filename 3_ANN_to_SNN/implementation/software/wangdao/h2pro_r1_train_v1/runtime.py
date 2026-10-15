"""User-started training, atomic minibatch recovery, and deterministic replay."""
from __future__ import annotations

import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from ..h2pro_longtrain_v1.storage import (PauseControl, PauseRequested, all_rng,
                                       load, save_atomic, set_rng)
from ..h2pro_direct_v1.core import load_layers
from .common import Config, atomic_json, lock_directory, read_json, run_identity, sha256
from .data import ResidentData
from .evaluation import delta, evaluate, export
from .model import R1T16, update


def setup_device(requested='cuda'):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if requested not in ('cuda', 'cpu'):
        raise ValueError('Select cuda or cpu explicitly')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: use C:\\Anaconda\\envs\\gazelle-train\\python.exe')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # No process-memory cap, AMP, arbitrary activation guard or silent CPU fallback.
    return torch.device('cuda:0' if requested == 'cuda' else 'cpu')


def engine(device):
    return {'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'cuda': str(getattr(getattr(torch, 'version', None), 'cuda', None)), 'device': str(device),
            'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
            'forward_precision': 'FP64 exact integer sums/scale decode, FP32 original logits',
            'optimizer_precision': 'FP32', 'deterministic': True}


def memory(device):
    if device.type != 'cuda':
        return {'peak_allocated_bytes': 0, 'peak_reserved_bytes': 0}
    return {'peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}


def fresh_totals():
    return {'events': 0, 'loss_sums': [0., 0., 0.], 'spikes': [0, 0],
            'gradient_norm_max': 0., 'logit_absmax': 0., 'train_seconds': 0.}


def add_totals(totals, result, seconds):
    n = result['events']
    totals['events'] += n
    totals['loss_sums'] = [a+b*n for a, b in zip(totals['loss_sums'], result['losses'])]
    totals['spikes'] = [a+b for a, b in zip(totals['spikes'], result['spikes'])]
    totals['gradient_norm_max'] = max(totals['gradient_norm_max'], result['gradient_norm'])
    totals['logit_absmax'] = max(totals['logit_absmax'], result['logit_absmax'])
    totals['train_seconds'] += seconds


def initial_state():
    return {'cursor': {'epoch': 1, 'batch': 0, 'phase': 'train'}, 'global_step': 0,
            'totals': fresh_totals(), 'history': [], 'best_epoch': 0, 'best_score': [-1., -1.]}


def checkpoint(path, model, optimizer, state, identity):
    save_atomic(path, {'identity': identity, 'model': model.state_dict(),
                       'optimizer': optimizer.state_dict(), 'state': state, 'rng': all_rng(),
                       'resumable': True})


def restore(path, model, optimizer, identity):
    saved = load(path)
    if saved.get('identity') != identity or saved.get('resumable') is not True:
        raise ValueError('Checkpoint identity mismatch; do not reuse another experiment')
    model.load_state_dict(saved['model'], strict=True)
    model.verify_frozen_scales()
    optimizer.load_state_dict(saved['optimizer'])
    set_rng(saved['rng'])
    return saved['state']


def emit(directory, status, state, data, device, target_epochs):
    report = {'status': status, 'cursor': state['cursor'], 'global_step': state['global_step'],
              'completed_epochs': len(state['history']), 'target_total_epochs': target_epochs,
              'best_epoch': state['best_epoch'], 'totals': state['totals'],
              'data': data.stats, **memory(device), 'updated_unix_s': time.time(),
              'resumable_checkpoint': str(directory/'last.pt')}
    atomic_json(directory/'progress.json', report)
    with (directory/'progress.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(report, allow_nan=False)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'data'}, allow_nan=False), flush=True)


def train(cache, directory, config=Config(), epochs=5, checkpoint_batches=10, device_name='cuda',
          *, _test_data=None, _test_pause_after=None):
    if type(epochs) is not int or epochs < 1 or type(checkpoint_batches) is not int or checkpoint_batches < 1:
        raise ValueError('Positive total epochs and checkpoint interval required')
    config.validate()
    device = setup_device(device_name)
    data = ResidentData(cache, device) if _test_data is None else _test_data
    identity = run_identity(cache, config, engine(device)) if _test_data is None else _test_data.identity
    with lock_directory(directory, identity) as directory:
        ready = read_json(directory/'ready.json')
        if ready.get('identity') != identity or ready.get('status') != 'passed':
            raise ValueError('Run the matching preflight check before training')
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        random.seed(config.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(config.seed)
            torch.cuda.reset_peak_memory_stats(device)
        model = R1T16().to(device)
        model.verify_frozen_scales()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=0., foreach=False)
        state = initial_state()
        if (directory/'last.pt').exists():
            state = restore(directory/'last.pt', model, optimizer, identity)
        else:
            zero = ready['initial_student']
            state['best_score'] = [zero['overall']['iou'], zero['overall']['recall']]
            checkpoint(directory/'best.pt', model, optimizer, state, identity)
            export(directory/'best_int8.npz', model)
            # Commit last only after the epoch-zero best artifacts exist. A kill
            # during initialization can then safely recreate all three files.
            checkpoint(directory/'last.pt', model, optimizer, state, identity)
        cursor = state['cursor']
        batches = (data.size('train_main')+config.batch_size-1)//config.batch_size
        if (cursor['epoch'] != len(state['history'])+1 or cursor['phase'] not in ('train', 'evaluate')
                or not 0 <= cursor['batch'] <= batches
                or state['totals']['events'] != min(cursor['batch']*config.batch_size, data.size('train_main'))):
            raise ValueError('Checkpoint cursor/sample coverage is inconsistent')
        (directory/'PAUSE').unlink(missing_ok=True)
        def save():
            checkpoint(directory/'last.pt', model, optimizer, state, identity)
        def write_history():
            atomic_json(directory/'training.json', {'identity': identity, 'epochs': state['history'],
                        'best_epoch': state['best_epoch'], 'initial_student': ready['initial_student'],
                        'frozen_ann': ready['frozen_ann']})
        try:
            with PauseControl(directory) as control:
                def pause_point():
                    if control.pending():
                        raise PauseRequested()
                while state['cursor']['epoch'] <= epochs:
                    cursor = state['cursor']
                    epoch = cursor['epoch']
                    emit(directory, 'running', state, data, device, epochs)
                    order = data.order(epoch, config.seed)
                    if cursor['phase'] == 'train':
                        for batch in range(cursor['batch'], batches):
                            pause_point()
                            started = time.perf_counter()
                            indices = order[batch*config.batch_size:(batch+1)*config.batch_size]
                            result = update(model, optimizer, data.batch('train_main', indices), config)
                            add_totals(state['totals'], result, time.perf_counter()-started)
                            cursor['batch'] = batch+1
                            state['global_step'] += 1
                            if cursor['batch'] % checkpoint_batches == 0:
                                save()
                                emit(directory, 'running', state, data, device, epochs)
                            if _test_pause_after is not None and state['global_step'] == _test_pause_after:
                                raise PauseRequested()
                        if state['totals']['events'] != data.size('train_main'):
                            raise AssertionError('Epoch did not visit every candidate exactly once')
                        cursor['phase'] = 'evaluate'
                        save()
                    pause_point()
                    result, raw, spikes = evaluate(model, data, config.batch_size, pause_point)
                    result['delta_to_ann_pp'] = delta(result, ready['frozen_ann'])
                    result['delta_to_initial_r1t16_pp'] = delta(result, ready['initial_student'])
                    totals = state['totals']
                    entry = {'epoch': epoch, 'global_step': state['global_step'],
                             'training': dict(totals, mean_losses=[v/totals['events'] for v in totals['loss_sums']]),
                             'evaluation': result, 'memory': memory(device)}
                    entry['student_updates'] = {
                        'changed_int8_weights_per_layer': [int(np.count_nonzero(w != base[0]))
                            for (w, _), base in zip(model.export_layers(), load_layers())],
                        'master_weight_delta_l2_per_layer': [float(torch.linalg.vector_norm(
                            parameter.detach().cpu()-torch.from_numpy(w.astype(np.float32)*s)))
                            for parameter, (w, s) in zip(model.parameters(), load_layers())]}
                    rank = [result['overall']['iou'], result['overall']['recall']]
                    state['history'].append(entry)
                    state['cursor'] = {'epoch': epoch+1, 'batch': 0, 'phase': 'train'}
                    state['totals'] = fresh_totals()
                    epoch_dir = directory/'epochs'
                    epoch_dir.mkdir(exist_ok=True)
                    atomic_json(epoch_dir/f'epoch_{epoch:03}.json', entry)
                    export(epoch_dir/f'epoch_{epoch:03}_int8.npz', model)
                    if rank > state['best_score']:
                        state['best_score'], state['best_epoch'] = rank, epoch
                        checkpoint(directory/'best.pt', model, optimizer, state, identity)
                        export(directory/'best_int8.npz', model)
                    # Retain every epoch's optimizer/RNG, including high-recall
                    # candidates that are not the single IoU-best checkpoint.
                    checkpoint(epoch_dir/f'epoch_{epoch:03}.pt', model, optimizer, state, identity)
                    save()
                    write_history()
                    emit(directory, 'epoch_complete', state, data, device, epochs)
                save()
                # Repair a missing/stale JSON mirror if the previous process died
                # after committing its final checkpoint but before writing it.
                write_history()
                emit(directory, 'complete', state, data, device, epochs)
        except PauseRequested:
            # Only called at committed update boundaries. During evaluation the
            # checkpoint restarts that evaluation, never repeats an optimizer step.
            save()
            write_history()
            emit(directory, 'paused', state, data, device, epochs)
        except Exception as error:
            # Do not replace a good checkpoint with potentially invalid parameters.
            stamp = time.time_ns()
            failure = {'error': repr(error), 'traceback': traceback.format_exc(),
                       'cursor': state['cursor'], 'global_step': state['global_step'],
                       'resumable_checkpoint': str(directory/'last.pt'),
                       'last_sha256': sha256(directory/'last.pt'), 'timestamp_ns': stamp}
            atomic_json(directory/f'failure_{stamp}.json', failure)
            emit(directory, 'failed', state, data, device, epochs)
            raise
        return read_json(directory/'progress.json')
