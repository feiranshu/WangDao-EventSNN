"""Sequential paired stages, full-epoch retention, and batch-exact recovery."""
from __future__ import annotations

import json
import time
import traceback

import torch

from ..h2pro_r1_train_v1.data import ResidentData
from ..h2pro_r1_train_v1.model import R1T16, update
from ..h2pro_r1_train_v1.evaluation import delta, evaluate, export
from ..h2pro_r1_train_v1.runtime import (
    setup_device, memory, checkpoint, restore, fresh_totals, add_totals,
)
from ..h2pro_longtrain_v1.storage import PauseControl, PauseRequested, load, all_rng
from .common import (
    ARMS, MILESTONES, SOURCE_EPOCH, DEFAULT_CACHE, DEFAULT_RUN, SOURCE_RUN,
    atomic_json, lock_directory, inputs, independent_output, read_json, sha256,
    arm_identity, arm_config, verify_prepared, digest,
)
from .comparison import compare


def status(suite):
    workflow = read_json(suite / 'suite_status.json') if (suite / 'suite_status.json').exists() else {}
    return {'suite': str(suite), 'workflow_status': workflow.get('status', 'not_prepared'),
            'target_additional_epochs': workflow.get('target_additional_epochs'),
            'pause_requested': (suite / 'PAUSE').exists(),
            'arms': {arm: read_json(suite / arm / 'progress.json')
                     if (suite / arm / 'progress.json').exists() else {'status': 'not_prepared'} for arm in ARMS},
            'completed_comparisons': [e for e in MILESTONES if (suite / f'comparison_added_{e:03}.json').exists()]}


def emit(directory, arm, state, data, target, status_name, lr):
    report = {'status': status_name, 'arm': arm, 'learning_rate': lr,
              'completed_additional_epochs': len(state['history']), 'source_epoch': SOURCE_EPOCH,
              'effective_completed_epoch': SOURCE_EPOCH + len(state['history']),
              'target_additional_epochs': target, 'cursor': state['cursor'],
              'global_step': state['global_step'],
              'additional_updates': state['global_step'] - state['origin_global_step'],
              'best_added_epoch': state['best_epoch'], 'best_effective_epoch': SOURCE_EPOCH + state['best_epoch'],
              'totals': state['totals'], 'data': data.stats, **memory(data.device),
              'resumable_checkpoint': str(directory / 'last.pt'), 'updated_unix_s': time.time()}
    atomic_json(directory / 'progress.json', report)
    with (directory / 'progress.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(report, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('data', 'totals')}, allow_nan=False), flush=True)
    return report


def validate_cursor(state, data, config):
    c = state['cursor']; n = data.size('train_main')
    batches = (n + config.batch_size - 1) // config.batch_size
    if (state['source_epoch'] != SOURCE_EPOCH or c['epoch'] != len(state['history']) + 1
            or c['phase'] not in ('train', 'evaluate') or not 0 <= c['batch'] <= batches
            or (c['phase'] == 'evaluate' and c['batch'] != batches)
            or state['totals']['events'] != min(c['batch'] * config.batch_size, n)
            or state['global_step'] - state['origin_global_step'] != len(state['history']) * batches + c['batch']):
        raise ValueError('Checkpoint cursor/update/sample coverage is inconsistent')


def historical_replay(identity, arm, absolute, model, optimizer, entry, synthetic):
    """Record a diagnostic against the old constant-LR path; no metric-based stop."""
    if arm != 'control' or synthetic:
        return None
    from pathlib import Path
    source = Path(identity['paths']['source'])
    path = source / 'epochs' / f'epoch_{absolute:03}.pt'
    if absolute > 200 or not path.exists():
        return None
    previous = load(path); old = previous['state']['history'][-1]
    fields = {
        'model_exact': digest(model.state_dict()) == digest(previous['model']),
        'adam_exact': digest(optimizer.state_dict()) == digest(previous['optimizer']),
        'rng_exact': digest(all_rng()) == digest(previous['rng']),
        'loss_sums_exact': entry['training']['loss_sums'] == old['training']['loss_sums'],
        'metrics_and_cost_exact': all(entry['evaluation'][k] == old['evaluation'][k]
                                      for k in ('overall', 'ordinary', 'high_density', 'cost')),
    }
    return {'effective_epoch': absolute, 'reference_sha256': sha256(path),
            **fields, 'all_exact': all(fields.values())}


def train_arm(data, suite, arm, identity, ready, config, target, checkpoint_batches,
              pause=lambda: None, *, _test_pause_after=None):
    directory = suite / arm
    cfg = arm_config(config, arm); aid = arm_identity(identity, arm)
    with lock_directory(directory, aid):
        model = R1T16().to(data.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=0., foreach=False)
        state = restore(directory / 'last.pt', model, optimizer, aid)
        validate_cursor(state, data, cfg)
        if (any(float(s['step']) != state['global_step'] for s in optimizer.state.values())
                or any(g['lr'] != cfg.learning_rate for g in optimizer.param_groups)):
            raise ValueError('Adam step or learning rate differs from this arm checkpoint')
        if data.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(data.device)
        def save():
            checkpoint(directory / 'last.pt', model, optimizer, state, aid)
        def history():
            result = {'identity': aid, 'source_epoch': SOURCE_EPOCH,
                      'initial_student': ready['initial_student'], 'frozen_ann': ready['frozen_ann'],
                      'original_r1_before_training': ready['original_r1_before_training'],
                      'best_added_epoch': state['best_epoch'], 'epochs': state['history']}
            path = directory / 'training.json'
            if not path.exists() or read_json(path) != result:
                atomic_json(path, result)
        if len(state['history']) >= target:
            history()
            report = read_json(directory / 'progress.json') if (directory / 'progress.json').exists() else {}
            if (report.get('completed_additional_epochs') != len(state['history'])
                    or report.get('status') not in ('complete', 'epoch_complete')):
                emit(directory, arm, state, data, target, 'complete', cfg.learning_rate)
            return {'status': 'already_complete', 'completed_additional_epochs': len(state['history'])}
        batches = (data.size('train_main') + cfg.batch_size - 1) // cfg.batch_size
        try:
            while state['cursor']['epoch'] <= target:
                pause()
                c = state['cursor']; added = c['epoch']; absolute = SOURCE_EPOCH + added
                emit(directory, arm, state, data, target, 'running', cfg.learning_rate)
                order = data.order(absolute, cfg.seed)
                if c['phase'] == 'train':
                    for batch_number in range(c['batch'], batches):
                        pause(); started = time.perf_counter()
                        ids = order[batch_number * cfg.batch_size:(batch_number + 1) * cfg.batch_size]
                        result = update(model, optimizer, data.batch('train_main', ids), cfg)
                        add_totals(state['totals'], result, time.perf_counter() - started)
                        c['batch'] = batch_number + 1; state['global_step'] += 1
                        if c['batch'] % checkpoint_batches == 0:
                            save(); emit(directory, arm, state, data, target, 'running', cfg.learning_rate)
                        if (_test_pause_after is not None
                                and state['global_step'] - state['origin_global_step'] == _test_pause_after):
                            raise PauseRequested()
                    if state['totals']['events'] != data.size('train_main'):
                        raise AssertionError('Epoch did not cover every Main candidate')
                    c['phase'] = 'evaluate'; save()
                pause()
                evaluation, _, _ = evaluate(model, data, cfg.batch_size, pause)
                evaluation['delta_to_ann_pp'] = delta(evaluation, ready['frozen_ann'])
                evaluation['delta_to_source159_pp'] = delta(evaluation, ready['initial_student'])
                evaluation['delta_to_original_r1_pp'] = delta(evaluation, ready['original_r1_before_training'])
                totals = state['totals']; n = totals['events']
                entry = {'added_epoch': added, 'effective_epoch': absolute, 'global_step': state['global_step'],
                         'learning_rate': cfg.learning_rate,
                         'training': dict(totals, mean_losses=[v / n for v in totals['loss_sums']]),
                         'evaluation': evaluation, 'memory': memory(data.device)}
                replay = historical_replay(identity, arm, absolute, model, optimizer, entry,
                                           data.stats.get('synthetic_only', False))
                if replay is not None:
                    entry['historical_control_replay'] = replay
                rank = [evaluation['overall']['iou'], evaluation['overall']['recall']]
                state['history'].append(entry)
                state['cursor'] = dict(epoch=added + 1, batch=0, phase='train')
                state['totals'] = fresh_totals()
                epoch_dir = directory / 'epochs'; epoch_dir.mkdir(exist_ok=True)
                atomic_json(epoch_dir / f'added_{added:03}.json', entry)
                export(epoch_dir / f'added_{added:03}_int8.npz', model)
                if rank > state['best_score']:
                    state['best_score'] = rank; state['best_epoch'] = added
                    checkpoint(directory / 'best.pt', model, optimizer, state, aid)
                    export(directory / 'best_int8.npz', model)
                checkpoint(epoch_dir / f'added_{added:03}.pt', model, optimizer, state, aid)
                save(); history()
                emit(directory, arm, state, data, target, 'epoch_complete', cfg.learning_rate)
            return emit(directory, arm, state, data, target, 'complete', cfg.learning_rate)
        except PauseRequested:
            save(); history()
            return emit(directory, arm, state, data, target, 'paused', cfg.learning_rate)
        except Exception as error:
            stamp = time.time_ns()
            atomic_json(directory / f'failure_{stamp}.json', {
                'error': repr(error), 'traceback': traceback.format_exc(), 'cursor': state['cursor'],
                'global_step': state['global_step'], 'resumable_checkpoint': str(directory / 'last.pt'),
                'last_sha256': sha256(directory / 'last.pt')})
            emit(directory, arm, state, data, target, 'failed', cfg.learning_rate)
            raise


def train_suite(cache=DEFAULT_CACHE, suite=DEFAULT_RUN, source=SOURCE_RUN,
                additional_epochs=50, checkpoint_batches=10, device_name='cuda', *, _test_context=None):
    if (type(additional_epochs) is not int or additional_epochs < 1
            or type(checkpoint_batches) is not int or checkpoint_batches < 1):
        raise ValueError('Positive total additional epochs and checkpoint interval required')
    device = setup_device(device_name)
    if _test_context is None:
        identity, _, config, _ = inputs(cache, source, device)
        suite = independent_output(suite, cache, source)
    else:
        identity, config, test_data = _test_context
        if test_data.stats.get('synthetic_only') is not True:
            raise ValueError('Test context requires synthetic data')
    with lock_directory(suite, identity):
        ready = verify_prepared(suite, identity)
        (suite / 'PAUSE').unlink(missing_ok=True)
        targets = sorted({e for e in MILESTONES if e <= additional_epochs} | {additional_epochs})
        with PauseControl(suite) as control:
            def pause():
                if control.pending():
                    raise PauseRequested()
            try:
                atomic_json(suite / 'suite_status.json', {'status': 'running', 'target_additional_epochs': additional_epochs})
                data = ResidentData(cache, device) if _test_context is None else test_data
                for target in targets:
                    for arm in ARMS:
                        pause()
                        result = train_arm(data, suite, arm, identity, ready, config, target, checkpoint_batches, pause)
                        if result['status'] == 'paused':
                            raise PauseRequested()
                    pause()
                    comparison = compare(suite, target, ready)
                    print(json.dumps({'stage': 'paired_comparison', 'added_epochs': target,
                        'low_lr_minus_control_pp': comparison['low_lr_minus_control_pp'],
                        'path': str(suite / f'comparison_added_{target:03}.json')}), flush=True)
                outcome = {**status(suite), 'status': 'complete', 'workflow_status': 'complete',
                           'target_additional_epochs': additional_epochs}
            except PauseRequested:
                outcome = {**status(suite), 'status': 'paused', 'workflow_status': 'paused',
                           'target_additional_epochs': additional_epochs}
            except Exception:
                atomic_json(suite / 'suite_status.json', {**status(suite), 'status': 'failed',
                            'workflow_status': 'failed', 'target_additional_epochs': additional_epochs})
                raise
        atomic_json(suite / 'suite_status.json', outcome)
        return outcome
