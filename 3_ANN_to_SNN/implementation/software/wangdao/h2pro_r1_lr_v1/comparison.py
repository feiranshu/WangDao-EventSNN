"""Milestone comparisons from saved full-Dev evaluations; never refit on Dev."""
from __future__ import annotations

import statistics

from ..h2pro_r1_train_v1.evaluation import delta
from .common import ARMS, SOURCE_EPOCH, atomic_json, read_json, sha256


def metric_row(result):
    return {**{k: result['overall'][k] for k in ('iou', 'precision', 'recall', 'tp', 'fp', 'fn')},
            'hd_recall': result['high_density']['recall'], 'cost': result['cost']}


def cost_delta(result, reference):
    c, b = result['cost'], reference['cost']
    return {'mac': c['inference_equivalent_mac_sparse_proposal'] - b['inference_equivalent_mac_sparse_proposal'],
            'ac': c['inference_equivalent_ac'] - b['inference_equivalent_ac'],
            'ac_relative_pct': 100 * (c['inference_equivalent_ac'] / b['inference_equivalent_ac'] - 1)
                if b['inference_equivalent_ac'] else None}


def compare(suite, added, ready):
    result = {'added_epochs': added, 'effective_epoch': SOURCE_EPOCH + added,
              'source_epoch': SOURCE_EPOCH, 'threshold': .35, 'role': 'train_dev',
              'initial_student': ready['initial_student'], 'frozen_ann': ready['frozen_ann'],
              'original_r1_before_training': ready['original_r1_before_training'], 'arms': {}}
    endpoints = {}
    for arm in ARMS:
        directory = suite / arm
        training = read_json(directory / 'training.json')
        if len(training['epochs']) < added:
            raise ValueError('Both arms must finish this milestone before comparison')
        history = training['epochs'][:added]
        entry = read_json(directory / 'epochs' / f'added_{added:03}.json')
        if entry != history[-1] or entry['added_epoch'] != added:
            raise ValueError('Milestone history/epoch mismatch')
        endpoint = entry['evaluation']; endpoints[arm] = endpoint
        candidates = [(0, ready['initial_student'])] + [(e['added_epoch'], e['evaluation']) for e in history]
        best_added, best = max(candidates, key=lambda x: (x[1]['overall']['iou'], x[1]['overall']['recall']))
        window = history[-min(10, len(history)):]
        trend = [dict(added_epoch=e['added_epoch'], effective_epoch=e['effective_epoch'],
                      **metric_row(e['evaluation']), losses=e['training']['mean_losses'], memory=e['memory'])
                 for e in history]
        result['arms'][arm] = {
            'learning_rate': training['identity']['config']['learning_rate'],
            'endpoint': endpoint, 'memory': entry['memory'],
            'best_added_epoch': best_added, 'best_effective_epoch': SOURCE_EPOCH + best_added,
            'best': best, 'delta_to_source159_pp': delta(endpoint, ready['initial_student']),
            'delta_to_ann_pp': delta(endpoint, ready['frozen_ann']),
            'cost_delta_to_source159': cost_delta(endpoint, ready['initial_student']),
            'last_window_epochs': len(window),
            'last_window': {name: {'mean': statistics.mean(values), 'std': statistics.pstdev(values)}
                for name, values in ((name, [metric_row(e['evaluation'])[name] for e in window])
                                     for name in ('iou', 'precision', 'recall', 'hd_recall'))},
            'learning_trend': trend,
            'checkpoint': str(directory / 'epochs' / f'added_{added:03}.pt'),
            'checkpoint_sha256': sha256(directory / 'epochs' / f'added_{added:03}.pt'),
        }
    result['low_lr_minus_control_pp'] = delta(endpoints['low_lr'], endpoints['control'])
    result['low_lr_minus_control_cost'] = cost_delta(endpoints['low_lr'], endpoints['control'])
    path = suite / f'comparison_added_{added:03}.json'
    if path.exists():
        if read_json(path) != result:
            raise ValueError('Existing milestone comparison differs')
    else:
        atomic_json(path, result)
    return result
