"""Label-aware classification calibration; no network parameter updates."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..h2pro_snn.foundation import read_json, sha256
from ..h2pro_direct_v1.workflow import load_sequence
from ..h2pro_direct_v1.core import quantize_hidden, validate_layers
from ..h2pro_calibration_v1.core import (
    Encoder, count_spikes, decode, forward, load_layers, margin, probabilities,
)
from ..h2pro_calibration_v1.calibration import parameter_key, refinement_grid
from .common import atomic_json, identity, inputs, lock_directory, protocol


def sample_data(cache, manifest, previous_sample):
    """Same rows as MSE calibration; include Proposal misses in the task estimate."""
    settings = protocol()
    rng = np.random.default_rng(settings['sample_seed'])
    if len(manifest['sequences']) != len(previous_sample):
        raise ValueError('Previous calibration sample length mismatch')
    parts, labels, groups, sequences, receipts = [], [], [], [], []
    offset = 0
    for index, row in enumerate(manifest['sequences']):
        arrays = load_sequence(cache, row)
        x, y, c = arrays['features'], arrays['label'], arrays['candidate']
        n = len(x)
        k = min(settings['candidate_rows_per_sequence'], n)
        if not k or not len(y):
            raise ValueError('Nonempty calibration sequences and candidates required')
        ids = np.sort(rng.choice(n, k, replace=False))
        digest = hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()
        old = previous_sample[index]
        if (old['sequence'] != row['sequence'] or old['sampled_candidates'] != k
                or old['available_candidates'] != n
                or old['selected_row_indices_sha256'] != digest):
            raise ValueError('Task calibration must use exactly the prior sampled rows')
        selected_y = y[c][ids].copy()
        parts.append(x[ids].copy())
        labels.append(selected_y)
        groups.extend([bool(row['high_density'])] * k)
        sequences.extend([index] * k)
        receipts.append({**old, 'start': offset, 'stop': offset + k,
            'events': len(y), 'target_events': int(y.sum()),
            'sampled_targets': int(selected_y.sum()),
            'proposal_fn': int(y[~c].sum()),
            'proposal_tn': int(np.count_nonzero(~c) - y[~c].sum())})
        offset += k
    return (np.concatenate(parts), np.concatenate(labels), np.asarray(groups, bool),
            np.asarray(sequences, np.int32), receipts)


def task_metrics(raw, teacher, labels, groups, sequences, receipts):
    """Sampling-weighted task estimates; full Dev metrics are computed separately."""
    predicted = probabilities(raw) >= protocol()['decision_threshold']
    expected = probabilities(teacher) >= protocol()['decision_threshold']
    positive = labels == 1
    teacher_right = expected == positive
    harm = teacher_right & (predicted != positive)
    corrected = ~teacher_right & (predicted == positive)
    delta = margin(raw) - margin(teacher)
    members = {'ordinary': [], 'high_density': []}
    for index, receipt in enumerate(receipts):
        ids = np.flatnonzero(sequences == index)
        if len(ids) != receipt['sampled_candidates']:
            raise ValueError('Sequence sample accounting mismatch')
        weight = receipt['available_candidates'] / len(ids)
        p, y = predicted[ids], positive[ids]
        counts = np.array([np.count_nonzero(p & y), np.count_nonzero(p & ~y),
                           np.count_nonzero(~p & y), np.count_nonzero(~p & ~y)], np.float64) * weight
        counts[2] += receipt['proposal_fn']
        counts[3] += receipt['proposal_tn']
        group = 'high_density' if bool(groups[ids[0]]) else 'ordinary'
        members[group].append((counts / receipt['events'], float(harm[ids].mean()),
                               float(corrected[ids].mean()), float(np.square(delta[ids]).mean())))
    result = {}
    for group, rows in members.items():
        if not rows:
            raise ValueError('Both density groups are required')
        tp, fp, fn, tn = np.mean([r[0] for r in rows], axis=0)
        result[group] = {'iou': float(tp / (tp + fp + fn)) if tp + fp + fn else 0.,
            'precision': float(tp / (tp + fp)) if tp + fp else 0.,
            'recall': float(tp / (tp + fn)) if tp + fn else 0.,
            'estimated_normalized_confusion': dict(zip(('tp', 'fp', 'fn', 'tn'), map(float, (tp, fp, fn, tn)))),
            'teacher_correct_harm_rate': float(np.mean([r[1] for r in rows])),
            'teacher_errors_corrected_rate': float(np.mean([r[2] for r in rows])),
            'margin_mse': float(np.mean([r[3] for r in rows]))}
    boundary = abs(margin(teacher) - np.log(.35 / .65)) <= protocol()['boundary_margin_width']
    return {'groups': result, 'balanced_iou': sum(g['iou'] for g in result.values()) / 2,
        'balanced_teacher_harm_rate': sum(g['teacher_correct_harm_rate'] for g in result.values()) / 2,
        'balanced_margin_mse': sum(g['margin_mse'] for g in result.values()) / 2,
        'sample_counts': {'tp': int(np.count_nonzero(positive & predicted)),
            'fp': int(np.count_nonzero(~positive & predicted)),
            'fn': int(np.count_nonzero(positive & ~predicted)),
            'tn': int(np.count_nonzero(~positive & ~predicted)),
            'teacher_correct_made_wrong': int(harm.sum()), 'teacher_errors_corrected': int(corrected.sum()),
            'boundary_events': int(boundary.sum()),
            'boundary_flips': int(np.count_nonzero(boundary & (predicted != expected)))}}


class PreparedForward:
    """Cache unchanged first-layer activations and up to eight layer-2 results.

    This is a count-reference search accelerator, never an AC execution claim.
    Selected candidates are independently checked with the original full forward.
    """
    def __init__(self, features, layers, steps):
        validate_layers(layers)
        self.layers, self.steps = layers, steps
        w, scale = layers[0]
        raw = features.astype(np.int32) @ w.astype(np.int32)
        self.first = quantize_hidden(raw.astype(np.float64) * scale.astype(np.float64)[None, :] / 255.)
        self.second_cache = OrderedDict()

    def __call__(self, stage, encoders):
        first, second = encoders
        if first not in self.second_cache:
            counts = count_spikes(self.first, self.steps, first)
            w, scale = self.layers[1]
            raw = counts @ w.astype(np.int32)
            hidden = quantize_hidden(decode(raw, scale, self.steps, first))
            self.second_cache[first] = (hidden, int(counts.sum(dtype=np.int64)))
            if len(self.second_cache) > 8:
                self.second_cache.popitem(last=False)
        self.second_cache.move_to_end(first)
        hidden, spikes1 = self.second_cache[first]
        w, scale = self.layers[2]
        if stage == 'r1':
            raw = hidden.astype(np.int32) @ w.astype(np.int32)
            result = raw.astype(np.float64) * scale.astype(np.float64)[None, :] / 255.
            spikes2 = 0
        elif stage == 'r2':
            counts = count_spikes(hidden, self.steps, second)
            result = decode(counts @ w.astype(np.int32), scale, self.steps, second)
            spikes2 = int(counts.sum(dtype=np.int64))
        else:
            raise ValueError('Prepared search supports only r1/r2')
        return result.astype(np.float32), {'unit_spikes': spikes1 + spikes2,
            'layer1_spikes': spikes1, 'layer2_spikes': spikes2, 'equivalent_ac': 32 * spikes1 + 4 * spikes2}


def task_key(candidate):
    return (-candidate['balanced_iou'], candidate['balanced_teacher_harm_rate'],
            candidate['balanced_margin_mse'], candidate['equivalent_ac'], candidate['key'])


def select_candidates(candidates, baseline):
    best = min(candidates, key=task_key)
    eligible = [c for c in candidates if c['equivalent_ac'] <= baseline['equivalent_ac']]
    if not eligible:
        raise ValueError('The original conversion must remain a budget-eligible candidate')
    return best, min(eligible, key=task_key)


def search_point(features, labels, groups, sequences, receipts, layers, previous, initial=None):
    settings = protocol()
    stage, steps = previous['stage'], previous['steps']
    teacher, _ = forward(features, layers, 'ann', steps)
    prepared = PreparedForward(features, layers, steps)
    candidates = {}
    started = time.perf_counter()

    def assess(encoders):
        key = parameter_key(encoders)
        if key not in candidates:
            raw, cost = prepared(stage, encoders)
            candidates[key] = {'key': str(key), 'encoders': [asdict(e) for e in encoders],
                **cost, **task_metrics(raw, teacher, labels, groups, sequences, receipts)}
        return candidates[key]

    baseline = assess((Encoder(), Encoder()))
    for old in previous['candidates']:
        scored = assess(tuple(Encoder(**e) for e in old['encoders']))
        if ('equivalent_ac' in old and (scored['equivalent_ac'] != old['equivalent_ac']
                or scored['unit_spikes'] != old['unit_spikes']
                or not np.isclose(scored['balanced_margin_mse'], old['balanced_margin_mse'], rtol=1e-12, atol=1e-12))):
            raise AssertionError('Old-pool replay changed arithmetic, costs or sample weighting')
    pool_count = len(candidates)
    rescored = min(candidates.values(), key=task_key)
    mse = assess(tuple(Encoder(**e) for e in previous['accuracy']['encoders']))
    if initial is not None:
        assess(initial)
    grid = [Encoder(theta, theta * q // 4) for theta in settings['threshold_grid']
            for q in settings['initial_phase_quarters']]
    active = 1 if stage == 'r1' else 2
    for path in ('task', 'task_matched_ac'):
        selected = select_candidates(list(candidates.values()), baseline)[path == 'task_matched_ac']
        current = tuple(Encoder(**e) for e in selected['encoders'])
        for phase, passes in (('coarse', settings['coordinate_passes']), ('refine', settings['refinement_passes'])):
            for iteration in range(passes):
                for layer in range(active):
                    local = [assess(current)]
                    choices = grid if phase == 'coarse' else refinement_grid(current[layer], settings)
                    for encoder in choices:
                        proposal = list(current)
                        proposal[layer] = encoder
                        local.append(assess(tuple(proposal)))
                    if path == 'task_matched_ac':
                        local = [c for c in local if c['equivalent_ac'] <= baseline['equivalent_ac']]
                    winner = min(local, key=task_key)
                    current = tuple(Encoder(**e) for e in winner['encoders'])
                print(json.dumps({'stage': 'task_calibrate', 'conversion': stage, 'T': steps,
                    'path': path, 'phase': phase, 'pass': iteration + 1, 'candidates': len(candidates),
                    'calibration_iou_estimate': assess(current)['balanced_iou']}), flush=True)
    best, matched = select_candidates(list(candidates.values()), baseline)
    choices = {'uncalibrated': baseline, 'mse_reference': mse, 'task_rescore': rescored,
               'task': best, 'task_matched_ac': matched}
    for choice in choices.values():
        encoders = tuple(Encoder(**e) for e in choice['encoders'])
        actual, cost = forward(features, layers, stage, steps, encoders, 'count_reference')
        fast, _ = prepared(stage, encoders)
        if (not np.array_equal(actual, fast) or cost.rate_network_ac != choice['equivalent_ac']
                or cost.unit_spikes != choice['unit_spikes']):
            raise AssertionError('Prepared calibration differs from unchanged reference arithmetic')
    return {'stage': stage, 'steps': steps, **choices, 'old_pool_candidates': pool_count,
        'candidates': list(candidates.values()), 'labels_used': True, 'weights_updated': False,
        'search_exhaustive_joint_grid': False, 'prepared_reference_exact': True,
        'calibration_seconds': time.perf_counter() - started}


def calibrate(cache, prior_directory, output):
    manifest, previous = inputs(cache, prior_directory)
    spec = {**identity('task_encoder_calibration'),
        'cache_manifest_sha256': sha256(Path(cache) / 'manifest.json'),
        'prior_calibration_sha256': sha256(Path(prior_directory) / 'calibration.json')}
    with lock_directory(output, spec) as directory:
        final = directory / 'calibration.json'
        if final.exists():
            result = read_json(final)
            if result['identity'] != spec:
                raise ValueError('Saved task calibration identity mismatch')
            return result
        x, y, groups, sequences, sample = sample_data(cache, manifest, previous['sample'])
        layers = load_layers()
        prior_points = {(p['stage'], p['steps']): p for p in previous['points']}
        points = []
        for steps in protocol()['steps']:
            initial = None
            for stage in protocol()['stages']:
                receipt = directory / f'{stage}_t{steps}.json'
                if receipt.exists():
                    point = read_json(receipt)
                    if point['stage'] != stage or point['steps'] != steps or not point['prepared_reference_exact']:
                        raise ValueError('Task calibration receipt mismatch')
                else:
                    point = search_point(x, y, groups, sequences, sample, layers,
                                         prior_points[(stage, steps)], initial)
                    atomic_json(receipt, point)
                initial = tuple(Encoder(**e) for e in point['task']['encoders'])
                points.append(point)
                atomic_json(directory / 'progress.json', {'status': 'calibrating',
                    'completed_points': len(points), 'total_points': 6, 'training_started': False})
        result = {'version': spec['version'], 'status': 'complete', 'identity': spec,
            'protocol': protocol(), 'data_role': 'train_main', 'labels_used': True,
            'training_started': False, 'weights_updated': False, 'dev_used_for_selection': False,
            'reserved_sequences_read': 0, 'sample': sample, 'sampled_candidates': len(x),
            'sampled_targets': int(y.sum()), 'points': points}
        atomic_json(final, result)
        atomic_json(directory / 'progress.json', {'status': 'complete', 'training_started': False})
        return result
