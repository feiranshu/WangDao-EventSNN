"""Two-stage, resumable inference. Selection is committed before Dev is loaded."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from ..h2pro_direct_v1.core import forward, load_layers, probabilities, validate_layers
from ..h2pro_distill_v1.common import (Config, DEFAULT_CACHE, DEFAULT_RUN,
    atomic_json, lock_directory, output_path, read_json, run_identity, sha256)
from ..h2pro_distill_v1.data import manifest
from ..h2pro_distill_v1.evaluation import score
from ..h2pro_distill_v1.model import R2T8
from ..h2pro_distill_v1.runtime import setup_device, memory
from .thresholds import CONTROL, GROUPS, POLICIES, choose, curve, distributions, measure, transitions

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]
DEFAULT_OUTPUT = ROOT/'results/runs/h2pro_fixed_diagnosis_v1'
EPOCHS = (36, 88, 95, 100)
PROTOCOL = {
    'version': 'h2pro_fixed_diagnosis_v1', 'epochs': list(EPOCHS),
    'weights': 'immutable epoch int8 exports; original scales, structure, features and Proposal',
    'optimizer_steps': 0, 'backward_calls': 0, 'steps': 8,
    'control_threshold': CONTROL, 'comparison': 'FP32 probability >= FP32 threshold in (0,1)',
    'selection_role': 'train_main', 'validation_role': 'train_dev',
    'policies': {'main_iou': 'maximize full-event Main IoU',
                 'main_recall_matched': 'maximize full-event Main IoU with overall AND HD recall >= frozen ANN at 0.35'},
    'tie_break': 'nearest 0.35, then higher threshold',
    'threshold_scope': 'ONE global scalar per model/policy; no per-sequence or density inference rules',
    'search': 'all distinct candidate FP32 scores in (0,1), both interior endpoints, and 0.35',
    'preserve_proposal_rejects': True, 'cost_changes_with_output_threshold': False,
    'dev_curve': 'descriptive only; never used to choose or revise thresholds',
    'reserved_splits_read': False,
    'limitations': ['epochs already chosen using historical Dev; this is development validation, not a fresh test',
                    'no independent-sequence uncertainty claim; Main is previously used training data',
                    'neural operation counts are not measured energy or sparse GPU execution'],
}


def utc():
    return datetime.now(timezone.utc).isoformat()


def sources():
    paths = sorted(PACKAGE.glob('*.py')) + [ROOT/'software/scripts/diagnose_h2pro_fixed.py']
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}


def tree_hashes(directory):
    return {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(directory.rglob('*')) if p.is_file()}


def export_layers(path):
    original = load_layers()
    with np.load(path, allow_pickle=False) as z:
        if (z['dimensions'].tolist() != [36, 64, 32, 4] or str(z['stage']) != 'r2'
                or int(z['steps']) != 8 or float(z['threshold']) != .35
                or str(z['variant']) != 'C1-proposal-motion'):
            raise ValueError('Not an unchanged r2/T8 architecture/export')
        layers = tuple((z[f'layer{i}_weight'].copy(), z[f'layer{i}_scale'].copy()) for i in range(3))
    validate_layers(layers)
    for (_, s), (_, fixed_scale) in zip(layers, original):
        np.testing.assert_array_equal(s, fixed_scale)
    return layers


class RoleData:
    """Load exactly one role; Dev arrays are not opened until selection is sealed."""
    def __init__(self, cache, rows, role, device):
        started = time.perf_counter()
        self.role, self.device = role, device
        self.sequences = {role: []}
        xs, ys, ts, hds = [], [], [], []
        offset = 0
        for row in rows:
            if row['role'] != role:
                raise ValueError('Mixed calibration and validation roles')
            with np.load(cache/row['cache_file'], allow_pickle=False) as z:
                x, y, c, teacher = (z[k].copy() for k in ('features','labels','candidate','teacher'))
            if (c.dtype != np.bool_ or c.shape != (row['events'],) or y.shape != c.shape
                    or not np.isin(y, [0, 1]).all() or int(y.sum()) != row['target_events']
                    or int(c.sum()) != row['candidate_events'] or int(y[c].sum()) != row['candidate_targets']
                    or x.dtype != np.uint8 or x.shape != (int(c.sum()), 36)
                    or teacher.dtype != np.float32 or teacher.shape != (int(c.sum()), 4)
                    or not np.isfinite(teacher).all()):
                raise ValueError('Invalid original candidate cache')
            n = len(x)
            self.sequences[role].append({'row': row, 'labels': y, 'candidate': c, 'lo': offset, 'hi': offset+n})
            offset += n
            xs.append(x); ys.append(y[c]); ts.append(teacher)
            hds.append(np.full(n, row['high_density'], bool))
        self.features = torch.from_numpy(np.concatenate(xs)).to(device)
        self.labels, self.teacher, self.hd = np.concatenate(ys), np.concatenate(ts), np.concatenate(hds)
        self.targets = {g: sum(s['row']['target_events'] for s in self.sequences[role]
            if g == 'overall' or s['row']['high_density'] == (g == 'high_density')) for g in GROUPS}
        self.stats = {'role': role, 'sequences': len(rows), 'candidate_events': len(self.labels),
                      'full_events': sum(r['events'] for r in rows), 'load_seconds': time.perf_counter()-started}


def save_arrays(path, **arrays):
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


@torch.inference_mode()
def infer(data, layers, name, directory, batch_size):
    path = directory/f'{data.role}_{name}_predictions.npz'
    receipt = path.with_suffix('.json')
    if receipt.exists():
        record = read_json(receipt)
        if sha256(path) != record['sha256']:
            raise ValueError('Prediction checksum mismatch; refusing to reuse')
        with np.load(path, allow_pickle=False) as z:
            return z['raw'], z['spikes'], record
    started = time.perf_counter()
    model = None
    if name == 'ann':
        raw, spikes = data.teacher.copy(), np.zeros((len(data.labels), 2), np.int64)
    else:
        model = R2T8(layers).to(data.device).eval().requires_grad_(False)
        for actual, expected in zip(model.export_layers(), layers):
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
        raw, spikes = np.empty((len(data.labels), 4), np.float32), np.empty((len(data.labels), 2), np.int64)
        for lo in range(0, len(data.labels), batch_size):
            hi = min(lo+batch_size, len(data.labels))
            r, s = model(data.features[lo:hi])
            raw[lo:hi], spikes[lo:hi] = r.cpu().numpy(), s.cpu().numpy()
    verified = 0
    for seq in data.sequences[data.role]:
        lo, hi = seq['lo'], seq['hi']
        ids = np.unique(np.linspace(lo, hi-1, min(64, hi-lo), dtype=np.int64)) if hi > lo else np.empty(0,np.int64)
        x = data.features[torch.as_tensor(ids, device=data.device)].cpu().numpy()
        reference, costs = forward(x, layers, 'ann' if name == 'ann' else 'r2', 8,
                                   'count_reference' if name == 'ann' else 'spike_add')
        np.testing.assert_array_equal(reference, raw[ids])
        if name != 'ann' and costs.executed_synaptic_ac != int((spikes[ids]*[32,4]).sum()):
            raise AssertionError('Explicit spikes disagree with count-reference AC')
        verified += len(ids)
    if model is not None:
        for actual, expected in zip(model.export_layers(), layers):
            for a, b in zip(actual, expected):
                np.testing.assert_array_equal(a, b)
        if any(p.grad is not None or p.requires_grad for p in model.parameters()):
            raise AssertionError('Frozen inference acquired gradients')
    save_arrays(path, raw=raw, spikes=spikes)
    record = {'sha256': sha256(path), 'created_at_utc': utc(), 'role': data.role, 'model': name,
              'candidates': len(raw), 'explicit_verified_candidates': verified,
              'seconds': time.perf_counter()-started, 'weights_unchanged': True,
              'optimizer_steps': 0, 'backward_calls': 0, 'memory': memory(data.device)}
    atomic_json(receipt, record)
    print(json.dumps({'stage': 'inference', **record}), flush=True)
    return raw, spikes, record


def analyse(data, raw, spikes, name, directory, selections=None):
    role = data.role
    p, teacher_p = probabilities(raw), probabilities(data.teacher)
    controls = measure(data.sequences[role], p, CONTROL)
    reference = score(data, raw, spikes, role)
    for group in GROUPS:
        if controls[group] != reference[group]:
            raise AssertionError('0.35 scorer differs from original full-event evaluation')
    costs = reference['cost']
    if name == 'ann':
        costs = {'mac_sparse_proposal': costs['frozen_ann_sparse_mac'],
                 'mac_dense_proposal': costs['frozen_ann_dense_mac'], 'ac': 0}
    curves = curve(p, data.labels, data.hd, data.targets)
    save_arrays(directory/f'{role}_{name}_curve.npz', **curves)
    points = {'control_035': controls}
    if selections:
        for policy, choice in selections.items():
            if choice['status'] == 'selected':
                point = measure(data.sequences[role], p, choice['threshold'])
                point['transitions'] = transitions(p, teacher_p, data.labels, data.hd, choice['threshold'])
                points[policy] = point
    for point in points.values():
        t = np.float32(point['threshold'])
        # Selected Main knots need not be Dev knots. Direct search evaluates them
        # exactly; no nearest-knot interpolation or threshold retuning occurs.
        for group, mask in zip(GROUPS, (np.ones(len(p), bool), data.hd, ~data.hd)):
            if (int((mask & (data.labels == 1) & (p >= t)).sum()) != point[group]['tp']
                    or int((mask & (data.labels == 0) & (p >= t)).sum()) != point[group]['fp']):
                raise AssertionError('Full-event and candidate decisions disagree')
    return {'operating_points': points, 'cost': costs,
            'score_distribution': distributions(p, data.labels, data.hd)}, curves


def checked_bundle(directory, summary_name, checksum_name):
    summary_path = directory/summary_name
    if not summary_path.exists():
        return None
    checksum_path = directory/checksum_name
    if not checksum_path.exists():
        return None  # An uncommitted partial stage may be rebuilt.
    hashes = read_json(checksum_path)
    if any(sha256(directory/name) != digest for name, digest in hashes.items()):
        raise ValueError('Sealed stage was changed; refusing to reuse')
    return read_json(summary_path)


def seal(directory, names, checksum_name):
    atomic_json(directory/checksum_name, {n: sha256(directory/n) for n in sorted(names)})


def run(cache=DEFAULT_CACHE, source=DEFAULT_RUN, output=DEFAULT_OUTPUT, action='all', device_name='cuda', batch_size=8192):
    if action not in ('main', 'dev', 'all') or batch_size < 1:
        raise ValueError('Invalid action or inference batch size')
    cache, source, output = map(output_path, (cache, source, output))
    if any(p == output or p in output.parents or output in p.parents for p in (cache, source)):
        raise ValueError('Diagnosis needs an independent output directory')
    training = read_json(source/'training.json')
    identity = training['identity']
    if run_identity(cache, Config(**identity['config']), identity['engine']) != identity:
        raise ValueError('Historical training identity no longer matches source/teacher/cache')
    features = manifest(cache)  # Hash verification only; Dev labels not loaded here.
    preserved = tree_hashes(source)
    if len(training['epochs']) != 100:
        raise ValueError('This protocol requires the completed 100-epoch experiment')
    layers = {'ann': load_layers(), **{f'epoch_{e:03}': export_layers(source/f'epochs/epoch_{e:03}_int8.npz') for e in EPOCHS}}
    device = setup_device(device_name)
    own_identity = {'protocol': PROTOCOL, 'sources': sources(), 'training_identity': identity,
                    'training_file_sha256': sha256(source/'training.json'),
                    'source_run_files_sha256': preserved, 'device': str(device), 'batch_size': batch_size,
                    'torch': str(torch.__version__), 'numpy': str(np.__version__)}
    with lock_directory(output, own_identity):
        selected = checked_bundle(output, 'selection.json', 'main_checksums.json')
        if action in ('main', 'all') and selected is None:
            data = RoleData(cache, features['roles']['train_main'], 'train_main', device)
            summary = {'role': 'train_main', 'models': {}, 'data': data.stats, 'protocol': PROTOCOL}
            choices = {}
            for name, weights in layers.items():
                raw, spikes, receipt = infer(data, weights, name, output, batch_size)
                result, curves = analyse(data, raw, spikes, name, output)
                if name == 'ann':
                    ann_counts = result['operating_points']['control_035']
                else:
                    choices[name] = {policy: choose(curves, ann_counts, policy, role='train_main') for policy in POLICIES}
                    p = probabilities(raw)
                    for policy, choice in choices[name].items():
                        if choice['status'] == 'selected':
                            point = measure(data.sequences['train_main'], p, choice['threshold'])
                            for group in GROUPS:
                                for metric in ('iou', 'precision', 'recall'):
                                    if point[group][metric] != choice['main'][group][metric]:
                                        raise AssertionError('Sorted curve optimum differs from direct rescore')
                            result['operating_points'][policy] = point
                result['inference'] = receipt
                summary['models'][name] = result
            atomic_json(output/'main_summary.json', summary)
            selected = {'created_at_utc': utc(), 'protocol': PROTOCOL, 'choices': choices,
                        'main_summary_sha256': sha256(output/'main_summary.json'),
                        'selection_role': 'train_main', 'dev_arrays_loaded': False}
            atomic_json(output/'selection.json', selected)
            seal(output, ['main_summary.json', 'selection.json'] + [p.name for p in output.glob('train_main_*')],
                 'main_checksums.json')
            print(json.dumps({'stage': 'main_selection_committed', 'selection': choices}), flush=True)
            del data, raw, spikes, curves
        if action in ('dev', 'all'):
            if selected is None:
                raise ValueError('Run Main and seal selection before loading Dev')
            if checked_bundle(output, 'dev_summary.json', 'dev_checksums.json') is None:
                selection_hash = sha256(output/'selection.json')
                validation_started = utc()
                data = RoleData(cache, features['roles']['train_dev'], 'train_dev', device)
                summary = {'role': 'train_dev', 'models': {}, 'data': data.stats,
                           'selection_sha256': selection_hash, 'validation_started_at_utc': validation_started,
                           'thresholds_reselected_on_dev': False, 'optimizer_steps': 0, 'backward_calls': 0}
                for name, weights in layers.items():
                    raw, spikes, receipt = infer(data, weights, name, output, batch_size)
                    result, _ = analyse(data, raw, spikes, name, output, selected['choices'].get(name))
                    expected = training['frozen_ann'] if name == 'ann' else read_json(
                        source/f"epochs/{name}.json")['evaluation']
                    for group in GROUPS:
                        if result['operating_points']['control_035'][group] != expected[group]:
                            raise AssertionError('Historical Dev metrics failed exact replay: '+name+'/'+group)
                    if result['cost'] != expected['cost']:
                        raise AssertionError('Historical Dev cost failed exact replay: '+name)
                    result['historical_035_metrics_and_cost_exact'] = True
                    result['inference'] = receipt
                    summary['models'][name] = result
                if sha256(output/'selection.json') != selection_hash:
                    raise AssertionError('Main thresholds changed during Dev validation')
                atomic_json(output/'dev_summary.json', summary)
                seal(output, ['dev_summary.json'] + [p.name for p in output.glob('train_dev_*')], 'dev_checksums.json')
        if tree_hashes(source) != preserved or run_identity(cache, Config(**identity['config']), identity['engine']) != identity:
            raise AssertionError('Historical training/teacher/source data changed')
        receipt = {'status': 'complete' if (output/'dev_checksums.json').exists() else 'main_complete',
                   'completed_at_utc': utc(), 'source_files_unchanged': len(preserved),
                   'training_sources_unchanged': True, 'teacher_identity_unchanged': True,
                   'optimizer_steps': 0, 'backward_calls': 0, 'reserved_splits_read': False,
                   'selection_sha256': sha256(output/'selection.json'), 'memory': memory(device)}
        # Do not rewrite an already completed receipt when checking resume.
        if not (output/'complete.json').exists() or read_json(output/'complete.json')['status'] != receipt['status']:
            atomic_json(output/'complete.json', receipt)
        return read_json(output/'complete.json')
