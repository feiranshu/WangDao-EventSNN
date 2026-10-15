"""Direction 7 follow-up: independently calibrate ONLY component-size context.

The first diagnosis and all weights remain untouched. Main-only selection over
11 context gates crossed with every distinct routing score; Dev is a diagnostic
surface plus a replay of the already frozen Main selections. No optimizer.
"""
from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[2]
CONTEXT_GATES = [.5, .50001, .5001, .501, .502, .505, .508, .51, .52, .55, .6]


@lru_cache(maxsize=8)
def neighbors(height, width):
    return [tuple((y+dy)*width+x+dx for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                  if (dy or dx) and 0 <= y+dy < height and 0 <= x+dx < width)
            for y in range(height) for x in range(width)]


def component_sizes(mask):
    import numpy as np
    if mask.ndim != 2 or mask.dtype != np.bool_:
        raise ValueError('Expected 2D boolean mask')
    if mask.all():
        return np.full(mask.shape, mask.size, np.uint16)
    out = np.zeros(mask.size, np.uint16)
    pending = set(int(i) for i in np.flatnonzero(mask))
    adjacent = neighbors(*mask.shape)
    while pending:
        start = pending.pop()
        group, stack = [start], [start]
        while stack:
            for j in adjacent[stack.pop()]:
                if j in pending:
                    pending.remove(j)
                    stack.append(j)
                    group.append(j)
        out[group] = len(group)
    return out.reshape(mask.shape)


def component_columns(packet, windows, heatmaps):
    import numpy as np
    from wangdao.h2pro_proposal_diagnosis_v1.core import ss
    cols = np.empty((len(CONTEXT_GATES), len(packet)), np.uint8)
    checked = 0
    for w in range(len(heatmaps)):
        lo, hi = np.searchsorted(windows, [w, w+1])
        if lo == hi:
            continue
        masks = ss.dilate_cells(heatmaps[w][None] >= np.asarray(CONTEXT_GATES, np.float32)[:, None, None])
        memo = {}
        for j, mask in enumerate(masks):
            key = mask.tobytes()
            if key not in memo:
                sizes = component_sizes(mask)
                # Independent original BFS checks on fixed chronological samples.
                if w in (0, len(heatmaps)//2, len(heatmaps)-1):
                    np.testing.assert_array_equal(sizes, ss._component_sizes(mask))
                    checked += 1
                memo[key] = sizes
            value = memo[key][packet['y'][lo:hi]//8, packet['x'][lo:hi]//8]
            cols[j, lo:hi] = np.rint(255*np.minimum(value.astype(np.float64)/64., 1.)).astype(np.uint8)
    return cols, checked


def checks():
    import numpy as np
    from wangdao.h2pro_proposal_diagnosis_v1.core import ss, threshold_curve
    rng = np.random.default_rng(71403)
    count = 0
    for shape in ((1, 1), (3, 4), (34, 44)):
        for p in (0., .05, .4, .8, 1.):
            for _ in range(4):
                mask = rng.random(shape) < p
                np.testing.assert_array_equal(component_sizes(mask), ss._component_sizes(mask))
                count += 1
    score = rng.choice(np.array([.2, .5, .9], np.float32), 100)
    y, h = rng.integers(0, 2, (2, 100))
    prob = rng.random(100)
    curve = threshold_curve(score, y, prob, h)
    levels = curve['threshold'][1:][::-1]
    bins = np.searchsorted(levels, score)
    mask = (y == 1) & (prob >= .35)
    counts = np.bincount(bins[mask], minlength=len(levels))
    np.testing.assert_array_equal(np.r_[0, counts[::-1].cumsum()], curve['tp'])
    return {'status': 'passed', 'component_masks_exact': count,
            'histogram_frontier_exact': True, 'training_started': False}


def run_role(directory, source, store, role):
    import numpy as np
    from wangdao.h2pro_proposal_diagnosis_v1.common import load_bundle, save_bundle
    from wangdao.h2pro_proposal_diagnosis_v1.core import context_features, event_packet, refiner
    from wangdao.h2pro_proposal_diagnosis_v1.workflow import curve_summary
    from wangdao.h2pro_proposal_transfer_v1.pipeline import TransferPipeline
    prefix = 'main' if role == 'train_main' else 'dev'
    template, _ = load_bundle(source/(prefix+'_exact_curve'))
    levels = template['threshold'][1:][::-1]
    hist = {key: np.zeros((len(CONTEXT_GATES), len(levels)), np.int64)
            for key in ('tp', 'fp', 'hd_tp', 'hd_fp')}
    layers = TransferPipeline('ann_dense').refiner_layers
    costs = {'mac': 0, 'ac': 0}
    component_checks = 0
    for name in store.names(role):
        record = store.get(role, name)
        arrays, meta = record.arrays, record.metadata
        data, receipt = load_bundle(source/'cache'/role/Path(name).stem)
        costs['mac'] += receipt['costs']['s1']['proposal_mac']
        costs['ac'] += receipt['costs']['s1']['proposal_ac']
        stem = directory/'probabilities'/role/Path(name).stem
        if stem.with_suffix('.json').exists():
            saved, audit = load_bundle(stem)
        else:
            packet = event_packet(arrays)
            f = context_features(packet, data['windows'], data['base'], data['s1_heatmaps'], .51, meta['t0_ms'])
            columns, checked = component_columns(packet, data['windows'], data['s1_heatmaps'])
            np.testing.assert_array_equal(columns[CONTEXT_GATES.index(.51)], f[:, 30])
            probabilities = np.repeat(data['s1_prob_all'][None], len(CONTEXT_GATES), axis=0)
            evaluated = 0
            for j in range(len(CONTEXT_GATES)):
                changed = columns[j] != f[:, 30]
                subset = f[changed].copy()
                subset[:, 30] = columns[j, changed]
                if len(subset):
                    probabilities[j, changed] = refiner(subset, layers)
                evaluated += len(subset)
            saved = {'probabilities_all_events': probabilities}
            audit = {'role': role, 'sequence': name, 'source_cache_sha256': receipt['payload_sha256'],
                     'component_gate_grid': CONTEXT_GATES, 'original_component_masks_checked': checked,
                     'only_feature_column_changed': 30, 'neural_weights_updated': False,
                     'extra_refiner_mac_for_diagnosis': evaluated*4480}
            save_bundle(stem, saved, audit)
        if audit['source_cache_sha256'] != receipt['payload_sha256']:
            raise ValueError('Source cache changed')
        component_checks += audit['original_component_masks_checked']
        bins = np.searchsorted(levels, data['s1_scores'])
        np.testing.assert_array_equal(levels[bins], data['s1_scores'].astype(np.float64))
        positive = arrays['label'] == 1
        for j, probability in enumerate(saved['probabilities_all_events']):
            predicted = probability >= .35
            for key, mask in (('tp', predicted & positive), ('fp', predicted & ~positive)):
                value = np.bincount(bins[mask], minlength=len(levels))
                hist[key][j] += value
                if meta['high_density']:
                    hist['hd_'+key][j] += value
        print({'stage': 'component_calibration', 'role': role, 'sequence': name}, flush=True)
    source_report = json.loads((source/('selection.json' if role == 'train_main' else 'comparison.json')).read_text('utf-8'))
    baseline = source_report['baseline']
    curves, summaries = {}, []
    for j, cg in enumerate(CONTEXT_GATES):
        curve = {k: v.copy() for k, v in template.items()}
        for key, value in hist.items():
            curve[key] = np.r_[0, np.cumsum(value[j, ::-1], dtype=np.int64)]
            curves[str(j)+'_'+key] = curve[key]
            if cg == .51:
                np.testing.assert_array_equal(curve[key], template[key])
        curve['iou'] = np.divide(curve['tp'], curve['target_events']+curve['fp'],
                                  out=np.zeros(len(curve['tp'])), where=curve['target_events']+curve['fp'] > 0)
        summaries.append({'context_gate': cg, **curve_summary(curve, baseline, costs['mac'], costs['ac'])})
    save_bundle(directory/(prefix+'_curves'), curves, {'role': role, 'context_gates': CONTEXT_GATES,
                'summaries': summaries, 'component_masks_checked': component_checks,
                'default_context_full_frontier_exact': True,
                'diagnostic_not_selection': role == 'train_dev'})
    return summaries, baseline, costs


def evaluate(teacher, source, directory):
    import numpy as np
    from wangdao.h2pro_proposal_diagnosis_v1.common import (identity, open_store, lock_directory,
                                                          atomic_json, sha256, load_bundle, read_json)
    from wangdao.h2pro_proposal_diagnosis_v1.experiments import acceptance
    from wangdao.h2pro_proposal_transfer_v1.workflow import metrics
    from wangdao.h2pro_snn.diagnostics import error_counts
    old = read_json(source/'comparison.json')
    if old['identity'] != identity(teacher):
        raise ValueError('Source diagnosis identity mismatch')
    for name, expected in old['evidence_sha256'].items():
        if sha256(source/name) != expected:
            raise ValueError('Source diagnosis evidence changed: '+name)
    frozen = {'version': 'h2pro_proposal_component_v1', 'source_comparison_sha256': sha256(source/'comparison.json'),
              'source_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'context_gate_grid': CONTEXT_GATES, 'only_feature_column_changed': 30,
              'main_only_selection': True, 'training_started': False}
    with lock_directory(directory, frozen) as out:
        completed = out/'comparison.json'
        if completed.exists():
            report = read_json(completed)
            for name, expected in report['evidence_sha256'].items():
                if sha256(out/name) != expected:
                    raise ValueError('Completed component evidence changed')
            return report
        store = open_store(teacher)
        if (out/'selection.json').exists():
            selection = read_json(out/'selection.json')
        else:
            summaries, baseline, costs = run_role(out, source, store, 'train_main')
            best = max(summaries, key=lambda r: r['best_iou']['iou'])
            budget = [r for r in summaries if r['best_iou_with_sparse_ann_mac_budget']]
            best_budget = max(budget, key=lambda r: r['best_iou_with_sparse_ann_mac_budget']['iou'])
            options = [{'name': 's1_default', 'context_gate': .51, 'gate': .51},
                       {'name': 'main_best_component_only', 'context_gate': best['context_gate'], 'gate': .51},
                       {'name': 'main_best_joint', 'context_gate': best['context_gate'], 'gate': best['best_iou']['gate']},
                       {'name': 'main_best_under_mac_budget', 'context_gate': best_budget['context_gate'],
                        'gate': best_budget['best_iou_with_sparse_ann_mac_budget']['gate']}]
            selection = {'role': 'train_main', 'frozen_before_dev': True, 'summaries': summaries,
                         'baseline': baseline, 'costs': costs, 'selected': options,
                         'accuracy_feasible_combinations': sum(r['accuracy_feasible_gates'] for r in summaries)}
            atomic_json(out/'selection.json', selection)
        selection_hash = sha256(out/'selection.json')
        summaries, baseline, costs = run_role(out, source, store, 'train_dev')
        rows = {s['name']: [] for s in selection['selected']}
        for name in store.names('train_dev'):
            record = store.get('train_dev', name)
            data, _ = load_bundle(source/'cache/train_dev'/Path(name).stem)
            probabilities, _ = load_bundle(out/'probabilities/train_dev'/Path(name).stem)
            for s in selection['selected']:
                p = probabilities['probabilities_all_events'][CONTEXT_GATES.index(s['context_gate'])]
                candidate = data['s1_scores'] >= s['gate']
                rows[s['name']].append({'sequence': name, 'high_density': record.metadata['high_density'],
                    'counts': error_counts(record.arrays['label'], np.where(candidate, p, 0), candidate)})
        results = []
        for s in selection['selected']:
            r = rows[s['name']]
            total = metrics([v['counts'] for v in r])
            result = {'name': s['name'], 'specification': s, 'overall': total,
                      'high_density': metrics([v['counts'] for v in r if v['high_density']]),
                      'mac': costs['mac']+4480*total['candidate_events'], 'ac': costs['ac'],
                      'diagnostic_only': False, 'sequences': r}
            result['acceptance'] = acceptance(result, baseline)
            results.append(result)
        if selection_hash != sha256(out/'selection.json'):
            raise AssertionError('Selection changed during development analysis')
        report = {'status': 'complete', 'identity': frozen, 'selection_sha256': selection_hash,
                  'results': results, 'baseline': baseline, 'development_surface': summaries,
                  'development_surface_is_diagnostic_not_selection': True,
                  'development_accuracy_feasible_combinations': sum(r['accuracy_feasible_gates'] for r in summaries),
                  'training_started': False, 'reserved_sequences_read': 0,
                  'evidence_sha256': {p.relative_to(out).as_posix(): sha256(p) for p in out.rglob('*')
                                      if p.is_file() and p.name not in ('.lock', 'comparison.json')}}
        atomic_json(completed, report)
        return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('check', 'all'))
    p.add_argument('--worker', action='store_true')
    p.add_argument('--source', type=Path, default=ROOT/'results/runs/h2pro_proposal_diagnosis_v1')
    p.add_argument('--teacher-cache', type=Path, default=ROOT/'results/runs/h2pro_teacher_v1')
    p.add_argument('--run-directory', type=Path, default=ROOT/'results/runs/h2pro_proposal_component_v1')
    a = p.parse_args()
    if not a.worker:
        ann = ROOT.parents[1]/'1_H2_model_improvement/implementation/software'
        bootstrap = "import sys,runpy; ann,snn,script=sys.argv[1:4]; sys.argv=[script]+sys.argv[4:]; sys.path.insert(0,ann); import wangdao; wangdao.__path__.append(snn); runpy.run_path(script,run_name='__main__')"
        command = [sys.executable, '-B', '-c', bootstrap, str(ann), str(ROOT/'software/wangdao'), str(Path(__file__).absolute()),
                   a.action, '--worker', '--source', str(a.source.absolute()),
                   '--teacher-cache', str(a.teacher_cache.absolute()), '--run-directory', str(a.run_directory.absolute())]
        raise SystemExit(subprocess.run(command, cwd=ROOT,
                          env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')).returncode)
    import torch
    torch.set_num_threads(1)
    print(json.dumps(checks(), indent=2), flush=True)
    if a.action == 'all':
        report = evaluate(a.teacher_cache, a.source, a.run_directory)
        print(json.dumps({'status': report['status'], 'results': report['results'],
                          'accuracy_feasible_combinations': report['development_accuracy_feasible_combinations'],
                          'training_started': False}, indent=2), flush=True)


if __name__ == '__main__':
    main()
