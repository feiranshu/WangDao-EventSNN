"""Main-only calibration followed by complete development factorial verification."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

from .common import (GATES, PROTOCOL, identity, lock_directory, open_store, read_json,
                     atomic_json, load_bundle, save_bundle, sha256)
from .cache import aggregate_baseline, collect_role
from .core import map_heatmaps, recover_gate, threshold_curve
from .experiments import acceptance, run_specs, spec


def curve_summary(curve, baseline, proposal_mac, proposal_ac):
    tp, fp = curve['tp'], curve['fp']
    target = curve['target_events']
    precision = np.divide(tp, tp+fp, out=np.zeros(len(tp)), where=tp+fp > 0)
    recall = np.divide(tp, target, out=np.zeros(len(tp)), where=target > 0)
    hd_recall = np.divide(curve['hd_tp'], curve['hd_target_events'], out=np.zeros(len(tp)),
                          where=curve['hd_target_events'] > 0)
    mac = proposal_mac+4480*curve['candidate_events']
    accuracy = ((curve['iou'] >= baseline['overall']['iou']) & (precision >= baseline['overall']['precision'])
                & (recall >= baseline['overall']['recall']) & (hd_recall >= baseline['high_density']['recall']))
    valid = curve['threshold'] <= 1
    def best(mask):
        ids = np.flatnonzero(mask & valid)
        if not len(ids):
            return None
        i = int(ids[np.argmax(curve['iou'][ids])])
        return {'gate': float(curve['threshold'][i]), 'iou': float(curve['iou'][i]),
                'precision': float(precision[i]), 'recall': float(recall[i]),
                'hd_recall': float(hd_recall[i]), 'candidate_events': int(curve['candidate_events'][i]),
                'proposal_recall': float(curve['candidate_targets'][i]/target[i]),
                'mac': int(mac[i]), 'ac': proposal_ac}
    recovered = recover_gate(curve, baseline['overall']['candidate_targets'],
                             baseline['high_density']['candidate_targets'])
    i = int(np.flatnonzero(curve['threshold'] == recovered)[0])
    return {'distinct_routes': int(len(curve['threshold'])), 'recovery_gate': recovered,
            'recovery_candidates': int(curve['candidate_events'][i]),
            'recovery_mac': int(mac[i]), 'proposal_ac': proposal_ac,
            'best_iou': best(valid), 'best_iou_with_sparse_ann_mac_budget': best(mac < baseline['mac']),
            'accuracy_feasible_gates': int((accuracy & valid).sum()),
            'accuracy_and_mac_feasible_gates': int((accuracy & valid & (mac < baseline['mac'])).sum()),
            'best_accuracy_feasible': best(accuracy)}


def fit_affine(moments):
    n, sx, sy, sxx, sxy, syy = moments
    variance = sxx-sx*sx/n
    slope = (sxy-sx*sy/n)/variance if variance > 0 else 0.
    if not np.isfinite(slope) or slope <= 0:
        return {'slope': 1., 'intercept': 0., 'status': 'nonpositive_fit_identity_fallback',
                'role': 'train_main', 'event_weighted_cells': int(n)}
    intercept = (sy-slope*sx)/n
    return {'slope': float(slope), 'intercept': float(intercept), 'status': 'fitted',
            'role': 'train_main', 'event_weighted_cells': int(n),
            'logit_mse': float((syy+slope*slope*sxx+n*intercept*intercept-2*slope*sxy
                                -2*intercept*sy+2*slope*intercept*sx)/n),
            'neural_weights_updated': False, 'optimizer_used': False}


def select(directory, store):
    selection_path = directory/'selection.json'
    if selection_path.exists():
        saved = read_json(selection_path)
        for name, digest in saved['evidence_sha256'].items():
            if sha256(directory/name) != digest:
                raise ValueError('Main selection evidence changed')
        return saved
    parts, receipts, moments = collect_role(directory, store, 'train_main')
    baseline = aggregate_baseline(receipts)
    curve = threshold_curve(parts['scores'], parts['labels'], parts['probabilities'], parts['hd'])
    prop_mac = sum(r['costs']['s1']['proposal_mac'] for r in receipts)
    prop_ac = sum(r['costs']['s1']['proposal_ac'] for r in receipts)
    summary = curve_summary(curve, baseline, prop_mac, prop_ac)
    calibration = fit_affine(moments)
    save_bundle(directory/'main_exact_curve', curve, {'role': 'train_main', 'summary': summary})
    del parts, curve
    gate = summary['recovery_gate']
    gates = sorted(set(GATES+[gate, summary['best_iou']['gate']]))
    specifications = [spec('coupled_%02d' % i, gate=g, context_gate=g) for i, g in enumerate(gates)]
    mapped_gate = float(map_heatmaps(np.array([gate], np.float32), calibration)[0])
    specifications += [spec('cal_context_recovered', gate=gate, context='calibrated'),
                       spec('cal_joint_default', route='calibrated', context='calibrated'),
                       spec('cal_joint_recovered', route='calibrated', gate=mapped_gate,
                            context='calibrated', context_gate=mapped_gate)]
    # Commit all thresholds/fit before ANY development inference in this workflow.
    atomic_json(directory/'main_plan.json', {'role': 'train_main', 'baseline': baseline,
                'fixed_context_curve': summary, 'calibration': calibration, 'specifications': specifications})
    results = run_specs(directory, store, 'train_main', specifications, calibration)
    for r in results:
        r['acceptance'] = acceptance(r, baseline)
    best = max(results, key=lambda r: (r['overall']['iou'], -r['mac']))
    budget = [r for r in results if r['mac'] < baseline['mac']]
    best_budget = max(budget, key=lambda r: (r['overall']['iou'], -r['mac'])) if budget else None
    selected = [best['specification']]
    if best_budget:
        selected.append(best_budget['specification'])
    selected += [r['specification'] for r in results if r['acceptance']['passes']]
    selected += [r['specification'] for r in results if r['name'].startswith('cal_')]
    unique = {s['name']: s for s in selected}
    # main-only selections; .51 and all-route endpoints and recall-matched gate
    # are controls defined in the protocol, never tuned on development results.
    specs = [spec('ann_baseline', route='ann', context='ann'), spec('s1_default'),
             spec('recovered_fixed_context', gate=gate),
             spec('recovered_coupled', gate=gate, context_gate=gate),
             spec('all_route_coupled', gate=0., context_gate=0.)]
    specs += list(unique.values())
    specs += [spec('candidate_only_default', context='ann'),
              spec('candidate_only_recovered', gate=gate, context='ann'),
              spec('context_only', route='ann'),
              spec('all_route_ann_context', gate=0., context='ann')]
    specs += [spec('swap_'+g, route='ann', context='ann', swap=g)
              for g in ('motion', 'current', 'history', 'component')]
    evidence = {p.relative_to(directory).as_posix(): sha256(p)
                for parent in (directory/'cache/train_main', directory/'points/train_main')
                for p in parent.rglob('*') if p.is_file()}
    evidence.update({name: sha256(directory/name) for name in ('main_plan.json', 'main_exact_curve.npz', 'main_exact_curve.json')})
    selected_report = {'selection_role': 'train_main', 'selection_frozen_before_dev': True,
                       'baseline': baseline, 'fixed_context_curve': summary, 'calibration': calibration,
                       'main_results': results, 'best_main_iou': best['name'],
                       'best_main_iou_under_mac_budget': best_budget['name'] if best_budget else None,
                       'development_specifications': specs, 'evidence_sha256': evidence,
                       'training_started': False}
    atomic_json(selection_path, selected_report)
    print({'stage': 'selection_frozen', 'recovery_gate': gate, 'calibration': calibration,
           'best_main_iou': best['name'], 'main_iou': best['overall']['iou'],
           'development_configurations': len(specs)}, flush=True)
    return selected_report


def write_table(path, results):
    buffer = io.StringIO(newline='')
    keys = ('name', 'diagnostic_only', 'iou', 'precision', 'recall', 'f1', 'hd_recall',
            'proposal_recall', 'candidates', 'mac', 'ac', 'passes')
    writer = csv.DictWriter(buffer, fieldnames=keys)
    writer.writeheader()
    for r in results:
        writer.writerow({'name': r['name'], 'diagnostic_only': r['diagnostic_only'],
                         **{k: r['overall'][k] for k in ('iou', 'precision', 'recall', 'f1')},
                         'hd_recall': r['high_density']['recall'],
                         'proposal_recall': r['overall']['proposal_target_recall'],
                         'candidates': r['overall']['candidate_events'], 'mac': r['mac'], 'ac': r['ac'],
                         'passes': r['acceptance']['passes']})
    path.write_text(buffer.getvalue(), encoding='utf-8', newline='')


def evaluate(teacher, output):
    frozen = identity(teacher)
    with lock_directory(output, frozen) as directory:
        completed = directory/'comparison.json'
        if completed.exists():
            report = read_json(completed)
            for name, digest in report['evidence_sha256'].items():
                p = (directory/name).resolve()
                if directory not in p.parents or sha256(p) != digest:
                    raise ValueError('Completed evidence changed')
            return report
        store = open_store(teacher)
        selection = select(directory, store)
        selection_digest = sha256(directory/'selection.json')
        parts, receipts, _ = collect_role(directory, store, 'train_dev')
        baseline = aggregate_baseline(receipts)
        curve = threshold_curve(parts['scores'], parts['labels'], parts['probabilities'], parts['hd'])
        summary = curve_summary(curve, baseline, sum(r['costs']['s1']['proposal_mac'] for r in receipts),
                                sum(r['costs']['s1']['proposal_ac'] for r in receipts))
        save_bundle(directory/'dev_exact_curve', curve,
                    {'role': 'train_dev', 'diagnostic_not_selection': True, 'summary': summary})
        del parts, curve
        results = run_specs(directory, store, 'train_dev', selection['development_specifications'], selection['calibration'])
        for r in results:
            r['acceptance'] = acceptance(r, baseline)
            r['delta_vs_ann_pp'] = {k: 100*(r['overall'][k]-baseline['overall'][k])
                                    for k in ('iou', 'precision', 'recall', 'f1', 'proposal_target_recall')}
        if sha256(directory/'selection.json') != selection_digest:
            raise AssertionError('Development analysis modified main selection')
        passing = [r['name'] for r in results if r['acceptance']['passes']]
        write_table(directory/'comparison.csv', results)
        # Independent causal-pipeline replay for deployable controls is performed
        # by verify.py and recorded separately; this report retains sweep costs.
        evidence = {p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob('*')
                    if p.is_file() and p.name not in ('.lock', 'comparison.json', 'progress.json')}
        report = {'status': 'complete', 'identity': frozen, 'protocol': PROTOCOL,
                  'selection_sha256': selection_digest, 'baseline': baseline,
                  'fixed_context_dev_exact_curve': summary, 'results': results,
                  'passing_deployable_candidates': passing,
                  'verdict': 'development_candidate_found' if passing else 'tested_frozen_weight_family_not_feasible',
                  'scope': PROTOCOL['scope_limit'], 'training_started': False, 'weights_updated': False,
                  'reserved_sequences_read': 0, 'evidence_sha256': evidence}
        atomic_json(completed, report)
        return report
