"""Complete frozen-model comparisons with per-sequence resume and auditable costs."""
from __future__ import annotations

import csv
from dataclasses import asdict
import io
import json
import os
from pathlib import Path
import time

import numpy as np

from ..h2pro_snn.diagnostics import aggregate_counts, error_counts
from .common import (MODES, PROTOCOL, atomic_json, identity, load_teacher, lock_directory,
                     read_json, sha256, teacher_manifest)


def metrics(counts):
    m = aggregate_counts(counts)
    denominator = 2 * m['tp'] + m['fp'] + m['fn']
    m['f1'] = 2 * m['tp'] / denominator if denominator else None
    return m


def context_delta(actual, baseline):
    old, new = baseline['candidate'], actual['candidate']
    common = old & new
    old_features = baseline['features'][common[old]]
    new_features = actual['features'][common[new]]
    base_columns = [0, 1] + list(range(3, 28))
    if not np.array_equal(old_features[:, base_columns], new_features[:, base_columns]):
        raise AssertionError('Changing Proposal modified independent base event features')
    return {'common_candidates': int(common.sum()),
            'motion_feature_changed': int(np.count_nonzero(old_features[:, 2] != new_features[:, 2])),
            'current_context_rows_changed': int(np.any(old_features[:, 28:32] != new_features[:, 28:32], axis=1).sum()),
            'history_context_rows_changed': int(np.any(old_features[:, 32:36] != new_features[:, 32:36], axis=1).sum()),
            'independent_base_features_exact': True,
            'heatmap_cells_changed': int(np.count_nonzero(actual['heatmaps'] != baseline['heatmaps']))}


def transition(labels, actual, baseline):
    old, new = baseline['candidate'], actual['candidate']
    positive = labels == 1
    old_pred, new_pred = baseline['probabilities'] >= .35, actual['probabilities'] >= .35
    return {'candidates_added': int((new & ~old).sum()), 'candidates_dropped': int((old & ~new).sum()),
            'target_candidates_added': int((new & ~old & positive).sum()),
            'target_candidates_dropped': int((old & ~new & positive).sum()),
            'teacher_tp_lost': int((positive & old_pred & ~new_pred).sum()),
            'teacher_fn_recovered': int((positive & ~old_pred & new_pred).sum()),
            'teacher_fp_removed': int((~positive & old_pred & ~new_pred).sum()),
            'teacher_tn_became_fp': int((~positive & ~old_pred & new_pred).sum())}


def write_npz(path, arrays):
    temporary = Path(path).with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def point(directory, row, arrays, metadata, mode, baseline, inference=None):
    target = Path(directory) / mode
    target.mkdir(exist_ok=True)
    receipt = target / (Path(row['sequence']).stem + '.json')
    payload = receipt.with_suffix('.npz')
    if receipt.exists():
        saved = read_json(receipt)
        if (saved['mode'] != mode or saved['sequence'] != row['sequence']
                or saved['teacher_cache_sha256'] != row['cache_sha256']
                or saved['payload_sha256'] != sha256(payload)):
            raise ValueError('Completed sequence integrity mismatch')
        with np.load(payload, allow_pickle=False) as z:
            output = {k: z[k].copy() for k in z.files}
        if saved['counts'] != error_counts(arrays['label'], output['probabilities'], output['candidate'], .35):
            raise ValueError('Completed metrics differ from saved predictions')
        return saved, output
    started = time.perf_counter()
    if inference is None:
        from .pipeline import infer
        inference = infer
    output, executor = inference(arrays, metadata, mode)
    if mode.startswith('ann_'):
        if not np.array_equal(output['candidate'], arrays['candidate']) or not np.array_equal(output['probabilities'], arrays['teacher']):
            raise AssertionError('Fresh ANN does not exactly replay existing frozen teacher')
    if baseline is None:
        baseline = output
    if mode == 'ann_sparse_proposal' and any(not np.array_equal(output[k], baseline[k]) for k in output):
        raise AssertionError('Sparse ANN changed features, routing, history or scores')
    saved = {'sequence': row['sequence'], 'mode': mode, 'high_density': row['high_density'],
             'teacher_cache_sha256': row['cache_sha256'], 'source_sha256': row['source_sha256'],
             'closed_windows': metadata['closed_windows'], 'excluded_trailing_events': metadata['trailing_events'],
             'counts': error_counts(arrays['label'], output['probabilities'], output['candidate'], .35),
             'cost': asdict(executor.cost), 'saturations': executor.saturations,
             'context_delta': context_delta(output, baseline),
             'transition': transition(arrays['label'], output, baseline),
             'all_weights_frozen': True, 'ann_teacher_exact': mode.startswith('ann_'),
             'forward_seconds_including_reference_and_diagnostics': time.perf_counter() - started}
    write_npz(payload, output)
    saved['payload_sha256'] = sha256(payload)
    atomic_json(receipt, saved)
    print(json.dumps({'stage': 'proposal_transfer', 'mode': mode, 'sequence': row['sequence'],
                      'events': saved['counts']['events'], 'candidates': saved['counts']['candidate_events'],
                      'seconds': saved['forward_seconds_including_reference_and_diagnostics']}), flush=True)
    return saved, output


def summarize(mode, rows):
    cost = {k: sum(r['cost'][k] for r in rows) for k in rows[0]['cost']}
    executed_mac = cost['proposal_mac'] + cost['refiner_mac']
    return {'mode': mode, 'role': 'train_dev', 'threshold': .35, 'proposal_threshold': .51,
            'overall': metrics([r['counts'] for r in rows]),
            'high_density': metrics([r['counts'] for r in rows if r['high_density']]),
            'ordinary': metrics([r['counts'] for r in rows if not r['high_density']]),
            'cost': cost, 'executed_neural_mac': executed_mac, 'executed_neural_ac': cost['proposal_ac'],
            'executed_neural_mac_plus_ac': executed_mac + cost['proposal_ac'],
            'ideal_snn_ac_if_sparse': 24 * cost['hidden1_spikes'] + 16 * cost['hidden2_spikes'],
            'transition': {k: sum(r['transition'][k] for r in rows) for k in rows[0]['transition']},
            'context_delta': {k: (all(r['context_delta'][k] for r in rows) if k == 'independent_base_features_exact'
                                 else sum(r['context_delta'][k] for r in rows)) for k in rows[0]['context_delta']},
            'saturations': {k: [sum(r['saturations'][k][i] for r in rows)
                               for i in range(len(rows[0]['saturations'][k]))] for k in rows[0]['saturations']},
            'forward_seconds_including_reference_and_diagnostics': sum(r['forward_seconds_including_reference_and_diagnostics'] for r in rows),
            'sequences': rows, 'end_to_end_cost_complete': False, 'energy_measured': False}


def evaluate(teacher_cache, run_directory):
    manifest = teacher_manifest(teacher_cache)
    frozen = identity(Path(teacher_cache) / 'manifest.json')
    with lock_directory(run_directory, frozen) as directory:
        completed = directory / 'comparison.json'
        if completed.exists():
            report = read_json(completed)
            if report['identity'] != frozen or report['status'] != 'complete':
                raise ValueError('Comparison identity mismatch')
            for name, expected in report['evidence_sha256'].items():
                p = (directory / name).resolve()
                if directory.resolve() not in p.parents or sha256(p) != expected:
                    raise ValueError('Completed evidence integrity mismatch')
            return report
        all_rows = {mode: [] for mode in MODES}
        for row in manifest['roles']['train_dev']:
            arrays, meta = load_teacher(teacher_cache, row, manifest)
            baseline = None
            for mode in MODES:
                saved, output = point(directory, row, arrays, meta, mode, baseline)
                all_rows[mode].append(saved)
                if mode == 'ann_dense':
                    baseline = output
                atomic_json(directory / 'progress.json', {'status': 'running', 'sequence': row['sequence'],
                            'mode': mode, 'completed_points': sum(len(v) for v in all_rows.values()),
                            'total_points': 4 * len(manifest['roles']['train_dev']), 'training_started': False})
        results = [summarize(mode, all_rows[mode]) for mode in MODES]
        dense, sparse = results[:2]
        for result in results:
            result['delta_vs_ann_pp'] = {k: 100 * (result['overall'][k] - dense['overall'][k])
                                        for k in ('iou', 'precision', 'recall', 'f1', 'proposal_target_recall')}
            result['hd_recall_delta_vs_ann_pp'] = 100 * (result['high_density']['recall'] - dense['high_density']['recall'])
            for label, base in (('dense_ann', dense), ('sparse_ann_proposal', sparse)):
                result['mac_reduction_vs_' + label] = 1 - result['executed_neural_mac'] / base['executed_neural_mac']
                result['mac_plus_ac_ratio_vs_' + label] = result['executed_neural_mac_plus_ac'] / base['executed_neural_mac_plus_ac']
        table = io.StringIO(newline='')
        keys = ('mode', 'iou', 'precision', 'recall', 'f1', 'hd_recall', 'proposal_recall',
                'candidates', 'mac', 'ac', 'mac_plus_ac')
        writer = csv.DictWriter(table, fieldnames=keys)
        writer.writeheader()
        for r in results:
            writer.writerow({'mode': r['mode'], **{k: r['overall'][k] for k in ('iou', 'precision', 'recall', 'f1')},
                             'hd_recall': r['high_density']['recall'], 'proposal_recall': r['overall']['proposal_target_recall'],
                             'candidates': r['overall']['candidate_events'], 'mac': r['executed_neural_mac'],
                             'ac': r['executed_neural_ac'], 'mac_plus_ac': r['executed_neural_mac_plus_ac']})
        (directory / 'comparison.csv').write_text(table.getvalue(), encoding='utf-8', newline='')
        evidence = {p.relative_to(directory).as_posix(): sha256(p) for mode in MODES
                    for p in sorted((directory / mode).iterdir()) if p.suffix in ('.json', '.npz')}
        evidence['comparison.csv'] = sha256(directory / 'comparison.csv')
        report = {'status': 'complete', 'identity': frozen, 'protocol': PROTOCOL, 'results': results,
                  'training_started': False, 'weights_updated': False, 'reserved_sequences_read': 0,
                  'comparisons': len(results), 'sequence_comparisons': sum(len(v) for v in all_rows.values()),
                  'event_coverage': dense['overall']['events'], 'evidence_sha256': evidence,
                  'limits': ['legacy pretrained Proposal transfer, not original H2-pro weights preserved throughout',
                            'Refiner remains ANN; 100ms closure remains; no hardware energy or latency measurement',
                            'dense S0P MAC is executed MAC; its hypothetical spike AC is reported separately',
                            'S1 actual sparse synaptic count excludes dense golden verification overhead',
                            'state/scaling/feature/routing/readout/memory work is not fully included in MAC+AC']}
        atomic_json(directory / 'progress.json', {'status': 'complete', 'comparisons': len(results),
                                                'sequence_comparisons': report['sequence_comparisons'],
                                                'training_started': False})
        atomic_json(completed, report)
        return report
