"""Frozen-selection evaluation using the unchanged explicit IF implementation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from ..h2pro_snn.foundation import read_json, sha256, verify_baseline
from ..h2pro_direct_v1.workflow import cache_manifest
from ..h2pro_calibration_v1.core import Encoder, forward, load_layers
from ..h2pro_calibration_v1.workflow import checks as original_checks, evaluate_point, write_table
from .calibration import PreparedForward
from .common import atomic_json, identity, lock_directory, protocol, source_hashes


def checks():
    previous = original_checks()
    layers = load_layers()
    x = np.random.default_rng(123).integers(0, 256, (513, 36), dtype=np.uint8)
    count = 0
    for steps in protocol()['steps']:
        prepared = PreparedForward(x, layers, steps)
        for stage in protocol()['stages']:
            for encoders in ((Encoder(), Encoder()), (Encoder(128, 0), Encoder(510, 382)),
                             (Encoder(247, 103), Encoder(271, 134))):
                actual, cost = forward(x, layers, stage, steps, encoders, 'spike_add')
                cached, stats = prepared(stage, encoders)
                if (not np.array_equal(actual, cached) or cost.unit_spikes != stats['unit_spikes']
                        or cost.executed_synaptic_ac != stats['equivalent_ac']):
                    raise AssertionError('Search accelerator does not match explicit spikes')
                count += 1
    return {'status': 'passed', 'baseline': previous['baseline'],
        'unchanged_encoder_checks': previous['exhaustive_encoder_configurations'],
        'prepared_explicit_checks': count, 'training_started': False, 'weights_updated': False,
        'sources_utf8_lf': source_hashes()}


def validate_selection(calibration):
    expected = identity('task_encoder_calibration')
    if (calibration.get('status') != 'complete' or calibration.get('data_role') != 'train_main'
            or calibration.get('labels_used') is not True
            or calibration.get('weights_updated') is not False
            or calibration.get('dev_used_for_selection') is not False
            or any(calibration.get('identity', {}).get(k) != v for k, v in expected.items())):
        raise ValueError('Freeze this version of Train-main task calibration before Dev')
    requested = {(s, t) for s in protocol()['stages'] for t in protocol()['steps']}
    points = calibration['points']
    if len(points) != len(requested) or {(p['stage'], p['steps']) for p in points} != requested:
        raise ValueError('Incomplete task calibration selection')
    for point in points:
        if not point['prepared_reference_exact']:
            raise ValueError('Unchecked calibration reference')
        for mode in protocol()['modes']:
            for encoder in point[mode]['encoders']:
                Encoder(**encoder).validate()
        if point['task_matched_ac']['equivalent_ac'] > point['uncalibrated']['equivalent_ac']:
            raise ValueError('Calibration AC budget receipt mismatch')


def evaluate(cache, calibration_directory, output, chunk_rows=8192):
    if type(chunk_rows) is not int or chunk_rows < 1:
        raise ValueError('Positive chunk size required')
    path = Path(calibration_directory) / 'calibration.json'
    calibration = read_json(path)
    validate_selection(calibration)
    manifest, layers = cache_manifest(cache), load_layers()
    spec = {**identity('task_calibrated_development_evaluation'),
        'calibration_sha256': sha256(path),
        'dev_cache_manifest_sha256': sha256(Path(cache) / 'manifest.json'), 'chunk_rows': chunk_rows}
    with lock_directory(output, spec) as directory:
        final = directory / 'comparison.json'
        if final.exists():
            report = read_json(final)
            if report['identity'] != spec:
                raise ValueError('Evaluation identity mismatch')
            return report
        atomic_json(directory / 'checks.json', checks())
        plan: list[tuple[str, int, str, tuple[Encoder, ...]]] = [('ann', 255, 'ann', (Encoder(), Encoder()))]
        for point in calibration['points']:
            for mode in protocol()['modes']:
                plan.append((point['stage'], point['steps'], mode,
                             tuple(Encoder(**e) for e in point[mode]['encoders'])))
        results, originals = [], {}
        for stage, steps, mode, encoders in plan:
            atomic_json(directory / 'progress.json', {'status': 'evaluating', 'stage': stage,
                'T': steps, 'mode': mode, 'completed_points': len(results),
                'total_points': len(plan), 'training_started': False})
            result = evaluate_point(cache, manifest, layers, stage, steps, mode, encoders, directory, chunk_rows)
            for group in ('overall', 'ordinary', 'high_density'):
                counts = result[group]
                denominator = 2 * counts['tp'] + counts['fp'] + counts['fn']
                counts['f1'] = 2 * counts['tp'] / denominator if denominator else None
            baseline = results[0] if results else result
            result['delta_to_frozen_ann_pp'] = {g: {m: 100 * (result[g][m] - baseline[g][m])
                for m in ('iou', 'precision', 'recall', 'f1')} for g in ('overall', 'ordinary', 'high_density')}
            if mode == 'uncalibrated':
                originals[(stage, steps)] = result
            if stage != 'ann':
                original = originals[(stage, steps)]
                result['delta_to_uncalibrated_pp'] = {g: {m: 100 * (result[g][m] - original[g][m])
                    for m in ('iou', 'precision', 'recall', 'f1')} for g in ('overall', 'ordinary', 'high_density')}
                result['ac_reduction_vs_uncalibrated'] = 1 - result['executed_neural_ac'] / original['executed_neural_ac']
                result['spike_reduction_vs_uncalibrated'] = 1 - result['refiner_cost']['unit_spikes'] / original['refiner_cost']['unit_spikes']
                result['dev_ac_within_uncalibrated'] = result['executed_neural_ac'] <= original['executed_neural_ac']
            atomic_json(directory / f'{stage}_t{steps}_{mode}.json', result)
            results.append(result)
            print(json.dumps({'stage': 'point_complete', 'conversion': stage, 'T': steps, 'mode': mode,
                'iou': result['overall']['iou'], 'precision': result['overall']['precision'],
                'recall': result['overall']['recall'], 'AC': result['executed_neural_ac']}), flush=True)
        report = {'version': spec['version'], 'status': 'complete', 'identity': spec,
            'protocol': protocol(), 'training_started': False, 'weights_updated': False,
            'calibration_labels_used': True, 'data_role': 'train_dev', 'complete_train_dev': True,
            'reserved_sequences_read': 0, 'decision_threshold_retuned': False,
            'dev_feedback_to_calibration': False, 'calibration_sample': calibration['sample'],
            'results': results, 'frozen_baseline_after': verify_baseline(),
            'cost_scope': 'Proposal + Refiner synaptic MAC/AC; scalar/IF separate. Incomplete end-to-end accounting; no measured energy.'}
        write_table(directory, report)
        atomic_json(final, report)
        atomic_json(directory / 'progress.json', {'status': 'complete', 'training_started': False,
            'completed_points': len(results), 'comparison': str(final)})
        return report
