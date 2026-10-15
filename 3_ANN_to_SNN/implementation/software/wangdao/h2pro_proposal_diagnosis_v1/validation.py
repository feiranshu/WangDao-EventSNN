"""Intervention validity checked against the untouched causal implementation."""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..h2pro_proposal_transfer_v1.pipeline import TransferPipeline
from .cache import build
from .core import (GROUPS, ss, context_features, component_column, event_packet, event_scores,
                   map_heatmaps, max_neighbors, recover_gate, refiner, swapped, threshold_curve)
from .experiments import acceptance, infer_spec, spec
from .workflow import fit_affine


def checks():
    rng = np.random.default_rng(71402)
    n, t0 = 120, 13.25
    packet = np.empty(n, dtype=ss.EVENT_DTYPE)
    packet['x'] = rng.integers(0, 346, n)
    packet['y'] = rng.integers(0, 260, n)
    packet['p'] = rng.integers(0, 2, n)
    packet['t'] = t0+np.r_[np.linspace(0, 99.999, 40), np.linspace(200, 299.999, 40),
                          np.linspace(300, 399.999, 40)]
    arrays = {k: packet[k].copy() for k in ('x', 'y', 't', 'p')}
    arrays['label'] = rng.integers(0, 2, n, dtype=np.uint8)
    meta = {'t0_ms': t0, 'closed_windows': 4}
    data, _ = build(arrays, meta)
    names = []
    for mode, backend in (('ann', 'ann_dense'), ('s1', 's1_event')):
        for gate in (.0, .49, .51, .8, 1.):
            pipe = TransferPipeline(backend)
            pipe.proposal = replace(pipe.proposal, proposal_threshold=gate)
            pipe.reset(t0)
            pipe.push(packet, t0+400)
            windows, tail = pipe.finalize()
            assert tail == 0
            features = context_features(packet, data['windows'], data['base'], data[mode+'_heatmaps'], gate, t0)
            output, _ = infer_spec(data, arrays, meta, spec('check', route=mode, context=mode, gate=gate, context_gate=gate),
                                   {'slope': 1., 'intercept': 0.}, pipe.refiner_layers)
            candidate = np.concatenate([w.candidate_mask for w in windows])
            np.testing.assert_array_equal(features[candidate], np.concatenate([w.features for w in windows]))
            np.testing.assert_array_equal(output['candidate'], candidate)
            np.testing.assert_array_equal(output['probabilities'], np.concatenate([w.probabilities for w in windows]))
            np.testing.assert_array_equal(component_column(packet, data['windows'], data[mode+'_heatmaps'], gate), features[:, 30])
        names.append(mode+'_five_gate_exact_causal_replay_including_empty_windows')
    hm = rng.random((4, 34, 44), dtype=np.float32)
    for gate in (.0, .49, .51, 1.):
        np.testing.assert_array_equal(max_neighbors(hm) >= gate, ss.dilate_cells(hm >= gate))
    names.append('dilation_score_threshold_exact')
    f = context_features(packet, data['windows'], data['base'], data['s1_heatmaps'], .51, t0)
    changed = data['s1_heatmaps'].copy()
    changed[2:] = 1-changed[2:]
    f2 = context_features(packet, data['windows'], data['base'], changed, .51, t0)
    np.testing.assert_array_equal(f[data['windows'] < 2], f2[data['windows'] < 2])
    names.append('future_heatmaps_do_not_affect_past_features')
    poison = dict(arrays, label=1-arrays['label'])
    data2, _ = build(poison, meta)
    for k in data:
        np.testing.assert_array_equal(data[k], data2[k])
    names.append('all_feature_inference_label_independent')
    ann = context_features(packet, data['windows'], data['base'], data['ann_heatmaps'], .51, t0)
    for group, columns in GROUPS.items():
        mixed = swapped(ann, f, group)
        untouched = [j for j in range(36) if j not in columns]
        np.testing.assert_array_equal(mixed[:, untouched], ann[:, untouched])
        np.testing.assert_array_equal(mixed[:, columns], f[:, columns])
    names.append('single_feature_group_interventions_leave_other_columns_exact')
    for size in (0, 1, 77):
        scores = rng.choice(np.array([.1, .5, .9, 1.], np.float32), size)
        label = rng.integers(0, 2, size)
        prob = rng.random(size)
        hd = rng.integers(0, 2, size).astype(bool)
        curve = threshold_curve(scores, label, prob, hd)
        for i, gate in enumerate(curve['threshold']):
            c = scores.astype(np.float64) >= gate
            p = c & (prob >= .35)
            assert curve['candidate_events'][i] == c.sum()
            assert curve['candidate_targets'][i] == (c & (label == 1)).sum()
            assert curve['tp'][i] == (p & (label == 1)).sum()
            assert curve['fp'][i] == (p & (label == 0)).sum()
            assert curve['hd_tp'][i] == (p & (label == 1) & hd).sum()
    names.append('exact_frontier_matches_bruteforce_with_ties_empty_and_one')
    curve = threshold_curve(np.array([.9, .8, .2]), np.ones(3), np.ones(3), np.array([False, False, True]))
    assert recover_gate(curve, 2, 1) == .2
    names.append('recovery_requires_both_overall_and_hd_targets')
    a = np.arange(10, dtype=np.float64)
    b = 2*a-3
    fit = fit_affine(np.array([len(a), a.sum(), b.sum(), a@a, a@b, b@b]))
    assert fit['slope'] == 2 and fit['intercept'] == -3 and fit['logit_mse'] == 0
    mapped = map_heatmaps(np.linspace(0, 1, 20, dtype=np.float32), fit)
    assert np.isfinite(mapped).all() and np.all(np.diff(mapped) >= 0)
    names.append('closed_form_positive_logit_calibration_is_monotonic_and_finite')
    base = {'overall': {'iou': .8, 'precision': .9, 'recall': .9}, 'high_density': {'recall': .8}, 'mac': 100}
    result: dict = dict(base, ac=1, diagnostic_only=True)
    result['mac'] = 80
    assert not acceptance(result, base)['passes']
    result['diagnostic_only'] = False
    assert acceptance(result, base)['passes']
    result['high_density'] = {'recall': .7}
    assert not acceptance(result, base)['passes']
    names.append('acceptance_rejects_oracles_and_hd_recall_loss')
    assert spec('counterfactual', route='s1', context='ann')['diagnostic_only']
    assert not spec('calibration', route='s1', context='calibrated')['diagnostic_only']
    names.append('mixed_proposal_context_is_explicitly_diagnostic')
    return {'status': 'passed', 'checks': names, 'check_count': len(names), 'training_started': False}
