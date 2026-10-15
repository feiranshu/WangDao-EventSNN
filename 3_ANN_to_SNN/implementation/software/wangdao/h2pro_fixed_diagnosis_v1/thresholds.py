"""Exact tied-score threshold curves, retaining every Proposal false negative."""
from __future__ import annotations

import numpy as np

from ..h2pro_distill_v1.evaluation import summarize
from ..h2pro_snn.diagnostics import error_counts

GROUPS = ('overall', 'high_density', 'ordinary')
CONTROL = float(np.float32(.35))
POLICIES = ('main_iou', 'main_recall_matched')


def validate_scores(probability, labels, high_density):
    p, y, hd = map(np.asarray, (probability, labels, high_density))
    if (p.ndim != 1 or p.shape != y.shape or p.shape != hd.shape
            or p.dtype != np.float32 or hd.dtype != np.bool_
            or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()
            or not np.isin(y, [0, 1]).all()):
        raise ValueError('Expected finite FP32 candidate scores, binary labels and HD flags')
    return p, y.astype(bool), hd


def curve(probability, labels, high_density, target_totals):
    """All attainable decisions for FP32 thresholds in (0,1), with >= ties.

    target_totals counts FULL events, including targets rejected by Proposal.
    Endpoints just inside (0,1) keep that historical threshold contract; scores
    equal to zero never pass and scores equal to one always pass.
    """
    p, y, hd = validate_scores(probability, labels, high_density)
    if set(target_totals) != set(GROUPS):
        raise ValueError('Full-event target totals required for all three groups')
    order = np.argsort(p, kind='stable')
    sorted_p, y, hd = p[order], y[order], hd[order]
    knots = np.unique(np.concatenate((sorted_p[(sorted_p > 0) & (sorted_p < 1)],
                    np.array([np.nextafter(np.float32(0), np.float32(1)), CONTROL,
                              np.nextafter(np.float32(1), np.float32(0))], np.float32))))
    excluded = np.searchsorted(sorted_p, knots, side='left')
    out = {'threshold': knots}
    for group, mask in zip(GROUPS, (np.ones(len(y), bool), hd, ~hd)):
        positive = np.concatenate(([0], np.cumsum(mask & y, dtype=np.int64)))
        negative = np.concatenate(([0], np.cumsum(mask & ~y, dtype=np.int64)))
        targets = int(target_totals[group])
        if targets < int(positive[-1]):
            raise ValueError('Full-event targets cannot be fewer than candidate targets')
        tp, fp = positive[-1]-positive[excluded], negative[-1]-negative[excluded]
        out[group+'_tp'], out[group+'_fp'] = tp, fp
        for name, denominator in (('recall', np.full(len(tp), targets)),
                                  ('iou', targets+fp), ('precision', tp+fp)):
            out[group+'_'+name] = np.divide(tp, denominator,
                out=np.zeros(len(tp), np.float64), where=denominator != 0)
    return out


def choose(curves, ann_counts, policy, *, role):
    """One global threshold; only Main may select. Dev can only score it."""
    if role != 'train_main' or policy not in POLICIES:
        raise ValueError('Threshold selection is restricted to Train-main and locked policies')
    feasible = np.ones(len(curves['threshold']), bool)
    if policy == 'main_recall_matched':
        # Same full-event denominators => integer TP comparisons avoid tolerance.
        for group in ('overall', 'high_density'):
            feasible &= curves[group+'_tp'] >= ann_counts[group]['tp']
    indices = np.flatnonzero(feasible)
    if not len(indices):
        return {'status': 'infeasible', 'threshold': None, 'policy': policy}
    values = curves['overall_iou'][indices]
    indices = indices[values == values.max()]
    # Fixed tie-break: nearest inherited threshold, then the higher threshold.
    distances = np.abs(curves['threshold'][indices].astype(np.float64)-CONTROL)
    indices = indices[distances == distances.min()]
    best = int(indices[np.argmax(curves['threshold'][indices])])
    return {'status': 'selected', 'policy': policy,
            'threshold': float(curves['threshold'][best]),
            'threshold_fp32_hex': curves['threshold'][best].tobytes().hex(),
            'unique_decisions_checked': len(curves['threshold']),
            'feasible_decisions': int(feasible.sum()),
            'main': {g: {k: float(curves[g+'_'+k][best]) for k in ('iou','precision','recall')}
                     for g in GROUPS}}


def measure(sequences, probability, threshold):
    """Independent direct rescore, not the sorted-curve implementation."""
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError('Threshold must lie in (0,1)')
    threshold = float(np.float32(threshold))
    if not 0 < threshold < 1:
        raise ValueError('Threshold must remain inside (0,1) in FP32')
    rows = []
    for seq in sequences:
        row = seq['row']
        full = np.zeros(row['events'], np.float32)
        full[seq['candidate']] = probability[seq['lo']:seq['hi']]
        rows.append({'sequence': row['sequence'], 'high_density': row['high_density'],
                     'counts': error_counts(seq['labels'], full, seq['candidate'], threshold)})
    result = {'threshold': threshold, 'sequences': rows}
    for group in GROUPS:
        subset = rows if group == 'overall' else [r for r in rows if r['high_density'] == (group == 'high_density')]
        result[group] = summarize([r['counts'] for r in subset])
    return result


def distributions(probability, labels, hd):
    p, y, hd = validate_scores(probability, labels, hd)
    bins = np.linspace(0, 1, 101)
    result = {'scope': 'candidate events only; Proposal rejects excluded', 'bins': bins.tolist()}
    for group, mask in zip(GROUPS, (np.ones(len(y), bool), hd, ~hd)):
        result[group] = {}
        for name, selected in (('target', mask & y), ('background', mask & ~y)):
            values = p[selected]
            result[group][name] = {'events': int(len(values)),
                'quantile_levels': [0, .01, .1, .25, .5, .75, .9, .99, 1],
                'quantiles': np.quantile(values, [0, .01, .1, .25, .5, .75, .9, .99, 1]).tolist() if len(values) else None,
                'histogram': np.histogram(values, bins)[0].tolist()}
    return result


def transitions(probability, teacher, labels, hd, threshold):
    """Target recovery and added false positives at the selected operating point."""
    p, y, hd = validate_scores(probability, labels, hd)
    selected, control, ann = p >= np.float32(threshold), p >= CONTROL, teacher >= CONTROL
    result = {}
    for group, mask in zip(GROUPS, (np.ones(len(y), bool), hd, ~hd)):
        predicates = {
            'recovered_targets_vs_035': y & selected & ~control,
            'lost_targets_vs_035': y & ~selected & control,
            'added_fp_vs_035': ~y & selected & ~control,
            'removed_fp_vs_035': ~y & ~selected & control,
            'ann_detected_student_missed_targets': y & ann & ~selected,
            'student_detected_ann_missed_targets': y & selected & ~ann,
            'student_only_fp_vs_ann': ~y & selected & ~ann,
            'ann_only_fp_vs_student': ~y & ~selected & ann,
        }
        result[group] = {key: int((mask & value).sum()) for key, value in predicates.items()}
    return result
