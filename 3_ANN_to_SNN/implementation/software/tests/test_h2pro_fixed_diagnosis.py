"""Numerical/role-boundary regressions; no optimizer or training steps."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from wangdao.h2pro_fixed_diagnosis_v1.thresholds import (
    CONTROL, GROUPS, choose, curve, measure, transitions, validate_scores)
from wangdao.h2pro_fixed_diagnosis_v1.workflow import checked_bundle, seal
from wangdao.h2pro_distill_v1.common import atomic_json


def fixture():
    # Include rejected positive/background events, duplicate scores, 0 and 1.
    rows, parts, ys, hds = [], [], [], []
    offset = 0
    for hd, p, y in ((False, [0, .35, .35, .9], [1,1,0,0,1,0]),
                     (True, [.1, .2, .8, 1], [1,0,1,0,1,0])):
        p = np.array(p, np.float32); y = np.array(y, np.uint8)
        c = np.array([1,1,1,1,0,0], bool)
        rows.append({'row': {'events': len(y), 'high_density': hd, 'sequence': str(hd)},
                     'labels': y, 'candidate': c, 'lo': offset, 'hi': offset+4})
        parts.append(p); ys.append(y[c]); hds.append(np.full(4, hd, bool)); offset += 4
    return rows, np.concatenate(parts), np.concatenate(ys), np.concatenate(hds), {
        'overall': 6, 'high_density': 3, 'ordinary': 3}


class ThresholdTests(unittest.TestCase):
    def test_every_tie_and_endpoint_matches_full_event_brute_force(self):
        rows, p, y, hd, totals = fixture()
        c = curve(p, y, hd, totals)
        for i, threshold in enumerate(c['threshold']):
            direct = measure(rows, p, threshold)
            for group in GROUPS:
                for name in ('tp','fp','iou','recall'):
                    self.assertEqual(c[group+'_'+name][i], direct[group][name])
        selected = measure(rows, p, CONTROL)
        self.assertEqual(selected['overall']['tp'], 2)
        self.assertEqual(selected['overall']['proposal_fn'], 2)
        self.assertEqual(selected['overall']['fp'], 3)

    def test_randomized_curves_match_bruteforce_and_constrained_optimum(self):
        rng = np.random.default_rng(17)
        for _ in range(40):
            p = rng.integers(0, 11, 200).astype(np.float32)/10
            y = rng.integers(0, 2, 200, dtype=np.uint8)
            hd = rng.random(200) < .5
            totals = {'overall': int(y.sum())+10, 'high_density': int(y[hd].sum())+4,
                      'ordinary': int(y[~hd].sum())+6}
            c = curve(p, y, hd, totals)
            limits = {'overall': {'tp': int(y.sum())//2},
                      'high_density': {'tp': int(y[hd].sum())//2}}
            feasible = []
            for i, t in enumerate(c['threshold']):
                for group, mask in zip(GROUPS, (np.ones(200,bool), hd, ~hd)):
                    tp = int((mask & (y==1) & (p>=t)).sum())
                    fp = int((mask & (y==0) & (p>=t)).sum())
                    self.assertEqual(tp, c[group+'_tp'][i])
                    self.assertEqual(fp, c[group+'_fp'][i])
                    self.assertEqual(tp/(totals[group]+fp), c[group+'_iou'][i])
                if all(c[g+'_tp'][i] >= limits[g]['tp'] for g in limits):
                    feasible.append(i)
            choice = choose(c, limits, 'main_recall_matched', role='train_main')
            expected = max(feasible, key=lambda i: (c['overall_iou'][i],
                -abs(float(c['threshold'][i])-CONTROL), float(c['threshold'][i])))
            self.assertEqual(choice['threshold'], float(c['threshold'][expected]))

    def test_both_recall_constraints_apply_to_one_global_threshold(self):
        _, p, y, hd, totals = fixture()
        c = curve(p, y, hd, totals)
        choice = choose(c, {'overall': {'tp': 2}, 'high_density': {'tp': 2}},
                        'main_recall_matched', role='train_main')
        self.assertLessEqual(choice['threshold'], float(np.float32(.1)))
        self.assertEqual(choice['main']['high_density']['recall'], 2/3)

    def test_infeasible_constraint_is_not_relaxed(self):
        _, p, y, hd, totals = fixture()
        c = curve(p, y, hd, totals)
        result = choose(c, {'overall': {'tp': 100}, 'high_density': {'tp': 2}},
                        'main_recall_matched', role='train_main')
        self.assertIsNone(result['threshold'])
        self.assertEqual(result['status'], 'infeasible')

    def test_dev_cannot_select(self):
        _, p, y, hd, totals = fixture()
        with self.assertRaisesRegex(ValueError, 'Train-main'):
            choose(curve(p, y, hd, totals), {}, 'main_iou', role='train_dev')

    def test_transition_differences_equal_metric_changes(self):
        rows, p, y, hd, _ = fixture()
        before, after = measure(rows, p, CONTROL), measure(rows, p, .1)
        changes = transitions(p, p, y, hd, .1)
        for group in GROUPS:
            d = changes[group]
            self.assertEqual(after[group]['tp']-before[group]['tp'],
                             d['recovered_targets_vs_035']-d['lost_targets_vs_035'])
            self.assertEqual(after[group]['fp']-before[group]['fp'],
                             d['added_fp_vs_035']-d['removed_fp_vs_035'])

    def test_invalid_scores_and_thresholds_rejected(self):
        rows, p, y, hd, _ = fixture()
        for t in (0, 1, float('nan'), -1):
            with self.assertRaises(ValueError):
                measure(rows, p, t)
        for invalid in (p.astype(np.float64), np.full_like(p, np.nan), p+1):
            with self.assertRaises(ValueError):
                validate_scores(invalid, y, hd)

    def test_sealed_selection_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            atomic_json(root/'selection.json', {'threshold': .2})
            seal(root, ['selection.json'], 'checksums.json')
            self.assertEqual(checked_bundle(root, 'selection.json', 'checksums.json'), {'threshold': .2})
            atomic_json(root/'selection.json', {'threshold': .3})
            with self.assertRaisesRegex(ValueError, 'Sealed stage'):
                checked_bundle(root, 'selection.json', 'checksums.json')


if __name__ == '__main__':
    unittest.main()
