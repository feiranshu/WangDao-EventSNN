"""Protocol/selection tests independent of model training and Test data."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wangdao import phase1_common as common
from wangdao import phase1_metrics as metric


class PhaseOneProtocolTests(unittest.TestCase):
    def test_reserved_data_roles_rejected_before_file_access(self):
        for role in ("test", "validation", "train", "../../test"):
            with self.assertRaises(ValueError):
                common.role_paths(role)

    def test_metric_counts_aggregate_before_division(self):
        loc = np.array([[1, 1, 1], [2, 2, 101]], dtype=float)
        labels, ids = np.array([1, 0]), np.array([1, 0])
        a = common.score_predictions(loc, labels, ids, np.array([1., 0.]), .28)
        b = common.score_predictions(loc, labels, ids, np.array([0., 1.]), .28)
        total = common.aggregate_scores([a, b])
        self.assertEqual(total["event"]["iou"], 1 / 3)
        self.assertEqual(total["event"]["official_acc"], .5)
        self.assertEqual(total["localization"]["targets"], 2)

    def test_constraints_reject_high_iou_unsafe_working_point(self):
        def result(iou, pd, fa):
            return {"event": {"iou": iou}, "localization": {"pd": pd, "fa": fa}}
        curve = {"0.2": result(.95, .8, .2), "0.3": result(.82, .9, .101),
                 "0.4": result(.82, .9, .101), "0.5": result(.84, .7, .08)}
        chosen = common.select_working_point(curve, .9, .1)
        self.assertTrue(chosen["eligible"])
        self.assertEqual(chosen["threshold"], .4)
        self.assertFalse(common.select_working_point(curve, None, None)["eligible"])
        self.assertFalse(common.select_working_point(curve, 1., 0.)["eligible"])

    def test_pr_curve_groups_ties_and_includes_rejected_events(self):
        curve = common.pr_curve(np.array([1, 0, 1]), np.array([.8, .8, 0.]))
        self.assertEqual(curve["thresholds"], [.8, 0.])
        self.assertAlmostEqual(curve["average_precision"], .5 * .5 + .5 * (2 / 3))
        self.assertIsNone(common.pr_curve(np.array([0]), np.array([0.]))["average_precision"])

    def test_latency_includes_wait_and_backlog(self):
        result = common.latency_summary([25., 50.], [40., 10.],
                                        [np.array([0., 20.]), np.array([30., 40.])])
        self.assertEqual(result["max_backlog_blocks"], 1)
        self.assertEqual(result["event_end_to_end_ms"]["p50"], 45.)
        self.assertEqual(result["event_queue_ms"]["p50"], 7.5)
        self.assertIsNone(result["power_w"])

    def test_json_no_overwrite_and_nan_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "result.json"
            common.write_json(p, {"v": 1})
            with self.assertRaises(FileExistsError):
                common.write_json(p, {"v": 2})
            common.write_json(p, {"v": 2}, overwrite=True)
            self.assertEqual(json.loads(p.read_text())["v"], 2)
            with self.assertRaises(ValueError):
                common.write_json(Path(temp) / "nan.json", {"v": float("nan")})

    def test_pinned_official_reference_parity(self):
        # Pinned ChenYichen9527/EV-UAV commit 33e2affb51a204676536e119803ab8494e2eb0a9.
        # Only metric .cuda() calls are redirected; no official network is run.
        source = Path(__file__).with_name("evuav_eval_33e2aff.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        tree = ast.Module(body=[n for n in tree.body if isinstance(n, ast.ClassDef)], type_ignores=[])
        class Cpu(ast.NodeTransformer):
            def visit_Attribute(self, node):
                self.generic_visit(node)
                if node.attr == "cuda":
                    node.attr = "cpu"
                return node
        class CV2:
            CV_32S = 4
            @staticmethod
            def connectedComponentsWithStats(mask, connectivity, ltype):
                # Independent raster union-find, not the evaluator's flood fill.
                active = set(np.flatnonzero(mask).tolist())
                parent = {p: p for p in active}
                def root(p):
                    while parent[p] != p:
                        parent[p] = parent[parent[p]]
                        p = parent[p]
                    return p
                width = mask.shape[1]
                for p in sorted(active):
                    y, x = divmod(p, width)
                    for dy, dx in ((0, -1), (-1, -1), (-1, 0), (-1, 1)):
                        yy, xx = y + dy, x + dx
                        q = yy * width + xx
                        if 0 <= yy < mask.shape[0] and 0 <= xx < width and q in active:
                            parent[root(p)] = root(q)
                return len({root(p) for p in active}) + 1, None, None, None
        scope = {"np": np, "torch": torch, "cv2": CV2, "pd": SimpleNamespace(DataFrame=dict)}
        exec(compile(ast.fix_missing_locations(Cpu().visit(tree)), "official_reference", "exec"), scope)
        rng = np.random.default_rng(12)
        n = 600
        loc = np.column_stack((rng.integers(0, 30, n), rng.integers(0, 30, n),
                               np.linspace(0, 201, n)))
        labels = (rng.random(n) > .7).astype(np.float32)
        ids = np.where(labels, rng.integers(1, 4, n), 0)
        probabilities = rng.random(n).astype(np.float32)
        for threshold in common.REFERENCE_THRESHOLDS:
            ref = scope["evalute"](SimpleNamespace(roc=True, pd_detT=50, correct_thresh=.0001))
            xyz = loc.astype(np.int64).astype(np.float32)
            tensor_loc = torch.from_numpy(np.column_stack((np.zeros(n, np.float32), xyz)))
            y = torch.from_numpy(labels)
            p = torch.from_numpy(probabilities.copy())
            ref.matches["synthetic"] = {"seg_gt": y, "seg_pred": p.clone()}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                ref.roc_update(tensor_loc[:, 3], p, ids, y, tensor_loc, thresh=threshold)
            actual = common.score_predictions(loc, labels, ids, probabilities, threshold)
            self.assertAlmostEqual(actual["event"]["iou"], float(ref.evaluate_semantic_segmantation_miou(threshold)), places=7)
            self.assertAlmostEqual(actual["event"]["official_acc"], float(ref.evaluate_semantic_segmantation_accuracy(threshold)), places=7)
            pd, fa = ref.cal_roc()
            self.assertEqual(actual["localization"]["pd"], pd)
            self.assertEqual(actual["localization"]["fa"], fa)


if __name__ == "__main__":
    unittest.main()
