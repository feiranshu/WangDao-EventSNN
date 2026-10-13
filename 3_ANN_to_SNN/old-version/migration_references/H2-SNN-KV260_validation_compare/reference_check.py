"""Execute pinned official metric methods with explicit CPU-only adaptations.

This does NOT execute EV-SpSegNet. The CUDA-to-CPU change affects metric tensors
only. If OpenCV is unavailable, an independent union-find implements just the
8-connected component COUNT consumed by the official evaluator.
"""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import warnings

import numpy as np
import torch

from metrics import CORRECT_THRESHOLD, WINDOW_MS


def union_find_components(mask):
    height, width = mask.shape
    active = set(np.flatnonzero(mask).tolist())
    parent = {point: point for point in active}

    def find(point):
        while parent[point] != point:
            parent[point] = parent[parent[point]]
            point = parent[point]
        return point

    for point in sorted(active):
        y, x = divmod(point, width)
        for dy, dx in ((0, -1), (-1, -1), (-1, 0), (-1, 1)):
            yy, xx = y + dy, x + dx
            neighbor = yy * width + xx
            if 0 <= yy < height and 0 <= xx < width and neighbor in active:
                parent[find(point)] = find(neighbor)
    return len({find(point) for point in active})


class CV2CountAdapter:
    CV_32S = 4

    @staticmethod
    def connectedComponentsWithStats(mask, connectivity, ltype):
        if connectivity != 8 or ltype != 4:
            raise ValueError("Only the pinned evaluator's count API is supported")
        return union_find_components(mask) + 1, None, None, None


class CpuMetricTensors(ast.NodeTransformer):
    def visit_Attribute(self, node):
        self.generic_visit(node)
        if node.attr == "cuda":
            node.attr = "cpu"
        return node


def reference_class():
    path = Path(__file__).resolve().parent / "official_reference/utils/eval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "evalute"]
    if len(classes) != 1:
        raise ValueError("Unexpected official evaluator structure")
    tree = ast.Module(body=classes, type_ignores=[])
    tree = ast.fix_missing_locations(CpuMetricTensors().visit(tree))
    try:
        import cv2
        backend = "opencv"
    except ImportError:
        cv2 = CV2CountAdapter
        backend = "independent_union_find_count_adapter_not_native_opencv"
    namespace = {"np": np, "torch": torch, "cv2": cv2,
                 "pd": SimpleNamespace(DataFrame=dict)}
    exec(compile(tree, str(path) + "[CPU-metric-adapter]", "exec"), namespace)
    return namespace["evalute"], backend


class ReferenceEvaluator:
    def __init__(self):
        cls, self.backend = reference_class()
        self.evaluator = cls(SimpleNamespace(roc=True, pd_detT=WINDOW_MS, correct_thresh=CORRECT_THRESHOLD))

    def add(self, name, loc, labels, identities, probabilities, threshold):
        xyz = np.asarray(loc).astype(np.int64).astype(np.float32)
        ev_locs = torch.from_numpy(np.column_stack((np.zeros(len(xyz), dtype=np.float32), xyz)))
        p = torch.from_numpy(np.ascontiguousarray(probabilities, dtype=np.float32)).clone()
        y = torch.from_numpy(np.ascontiguousarray(labels, dtype=np.float32))
        self.evaluator.matches[name] = {"seg_gt": y, "seg_pred": p.clone()}
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="overflow encountered in scalar add", category=RuntimeWarning)
            self.evaluator.roc_update(ev_locs[:, 3], p, np.asarray(identities), y,
                                      ev_locs, thresh=threshold)

    def summary(self, threshold):
        evaluator = self.evaluator
        pd, fa = evaluator.cal_roc()
        return {"iou": float(evaluator.evaluate_semantic_segmantation_miou(thresh=threshold)),
                "official_acc": float(evaluator.evaluate_semantic_segmantation_accuracy(thresh=threshold)),
                "pd": float(pd), "fa": float(fa), "targets": int(evaluator.obj_num),
                "correct_targets": int(evaluator.correct_num),
                "false_components": int(evaluator.false_num),
                "frame_denominator": int(evaluator.frame_num),
                "connected_component_backend": self.backend}
