"""Check cycle accounting and numerical prerequisites for a future zero-group bypass."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))
from host_tools.rtl_verification.profile_rtl_sparsity import calibration_sources, choose_windows, historical_cycles, profile_mode
from host_tools.rtl_verification.generate_rtl_vectors import frozen_first_layer, lif_step
from wangdao.fixed import fixed_balanced_add_tree
from wangdao.rtl_sparsity import (COMPUTE_CYCLES, INPUTS, OUTPUTS, PATCH_CYCLES,
                                  cycle_budget, group_activity, recommend, summarize_cycles)


class RTLSparsityTests(unittest.TestCase):
    def test_cycle_contract_matches_historical_xsim(self) -> None:
        self.assertEqual(COMPUTE_CYCLES, (6912, 1536, 640))
        self.assertEqual(PATCH_CYCLES, 9600)
        rows = historical_cycles()
        self.assertEqual(rows["tb_proposal_engine"]["historical_xsim_cycles"], 9601)
        self.assertEqual(len(rows), 5)

    def test_schedule_counts_zero_group_once_per_output(self) -> None:
        rng = np.random.default_rng(930)
        inputs = [rng.integers(0, 2, (2, 4, n), dtype=np.uint8) for n in INPUTS]
        for values in inputs:
            values[:, :, :16] = 0
        empty = np.stack([group_activity(x, l)[1] for l, x in enumerate(inputs)], axis=1)
        # Walk the proposed states, including EMIT on all four readout steps.
        for selected in ((0,), (1,), (2,), (0, 1, 2)):
            measured = []
            for patch in range(2):
                cycles = 512
                for layer, values in enumerate(inputs):
                    for step in range(4):
                        for _channel in range(OUTPUTS[layer]):
                            for group in values[patch, step].reshape(-1, 8):
                                cycles += 1  # PRODUCT always executes.
                                if layer not in selected or np.any(group):
                                    cycles += 1  # GROUP_SUM
                            cycles += 4  # FINAL_SUM, ALIGN, UPDATE, EMIT
                measured.append(cycles)
            np.testing.assert_array_equal(cycle_budget(empty, selected), measured)

    def test_element_sparsity_does_not_imply_empty_groups(self) -> None:
        values = np.zeros((1, 4, 128), dtype=np.uint8)
        values[..., ::8] = 1
        activity, counts = group_activity(values, 0)
        self.assertEqual(activity["input_nonzero_fraction"], 1 / 8)
        self.assertEqual(activity["empty_group_fraction"], 0)
        self.assertEqual(activity["group_active_lane_histogram"], [0, 64, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(counts.tolist(), [0])

    def test_empty_dense_and_detection_overhead(self) -> None:
        counts = []
        for layer, width in enumerate(INPUTS):
            values = np.zeros((2, 4, width), dtype=np.uint8)
            values[1] = 255 if layer == 0 else 1
            activity, empty = group_activity(values, layer)
            self.assertEqual(activity["empty_group_fraction"], .5)
            counts.append(empty)
        empty = np.stack(counts, axis=1)
        np.testing.assert_array_equal(cycle_budget(empty, (0,)), [6528, 9600])
        np.testing.assert_array_equal(cycle_budget(empty, (0,), overhead="per_output_step"), [6720, 9792])
        np.testing.assert_array_equal(cycle_budget(empty, (0,), overhead="per_group_visit"), [9600, 12672])
        np.testing.assert_array_equal(cycle_budget(empty, (0, 1, 2)), [5760, 9600])

    def test_invalid_activity_and_cycle_counts_rejected(self) -> None:
        for values, layer in ((np.zeros((0, 4, 128), dtype=np.uint8), 0),
                              (np.zeros((1, 3, 128), dtype=np.uint8), 0),
                              (np.zeros((1, 4, 128)), 0),
                              (np.full((1, 4, 48), 2), 1),
                              (np.full((1, 4, 128), -1), 0)):
            with self.assertRaises(ValueError):
                group_activity(values, layer)
        for counts, layers, overhead in (([[65, 0, 0]], (0,), "none"),
                                         ([[0, -1, 0]], (1,), "none"),
                                         ([[0, 0, 0]], (0, 0), "none"),
                                         ([[0, 0, 0]], (3,), "none"),
                                         ([[0, 0, 0]], (0,), "unknown")):
            with self.assertRaises(ValueError):
                cycle_budget(np.asarray(counts), layers, overhead=overhead)

    def test_zero_partial_keeps_tree_order_values_and_saturation(self) -> None:
        rng = np.random.default_rng(931)
        for width in INPUTS:
            values = rng.integers(-(1 << 23), 1 << 23, (7, width), dtype=np.int64)
            groups = values.reshape(7, -1, 8)
            groups[:, ::2] = 0
            groups[0] = 0
            active = np.any(groups != 0, axis=-1)
            for k in (0, 2):
                full, full_sats, _ = fixed_balanced_add_tree(values, 24, k)
                partial = np.zeros(groups.shape[:-1], dtype=np.int64)
                partial[active], group_sats, _ = fixed_balanced_add_tree(groups[active], 24, k)
                skipped, final_sats, _ = fixed_balanced_add_tree(partial, 24, k)
                np.testing.assert_array_equal(skipped, full)
                self.assertEqual(group_sats + final_sats, full_sats)
                self.assertGreater(full_sats, 0)
                zero, zero_sats, _ = fixed_balanced_add_tree(np.zeros((1, 8), dtype=np.int64), 24, k)
                self.assertEqual((int(zero[0]), zero_sats), (0, 0))

    def test_zero_current_does_not_allow_skipping_membrane(self) -> None:
        for k in (0, 2):
            membrane, spike, saturated = lif_step(4096, 0, True, k)
            self.assertEqual((membrane, spike, saturated), (2560, 1, 0))
            membrane, spike, _ = lif_step(4096, 0, False, k)
            self.assertEqual((membrane, spike), (3584, 0))

    def test_profile_preserves_patch_isolation_and_rejects_bad_logits(self) -> None:
        features = np.random.default_rng(932).integers(0, 256, (3, 4, 128), dtype=np.uint8)
        features[1] = 0
        for model, k in zip(frozen_first_layer(), (0, 2)):
            _, counts, traces = profile_mode(features, model, k)
            for index in range(3):
                _, isolated_counts, isolated_traces = profile_mode(features[index:index + 1], model, k)
                np.testing.assert_array_equal(isolated_counts, counts[index:index + 1])
                for actual, expected in zip(isolated_traces, traces):
                    np.testing.assert_array_equal(actual, expected[index:index + 1])
            bad_final = traces[-1].copy()
            bad_final[0, 0, 0] ^= 1
            with patch("host_tools.rtl_verification.profile_rtl_sparsity.readout_trace", return_value=(traces[-2], bad_final)):
                with self.assertRaisesRegex(ValueError, "complete frozen forward"):
                    profile_mode(features, model, k)

    def test_window_selection_is_deterministic_and_anchor_deduplicated(self) -> None:
        self.assertEqual(choose_windows(np.ones(80, dtype=np.int64), True),
                         {0: ["minimum_events", "historical_rtl_anchor"],
                          40: ["median_events"], 79: ["maximum_events"]})
        selected = choose_windows(np.arange(80)[::-1], True)
        self.assertEqual(set(selected), {0, 39, 79})
        self.assertEqual(selected[0], ["maximum_events", "historical_rtl_anchor"])
        with self.assertRaises(ValueError):
            choose_windows(np.zeros(79), False)

    def test_calibration_role_and_official_root_enforced(self) -> None:
        index = {"status": "ok", "location": "dataset/EV-UAV-dataset",
                 "roles": {"hw_calibration": ["train_022.npz"], "test": ["test_001.npz"]},
                 "files": [{"path": "train/train_022.npz"}]}
        self.assertEqual(calibration_sources(index, ROOT)[0][0],
                         ROOT / "dataset/EV-UAV-dataset/train/train_022.npz")
        overlap = copy.deepcopy(index)
        overlap["roles"]["test"].append("train_022.npz")
        escape = copy.deepcopy(index)
        escape["files"][0]["path"] = "../train_022.npz"
        external = copy.deepcopy(index)
        external["location"] = "tmp/other-data"
        for invalid in (overlap, escape, external):
            with self.assertRaises(ValueError):
                calibration_sources(invalid, ROOT)

    def test_pilot_cannot_recommend_implementation_and_dense_case_fails(self) -> None:
        def window(counts):
            return {"modes": {mode: {"cycle_budget": summarize_cycles(np.asarray([counts]))}
                              for mode in ("exact", "approximate")}}
        zero = window([64, 24, 12])
        self.assertIsNone(recommend([zero], False)["recommended_first_layer"])
        self.assertEqual(recommend([zero], True)["recommended_first_layer"], "layer1")
        self.assertIsNone(recommend([window([0, 0, 0])], True)["recommended_first_layer"])
        self.assertIsNone(recommend([zero, window([0, 0, 0])], True)["recommended_first_layer"])


if __name__ == "__main__":
    unittest.main()
