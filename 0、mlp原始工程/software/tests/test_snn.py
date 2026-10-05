"""S0-P 的 LIF、时间切片、数据边界和梯度测试。"""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.h2.numpy_model import (
    load_export,
    quantized_mlp,
    quantized_mlp_with_activity,
)
from wangdao.h2.constants import (
    FEATURE_CELL,
    OUTPUT_CELL,
    PADDED_HEIGHT,
    PADDED_WIDTH,
    PATCH_COLUMNS,
    PATCH_FEATURE_DIM,
    PATCH_OUTPUT_DIM,
    PATCH_ROWS,
    PATCH_SIZE,
    PATCH_STRIDE,
    PATCHES_PER_WINDOW,
)
from wangdao.h2.features import compress_counts
from wangdao.fixed import (
    ApproximateProposalSNN,
    FixedH2,
    FixedH2Config,
    FixedProposalSNN,
    FixedSNNConfig,
    LOAPlacement,
    PoTProposalSNN,
    TorchApproximateProposalSNN,
    export_approximate_proposal_snn,
    export_fixed_proposal_snn,
    export_pot_proposal_snn,
    fixed_balanced_add_tree,
    load_fixed_proposal_snn,
    load_approximate_proposal_snn,
    load_pot_proposal_snn,
    loa_add_signed,
    signed_round_divide,
)
from wangdao.h2.representation import (
    build_patch_features,
    build_patch_targets,
    build_temporal_patch_features,
    build_temporal_patch_targets,
    patch_scores_to_heatmaps,
    temporal_window_indices,
    window_indices,
)
from wangdao.h2.torch_model import load_exported_models
from wangdao.s0p import role_file_paths, s0p_loss, sample_patch_indices
from wangdao.s0r import S0RWindow, flatten_s0r_windows, s0r_loss
from wangdao.snn import (
    LIFCell,
    ProposalSNN,
    RefinerSNN,
    export_proposal_snn,
    export_refiner_snn,
    load_proposal_snn_export,
    load_refiner_snn_export,
    profile_proposal_snn,
)
from wangdao.stc import (
    binary_rank_auc,
    patch_probabilities_to_heatmaps,
    proposal_stc_loss,
    stc_block_ranges,
    stc_weights,
)


WEIGHT = ROOT / "software" / "weights" / "H2_D1_seed42.npz"
INDEX = ROOT / "data" / "index.json"


def temporal_events() -> np.ndarray:
    dtype = np.dtype([
        ("x", "<i2"), ("y", "<i2"), ("t", "<f8"), ("p", "u1"),
        ("label", "u1"), ("name", "<i2"),
    ])
    timestamps = np.asarray([0.0, 24.999, 25.0, 99.999, 100.0, 124.999])
    events = np.zeros(len(timestamps), dtype=dtype)
    events["x"] = 2
    events["y"] = 3
    events["t"] = timestamps
    events["p"] = np.arange(len(events)) % 2
    return events


def reference_patch_features(
    events: np.ndarray, windows: list[int], time_steps: int | None = None
) -> np.ndarray:
    """直接复刻优化前的逐窗口、逐小块实现，作为数值对照。"""
    assigned = window_indices(events)
    micro = None
    if time_steps is not None:
        assigned, micro = temporal_window_indices(events, time_steps)
    prefix = (len(windows),) if time_steps is None else (len(windows), time_steps)
    result = np.zeros((*prefix, PATCHES_PER_WINDOW, PATCH_FEATURE_DIM), dtype=np.uint8)
    for output_window, window in enumerate(windows):
        steps = (None,) if time_steps is None else range(time_steps)
        for step in steps:
            member = assigned == int(window)
            if step is not None:
                member &= micro == step
            selected = events[member]
            fine = np.zeros(
                (2, PADDED_HEIGHT // FEATURE_CELL, PADDED_WIDTH // FEATURE_CELL),
                dtype=np.int32,
            )
            np.add.at(
                fine,
                (
                    selected["p"].astype(np.int64),
                    selected["y"].astype(np.int64) // FEATURE_CELL,
                    selected["x"].astype(np.int64) // FEATURE_CELL,
                ),
                1,
            )
            row = 0
            for patch_y in range(PATCH_ROWS):
                y0 = patch_y * PATCH_STRIDE // FEATURE_CELL
                for patch_x in range(PATCH_COLUMNS):
                    x0 = patch_x * PATCH_STRIDE // FEATURE_CELL
                    values = fine[
                        :,
                        y0:y0 + PATCH_SIZE // FEATURE_CELL,
                        x0:x0 + PATCH_SIZE // FEATURE_CELL,
                    ].reshape(-1)
                    if step is None:
                        target = result[output_window, row]
                    else:
                        target = result[output_window, step, row]
                    target[:] = compress_counts(values)
                    row += 1
    return result


def reference_patch_targets(events: np.ndarray, windows: list[int]) -> np.ndarray:
    """直接复刻优化前的逐窗口、逐小块标签构造。"""
    assigned = window_indices(events)
    result = np.zeros(
        (len(windows), PATCHES_PER_WINDOW, PATCH_OUTPUT_DIM), dtype=np.uint8
    )
    for output_window, window in enumerate(windows):
        target = (assigned == int(window)) & events["label"].astype(bool)
        coarse = np.zeros(
            (PADDED_HEIGHT // OUTPUT_CELL, PADDED_WIDTH // OUTPUT_CELL), dtype=np.uint8
        )
        coarse[
            events["y"][target].astype(np.int64) // OUTPUT_CELL,
            events["x"][target].astype(np.int64) // OUTPUT_CELL,
        ] = 1
        row = 0
        for patch_y in range(PATCH_ROWS):
            y0 = patch_y * PATCH_STRIDE // OUTPUT_CELL
            for patch_x in range(PATCH_COLUMNS):
                x0 = patch_x * PATCH_STRIDE // OUTPUT_CELL
                result[output_window, row] = coarse[
                    y0:y0 + PATCH_SIZE // OUTPUT_CELL,
                    x0:x0 + PATCH_SIZE // OUTPUT_CELL,
                ].reshape(-1)
                row += 1
    return result


class LIFTests(unittest.TestCase):
    def test_accumulation_soft_reset_and_gradient(self) -> None:
        cell = LIFCell(beta=0.5, threshold=1.0, surrogate_slope=5.0)
        membrane = None
        first = torch.tensor([[0.75]], requires_grad=True)
        spike_1, membrane = cell(first, membrane)
        spike_2, membrane = cell(torch.tensor([[0.75]]), membrane)
        self.assertEqual(float(spike_1.item()), 0.0)
        self.assertEqual(float(spike_2.item()), 1.0)
        self.assertAlmostEqual(float(membrane.item()), 0.125, places=6)
        (spike_1 + spike_2 + membrane).sum().backward()
        first_gradient = first.grad
        self.assertIsNotNone(first_gradient)
        if first_gradient is None:
            raise AssertionError("surrogate gradient is absent")
        self.assertGreater(abs(float(first_gradient.item())), 0.0)

    def test_proposal_shape_initialization_and_state_isolation(self) -> None:
        teacher, _ = load_exported_models(WEIGHT)
        features = torch.randint(0, 256, (4, 5, 128), dtype=torch.uint8)
        for readout in ProposalSNN.READOUTS:
            model = ProposalSNN(readout=readout)
            model.initialize_from_h2(teacher)
            first = model(features)
            second = model(features)
            self.assertEqual(first.logits.shape, (5, 16))
            self.assertEqual(first.spike_rates.shape, (3,))
            self.assertTrue(torch.equal(first.logits, second.logits))
            self.assertTrue(bool(torch.all((first.spike_rates >= 0.0) & (first.spike_rates <= 1.0))))

    def test_proposal_activity_profile_preserves_forward(self) -> None:
        model = ProposalSNN(readout="mean_membrane")
        features = torch.randint(0, 4, (4, 7, 128), dtype=torch.uint8)
        expected = model(features)
        activity = profile_proposal_snn(model, features)
        self.assertTrue(torch.equal(activity.output.logits, expected.logits))
        self.assertEqual(activity.input_nonzero, int(torch.count_nonzero(features)))
        denominators = (4 * 7 * 48, 4 * 7 * 24, 4 * 7 * 16)
        measured = torch.tensor(
            [count / denominator for count, denominator in zip(activity.spike_counts, denominators)]
        )
        self.assertTrue(torch.allclose(measured, expected.spike_rates))

    def test_temporal_output_preserves_default_forward(self) -> None:
        model = ProposalSNN(readout="mean_membrane")
        features = torch.randint(0, 4, (4, 7, 128), dtype=torch.uint8)
        expected = model(features)
        temporal = model.forward_temporal(features)
        self.assertTrue(torch.equal(temporal.output.logits, expected.logits))
        self.assertTrue(torch.equal(temporal.output.spike_rates, expected.spike_rates))
        self.assertEqual(temporal.step_currents.shape, (4, 7, 16))

    def test_h2_activity_profile_preserves_logits(self) -> None:
        model = load_export(WEIGHT)
        features = np.arange(17 * 128, dtype=np.uint8).reshape(17, 128)
        expected = quantized_mlp(features, model.proposal_layers)
        activity = quantized_mlp_with_activity(features, model.proposal_layers)
        np.testing.assert_array_equal(activity.logits, expected)
        self.assertEqual(activity.activation_nonzero[0], int(np.count_nonzero(features)))
        self.assertEqual(activity.activation_values, (17 * 128, 17 * 48, 17 * 24))

    def test_signed_fixed_rounding(self) -> None:
        values = np.asarray([-8, -7, -4, -3, 3, 4, 7, 8], dtype=np.int64)
        np.testing.assert_array_equal(
            signed_round_divide(values, 4), [-2, -2, -1, -1, 1, 1, 2, 2]
        )

    def test_loa_signed_patterns_and_saturation(self) -> None:
        first = np.asarray([3, 7, -1, 127, -128], dtype=np.int64)
        second = np.asarray([3, 1, -1, 127, -128], dtype=np.int64)
        actual, saturations = loa_add_signed(first, second, bits=8, low_bits=2)
        np.testing.assert_array_equal(actual, [7, 7, -1, 127, -128])
        self.assertEqual(saturations, 2)
        exact, exact_saturations = loa_add_signed(
            first, second, bits=8, low_bits=0
        )
        np.testing.assert_array_equal(exact, [6, 8, -2, 127, -128])
        self.assertEqual(exact_saturations, 2)

    def test_fixed_balanced_tree_is_layout_and_batch_invariant(self) -> None:
        values = np.asarray(
            [[3, 3, 7, 1, -1], [1, 2, 3, 4, 5]], dtype=np.int64
        )
        expected, expected_saturations, expected_additions = fixed_balanced_add_tree(
            values, bits=24, low_bits=4
        )
        fortran = np.asfortranarray(values)
        actual, actual_saturations, actual_additions = fixed_balanced_add_tree(
            fortran, bits=24, low_bits=4
        )
        parts = [
            fixed_balanced_add_tree(row[None, :], bits=24, low_bits=4)[0]
            for row in values
        ]
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(np.concatenate(parts), expected)
        self.assertEqual(actual_saturations, expected_saturations)
        self.assertEqual(actual_additions, expected_additions)
        self.assertEqual(expected_additions, values.shape[0] * (values.shape[1] - 1))

    def test_loa_rejects_invalid_contracts(self) -> None:
        with self.assertRaises(ValueError):
            loa_add_signed(1, 2, bits=1, low_bits=0)
        with self.assertRaises(ValueError):
            loa_add_signed(1, 2, bits=24, low_bits=24)
        with self.assertRaises(ValueError):
            fixed_balanced_add_tree(
                np.empty((2, 0), dtype=np.int64), bits=24, low_bits=2
            )
        with self.assertRaises(ValueError):
            LOAPlacement(low_bits=-1)

    def test_approximate_snn_k0_matches_s1_and_accepts_empty_batch(self) -> None:
        source = ProposalSNN(beta=0.875, threshold=0.25, readout="mean_membrane")
        fixed = FixedProposalSNN.from_float(source, 4, FixedSNNConfig())
        placement = LOAPlacement(
            low_bits=0,
            synaptic_layers=(True, True, True),
            membrane_updates=(True, True, True),
        )
        exact_tree = ApproximateProposalSNN(fixed, placement)
        features = np.arange(4 * 11 * 128, dtype=np.uint8).reshape(4, 11, 128)
        expected = fixed.forward(features)
        actual = exact_tree.forward(features)
        np.testing.assert_array_equal(actual.logits_q, expected.logits_q)
        self.assertEqual(actual.spike_counts, expected.spike_counts)
        self.assertEqual(actual.accumulator_saturations, expected.accumulator_saturations)
        self.assertEqual(actual.membrane_saturations, expected.membrane_saturations)
        self.assertEqual(actual.approximate_additions, (0, 0, 0, 0, 0, 0))
        empty = exact_tree.forward(np.empty((4, 0, 128), dtype=np.uint8))
        self.assertEqual(empty.logits_q.shape, (0, 16))

    def test_approximate_snn_counts_only_selected_rtl_positions(self) -> None:
        source = ProposalSNN(beta=0.875, threshold=0.25, readout="mean_membrane")
        fixed = FixedProposalSNN.from_float(source, 4, FixedSNNConfig())
        features = np.arange(4 * 3 * 128, dtype=np.uint8).reshape(4, 3, 128)
        synaptic = ApproximateProposalSNN(
            fixed,
            LOAPlacement(low_bits=2, synaptic_layers=(True, False, False)),
        ).forward(features)
        self.assertEqual(
            synaptic.approximate_additions,
            (4 * 3 * 48 * 127, 0, 0, 0, 0, 0),
        )
        readout = ApproximateProposalSNN(
            fixed,
            LOAPlacement(low_bits=2, membrane_updates=(False, False, True)),
        ).forward(features)
        self.assertEqual(
            readout.approximate_additions,
            (0, 0, 0, 0, 0, 4 * 3 * 16),
        )
        self.assertEqual(readout.spike_counts, fixed.forward(features).spike_counts)

    def test_torch_approximate_backend_matches_numpy_golden_model(self) -> None:
        source = ProposalSNN(beta=0.875, threshold=0.25, readout="mean_membrane")
        fixed = FixedProposalSNN.from_float(source, 4, FixedSNNConfig())
        features = np.arange(4 * 5 * 128, dtype=np.uint8).reshape(4, 5, 128)
        for placement in (
            LOAPlacement(
                low_bits=4,
                synaptic_layers=(True, True, True),
            ),
            LOAPlacement(
                low_bits=4,
                membrane_updates=(True, True, True),
            ),
        ):
            expected = ApproximateProposalSNN(fixed, placement).forward(features)
            devices = [torch.device("cpu")]
            if torch.cuda.is_available():
                devices.append(torch.device("cuda"))
            for device in devices:
                with self.subTest(placement=placement, device=device):
                    actual = TorchApproximateProposalSNN(
                        fixed, placement, device
                    ).forward(features)
                    np.testing.assert_array_equal(actual.logits_q, expected.logits_q)
                    self.assertEqual(actual.spike_counts, expected.spike_counts)
                    self.assertEqual(
                        actual.accumulator_saturations,
                        expected.accumulator_saturations,
                    )
                    self.assertEqual(
                        actual.membrane_saturations, expected.membrane_saturations
                    )
                    self.assertEqual(
                        actual.approximate_additions,
                        expected.approximate_additions,
                    )

    def test_approximate_snn_export_round_trip_is_bit_exact(self) -> None:
        source = ProposalSNN(beta=0.875, threshold=0.25, readout="mean_membrane")
        fixed = FixedProposalSNN.from_float(source, 4, FixedSNNConfig())
        placement = LOAPlacement(
            low_bits=4,
            synaptic_layers=(True, False, True),
            membrane_updates=(True, True, True),
        )
        model = ApproximateProposalSNN(fixed, placement)
        features = np.arange(4 * 7 * 128, dtype=np.uint8).reshape(4, 7, 128)
        expected = model.forward(features)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s3.npz"
            export_approximate_proposal_snn(path, model, {"model_id": "s3-test"})
            restored, configuration, metadata = load_approximate_proposal_snn(path)
        actual = restored.forward(features)
        np.testing.assert_array_equal(actual.logits_q, expected.logits_q)
        self.assertEqual(actual.approximate_additions, expected.approximate_additions)
        self.assertEqual(configuration["low_bits"], 4)
        self.assertEqual(configuration["synaptic_layers"], [True, False, True])
        self.assertEqual(metadata["model_id"], "s3-test")

    def test_fixed_h2_matches_float_scale_closely(self) -> None:
        source = load_export(WEIGHT)
        fixed = FixedH2(source, FixedH2Config(accumulator_bits=32))
        features = np.arange(31 * 128, dtype=np.uint8).reshape(31, 128)
        expected = quantized_mlp(features, source.proposal_layers)
        actual = fixed.proposal(features)
        self.assertEqual(actual.logits_q.dtype, np.int64)
        self.assertEqual(actual.accumulator_saturations, (0, 0, 0))
        np.testing.assert_allclose(actual.logits, expected, atol=2e-3, rtol=0.0)

    def test_fixed_snn_export_round_trip_is_bit_exact(self) -> None:
        source = ProposalSNN(beta=0.875, threshold=0.25, readout="mean_membrane")
        configuration = FixedSNNConfig(
            accumulator_bits=24, membrane_bits=24, membrane_fraction_bits=12
        )
        model = FixedProposalSNN.from_float(source, 4, configuration)
        features = np.arange(4 * 13 * 128, dtype=np.uint8).reshape(4, 13, 128)
        expected = model.forward(features)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixed_s0p.npz"
            export_fixed_proposal_snn(path, model, {"model_id": "unit-test"})
            restored, restored_configuration, metadata = load_fixed_proposal_snn(path)
        actual = restored.forward(features)
        np.testing.assert_array_equal(actual.logits_q, expected.logits_q)
        self.assertEqual(actual.spike_counts, expected.spike_counts)
        self.assertEqual(restored_configuration["membrane_bits"], 24)
        self.assertEqual(metadata["model_id"], "unit-test")

    def test_pot_modes_and_export_are_bit_exact(self) -> None:
        source = ProposalSNN(beta=0.875, threshold=0.25, readout="mean_membrane")
        fixed = FixedProposalSNN.from_float(source, 4, FixedSNNConfig())
        features = np.arange(4 * 11 * 128, dtype=np.uint8).reshape(4, 11, 128)
        int8_only = PoTProposalSNN(
            fixed, ("int8", "int8", "int8"), -6, 3
        )
        np.testing.assert_array_equal(
            int8_only.forward(features).logits_q, fixed.forward(features).logits_q
        )
        full = PoTProposalSNN(fixed, ("pot", "pot", "pot"), -6, 3)
        for layer in full.weight_statistics():
            self.assertEqual(layer["mode"], "pot")
            self.assertEqual(
                int(layer["zero_weights"]) + int(layer["nonzero_weights"]),
                int(layer["weights"]),
            )
        expected = full.forward(features)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pot_s0p.npz"
            export_pot_proposal_snn(path, full, {"model_id": "pot-unit-test"})
            restored, configuration, metadata = load_pot_proposal_snn(path)
        actual = restored.forward(features)
        np.testing.assert_array_equal(actual.logits_q, expected.logits_q)
        self.assertEqual(configuration["layer_modes"], ["pot", "pot", "pot"])
        self.assertEqual(metadata["model_id"], "pot-unit-test")

    def test_s0p_npz_export_round_trip(self) -> None:
        model = ProposalSNN(beta=0.875, threshold=0.25, readout="mean_membrane")
        configuration = {
            "beta": 0.875,
            "threshold": 0.25,
            "surrogate_slope": 10.0,
            "readout": "mean_membrane",
            "time_steps": 4,
        }
        metadata = {"model_id": "unit-test", "checkpoint_epoch": 3}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s0p.npz"
            export_proposal_snn(path, model, configuration, metadata)
            restored, restored_configuration, restored_metadata = (
                load_proposal_snn_export(path)
            )
        self.assertEqual(restored_configuration, configuration)
        self.assertEqual(restored_metadata, metadata)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_refiner_shape_initialization_and_state_isolation(self) -> None:
        _, teacher = load_exported_models(WEIGHT)
        model = RefinerSNN()
        model.initialize_from_h2(teacher)
        features = torch.randint(0, 256, (4, 9, 32), dtype=torch.uint8)
        first = model(features)
        second = model(features)
        self.assertEqual(first.raw_logits.shape, (9, 4))
        self.assertTrue(torch.equal(first.raw_logits, second.raw_logits))
        self.assertTrue(
            bool(torch.all((first.spike_rates >= 0.0) & (first.spike_rates <= 1.0)))
        )

    def test_refiner_batch_events_are_independent(self) -> None:
        _, teacher = load_exported_models(WEIGHT)
        model = RefinerSNN()
        model.initialize_from_h2(teacher)
        features = torch.randint(0, 256, (4, 2, 32), dtype=torch.uint8)
        combined = model(features)
        first = model(features[:, :1])
        second = model(features[:, 1:2])
        self.assertTrue(torch.allclose(combined.raw_logits[:1], first.raw_logits))
        self.assertTrue(torch.allclose(combined.raw_logits[1:2], second.raw_logits))

    def test_s0r_npz_export_round_trip(self) -> None:
        model = RefinerSNN(beta=0.875, threshold=0.25, readout="mean_membrane")
        configuration = {
            "time_steps": 4,
            "beta": 0.875,
            "threshold": 0.25,
            "surrogate_slope": 10.0,
            "readout": "mean_membrane",
        }
        metadata = {
            "model_id": "unit-test",
            "checkpoint_epoch": 12,
            "event_threshold": 0.64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "s0r.npz"
            export_refiner_snn(path, model, configuration, metadata)
            restored, restored_configuration, restored_metadata = (
                load_refiner_snn_export(path)
            )
        self.assertEqual(restored_configuration, configuration)
        self.assertEqual(restored_metadata, metadata)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, restored.state_dict()[key]))

    def test_s0r_window_flattening_preserves_event_order(self) -> None:
        windows = [
            S0RWindow(
                path=Path("first.npz"),
                window=0,
                event_indices=np.asarray([4, 7, 9]),
                features=np.full((3, 32), 11, dtype=np.uint8),
                labels=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
                teacher_probability=np.asarray([0.1, 0.9, 0.2], dtype=np.float32),
            ),
            S0RWindow(
                path=Path("second.npz"),
                window=1,
                event_indices=np.asarray([2]),
                features=np.full((1, 32), 22, dtype=np.uint8),
                labels=np.asarray([1.0], dtype=np.float32),
                teacher_probability=np.asarray([0.8], dtype=np.float32),
            ),
        ]
        features, labels, teacher = flatten_s0r_windows(windows)
        self.assertEqual(features.shape, (4, 32))
        np.testing.assert_array_equal(labels, [0.0, 1.0, 0.0, 1.0])
        np.testing.assert_allclose(teacher, [0.1, 0.9, 0.2, 0.8])
        self.assertTrue(np.all(features[:3] == 11))
        self.assertTrue(np.all(features[3:] == 22))


class TemporalRepresentationTests(unittest.TestCase):
    def test_boundaries_and_feature_shape(self) -> None:
        events = temporal_events()
        windows, micro = temporal_window_indices(events, 4)
        np.testing.assert_array_equal(windows, [0, 0, 0, 0, 1, 1])
        np.testing.assert_array_equal(micro, [0, 0, 1, 3, 0, 0])
        features = build_temporal_patch_features(events, [0, 1], 4)
        self.assertEqual(features.shape, (2, 4, 336, 128))
        self.assertTrue(np.all(np.any(features[0], axis=(1, 2)) == [True, True, False, True]))
        self.assertTrue(np.any(features[1, 0]))

    def test_invalid_time_steps(self) -> None:
        with self.assertRaises(ValueError):
            temporal_window_indices(temporal_events(), 5)

    def test_temporal_target_boundaries(self) -> None:
        events = temporal_events()
        events["label"] = 1
        targets = build_temporal_patch_targets(events, [0, 1], 4)
        self.assertEqual(targets.shape, (2, 4, 336, 16))
        self.assertTrue(np.any(targets[0, 0]))
        self.assertTrue(np.any(targets[0, 1]))
        self.assertTrue(np.any(targets[0, 3]))
        self.assertTrue(np.any(targets[1, 0]))
        self.assertFalse(np.any(targets[0, 2]))
        self.assertFalse(np.any(targets[1, 1:]))

        empty = np.zeros(0, dtype=events.dtype)
        empty_targets = build_temporal_patch_targets(empty, [0, 1], 4)
        self.assertEqual(empty_targets.shape, targets.shape)
        self.assertFalse(np.any(empty_targets))

    def test_vectorized_representation_matches_original_loops(self) -> None:
        rng = np.random.default_rng(42)
        events = np.zeros(400, dtype=temporal_events().dtype)
        events["x"] = rng.integers(0, 346, len(events))
        events["y"] = rng.integers(0, 260, len(events))
        events["t"] = np.sort(rng.uniform(0.0, 199.999, len(events)))
        events["p"] = rng.integers(0, 2, len(events))
        events["label"] = rng.integers(0, 2, len(events))
        windows = [1, 0, 1]
        np.testing.assert_array_equal(
            build_patch_features(events, windows),
            reference_patch_features(events, windows),
        )
        np.testing.assert_array_equal(
            build_temporal_patch_features(events, windows, 4),
            reference_patch_features(events, windows, 4),
        )
        np.testing.assert_array_equal(
            build_patch_targets(events, windows),
            reference_patch_targets(events, windows),
        )


class S0PContractTests(unittest.TestCase):

    def test_differentiable_patch_fusion_matches_numpy(self) -> None:
        generator = np.random.default_rng(2026)
        patches = generator.random((3, PATCHES_PER_WINDOW, PATCH_OUTPUT_DIM)).astype(
            np.float32
        )
        expected = patch_scores_to_heatmaps(patches)
        actual = patch_probabilities_to_heatmaps(
            torch.from_numpy(patches[:, None])
        )[:, 0].numpy()
        np.testing.assert_array_equal(actual, expected)

    def test_stc_weights_are_detached_and_loss_has_gradient(self) -> None:
        heatmaps = torch.zeros((2, 4, 34, 44), requires_grad=True)
        released = stc_weights(heatmaps, 1, 3, variant="released")
        paper = stc_weights(heatmaps, 1, 3, variant="paper")
        self.assertFalse(released.requires_grad)
        self.assertTrue(torch.all(released == 0.5))
        self.assertTrue(torch.all(paper == 0.5))

        currents = torch.zeros(
            (4, 2 * PATCHES_PER_WINDOW, PATCH_OUTPUT_DIM), requires_grad=True
        )
        targets = torch.zeros((2, 4, PATCHES_PER_WINDOW, PATCH_OUTPUT_DIM))
        targets[0, 0, 0, 0] = 1.0
        losses = proposal_stc_loss(currents, targets, 1, 3)
        self.assertTrue(torch.isfinite(losses["loss"]))
        losses["loss"].backward()
        gradient = currents.grad
        self.assertIsNotNone(gradient)
        if gradient is None:
            raise AssertionError("STC loss gradient is absent")
        self.assertGreater(float(gradient.abs().sum()), 0.0)

        extreme = torch.full_like(currents.detach(), 1000.0, requires_grad=True)
        extreme_loss = proposal_stc_loss(extreme, targets, 1, 3)["loss"]
        self.assertTrue(torch.isfinite(extreme_loss))

    def test_zero_stc_weight_preserves_optimizer_update(self) -> None:
        torch.manual_seed(2026)
        baseline = ProposalSNN(readout="mean_membrane")
        candidate = copy.deepcopy(baseline)
        features = torch.randint(
            0, 4, (4, PATCHES_PER_WINDOW, PATCH_FEATURE_DIM), dtype=torch.uint8
        )
        targets = torch.zeros((PATCHES_PER_WINDOW, PATCH_OUTPUT_DIM))
        targets[0, 0] = 1.0
        teacher = torch.zeros_like(targets)
        temporal_targets = torch.zeros(
            (1, 4, PATCHES_PER_WINDOW, PATCH_OUTPUT_DIM)
        )
        temporal_targets[0, 0, 0, 0] = 1.0
        baseline_optimizer = torch.optim.SGD(baseline.parameters(), lr=1.0e-3)
        candidate_optimizer = torch.optim.SGD(candidate.parameters(), lr=1.0e-3)

        baseline_output = baseline(features)
        baseline_loss = s0p_loss(
            baseline_output.logits,
            targets,
            teacher,
            baseline_output.spike_rates,
            distillation_weight=0.5,
        )["total"]
        baseline_loss.backward()
        baseline_optimizer.step()

        candidate_output = candidate.forward_temporal(features)
        candidate_base = s0p_loss(
            candidate_output.output.logits,
            targets,
            teacher,
            candidate_output.output.spike_rates,
            distillation_weight=0.5,
        )["total"]
        candidate_stc = proposal_stc_loss(
            candidate_output.step_currents, temporal_targets, 1, 3
        )["loss"]
        (candidate_base + 0.0 * candidate_stc).backward()
        candidate_optimizer.step()

        self.assertTrue(torch.equal(candidate_base, baseline_loss))
        for baseline_value, candidate_value in zip(
            baseline.state_dict().values(), candidate.state_dict().values()
        ):
            self.assertTrue(torch.equal(baseline_value, candidate_value))

    def test_stc_blocks_cover_each_core_window_once(self) -> None:
        ranges = stc_block_ranges(80, core_windows=4, halo_windows=1)
        covered = []
        for load_start, load_stop, core_start, core_stop in ranges:
            covered.extend(range(load_start + core_start, load_start + core_stop))
            self.assertLessEqual(load_stop - load_start, 6)
        self.assertEqual(covered, list(range(80)))

    def test_binary_rank_auc_handles_ties(self) -> None:
        self.assertEqual(
            binary_rank_auc(
                np.asarray([0.1, 0.2, 0.8, 0.9]),
                np.asarray([0, 0, 1, 1]),
            ),
            1.0,
        )

    def test_positive_patch_sampling(self) -> None:
        targets = torch.zeros((100, 16))
        targets[:5, 3] = 1.0
        first = sample_patch_indices(
            targets, 0.25, torch.Generator().manual_seed(2026)
        )
        second = sample_patch_indices(
            targets, 0.25, torch.Generator().manual_seed(2026)
        )
        self.assertTrue(torch.equal(first, second))
        self.assertEqual(len(first), len(targets))
        sampled_positive = torch.any(targets[first] > 0.5, dim=1)
        self.assertEqual(int(sampled_positive.sum()), 25)

    def test_roles_never_expose_test(self) -> None:
        paths = role_file_paths(INDEX, "train_main")
        self.assertEqual(len(paths), 79)
        self.assertTrue(all(path.parent.name == "train" for path in paths))
        with self.assertRaises(ValueError):
            role_file_paths(INDEX, "test")

    def test_combined_loss_has_student_gradient(self) -> None:
        logits = torch.zeros((3, 16), requires_grad=True)
        targets = torch.zeros_like(logits)
        teacher = torch.ones_like(logits)
        rates = torch.tensor([0.0, 0.1, 0.5], requires_grad=True)
        losses = s0p_loss(logits, targets, teacher, rates)
        losses["total"].backward()
        logits_gradient = logits.grad
        rates_gradient = rates.grad
        self.assertIsNotNone(logits_gradient)
        self.assertIsNotNone(rates_gradient)
        if logits_gradient is None or rates_gradient is None:
            raise AssertionError("S0-P loss gradient is absent")
        self.assertGreater(float(logits_gradient.abs().sum()), 0.0)
        self.assertGreater(float(rates_gradient.abs().sum()), 0.0)
        self.assertEqual(load_export(WEIGHT).recipe, "D1")

    def test_s0r_loss_has_gradient(self) -> None:
        raw = torch.zeros((6, 4), requires_grad=True)
        labels = torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        teacher = torch.full_like(labels, 0.6)
        rates = torch.tensor([0.05, 0.10, 0.02], requires_grad=True)
        losses = s0r_loss(raw, labels, teacher, rates)
        losses["total"].backward()
        gradient = raw.grad
        self.assertIsNotNone(gradient)
        if gradient is None:
            raise AssertionError("S0-R gradient is absent")
        self.assertGreater(float(gradient.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
