"""Manifest, failure-reporting and frozen first-layer vector contracts."""

from __future__ import annotations

import json
import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))

from host_tools.rtl_verification.generate_rtl_vectors import (engine_trace, first_layer_result, frozen_first_layer,
    hardware_calibration_path, second_layer_trace, generate_two_layers, lif_step)
from host_tools.rtl_verification.run_rtl_simulation import check_rows, parse_success, run_command, validate_manifest


class RTLManifestTests(unittest.TestCase):
    def test_existing_unit_and_engine_manifests(self) -> None:
        for suite in ("units", "engine", "two_layers"):
            manifest, files = validate_manifest(ROOT / "pl_design/test_vectors", suite)
            self.assertEqual(manifest["status"], "ok")
            self.assertTrue(files)

    def test_modified_vector_and_weight_are_rejected(self) -> None:
        for filename in ("fixed_add.txt", "layer1_weight_f4.txt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                for source in (ROOT / "pl_design/test_vectors").glob("*.txt"):
                    shutil.copyfile(source, directory / source.name)
                shutil.copyfile(ROOT / "pl_design/test_vectors/manifest.json", directory / "manifest.json")
                (directory / filename).write_text("00\n", encoding="ascii")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    validate_manifest(directory, "units")

    def test_bad_rows_are_rejected_even_with_matching_hash(self) -> None:
        schema = [(16, 255), (10, 1)]
        for content, count in (("", 1), ("00 0\n", 2), ("GG 0\n", 1),
                               ("100 0\n", 1), ("00 2\n", 1), ("00 0 extra\n", 1),
                               ("00 0\n00 0\n", 1), ("00 0\n", 0)):
            with self.subTest(content=content, count=count), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "vector.txt"
                path.write_text(content, encoding="ascii")
                with self.assertRaises(ValueError):
                    check_rows(path, count, schema)

    def test_success_requires_unique_correct_marker_and_no_fatal(self) -> None:
        marker = "WD_PASS top=tb_fixed_add records=144"
        self.assertEqual(parse_success(marker, "tb_fixed_add", 144)["records"], 144)
        for output in ("", "Simulation finished", marker.replace("144", "0"),
                       marker + "\n" + marker, "Fatal: failure\n" + marker,
                       marker.replace("tb_fixed_add", "wrong_top")):
            with self.subTest(output=output), self.assertRaises(RuntimeError):
                parse_success(output, "tb_fixed_add", 144)

    def test_process_failure_and_timeout_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "Tool exit"):
                run_command([sys.executable, "-c", "raise SystemExit(7)"], directory,
                            "failure.log", time.monotonic() + 5)
            with self.assertRaises(TimeoutError):
                run_command([sys.executable, "-c", "import time; time.sleep(10)"], directory,
                            "timeout.log", time.monotonic() + 0.3)

    def test_engine_default_id_failure_json_and_no_overwrite(self) -> None:
        from host_tools.rtl_verification.run_rtl_simulation import main
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            output = directory / "result.json"
            argv = ["run_rtl_simulation.py", "--suite", "engine", "--vector-directory",
                    str(directory / "missing"), "--output", str(output)]
            with patch.object(sys, "argv", argv), redirect_stdout(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 1)
            before = output.read_bytes()
            result = json.loads(before)
            self.assertEqual(result["experiment_id"], "RTL-FIRST-LAYER-X2019-v1")
            self.assertEqual(result["status"], "failed")
            with patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 2)
            self.assertEqual(output.read_bytes(), before)


class FirstLayerGoldenTests(unittest.TestCase):
    def test_existing_engine_vectors_are_not_overwritten(self) -> None:
        from host_tools.rtl_verification.generate_rtl_vectors import generate_engine
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = directory / "first_layer_manifest.json"
            manifest.write_text("{}", encoding="ascii")
            with self.assertRaises(FileExistsError):
                generate_engine(directory, 42, 2)
            self.assertEqual(manifest.read_text(), "{}")

    def test_full_channel_trace_matches_scalar_reference(self) -> None:
        exact, approximate = frozen_first_layer()
        rng = np.random.default_rng(718)
        features = rng.integers(0, 256, (1, 4, 128), dtype=np.uint8)
        for model, low_bits in ((exact, 0), (approximate, 2)):
            trace = engine_trace(features, model, low_bits)
            for channel in range(48):
                membrane = 0
                for step in range(4):
                    membrane, spike, tree, saturated = first_layer_result(
                        features[0, step], model.weights[0][channel],
                        int(model.weight_fraction_bits[0][channel]), membrane, low_bits)
                    self.assertEqual(trace[0, step, channel].tolist(),
                        [(membrane & 0xFFFFFF) | (spike << 24), tree, saturated])

    def test_frozen_export_mismatch_is_rejected(self) -> None:
        import host_tools.rtl_verification.generate_rtl_vectors as vectors
        original = vectors.sha256_file
        with patch.object(vectors, "sha256_file", side_effect=lambda path:
                          "0" * 64 if path == vectors.S3_WEIGHT_PATH else original(path)):
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                frozen_first_layer()

    def test_calibration_role_must_be_preserved(self) -> None:
        import host_tools.rtl_verification.generate_rtl_vectors as vectors
        index = json.loads(vectors.INDEX_PATH.read_text(encoding="utf-8"))
        index["roles"]["hw_calibration"].remove("train_022.npz")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "index.json"
            path.write_text(json.dumps(index), encoding="utf-8")
            with patch.object(vectors, "INDEX_PATH", path):
                with self.assertRaisesRegex(ValueError, "must remain"):
                    hardware_calibration_path()

    def test_grouped_tree_matches_full_tree(self) -> None:
        from wangdao.fixed import fixed_balanced_add_tree
        rng = np.random.default_rng(810)
        # Include saturating internal nodes: grouping must preserve their order too.
        values = rng.integers(-(1 << 23), 1 << 23, (9, 128), dtype=np.int64)
        for k in (0, 2):
            expected, expected_sats, _ = fixed_balanced_add_tree(values, 24, k)
            partial, first_sats, _ = fixed_balanced_add_tree(values.reshape(9, 16, 8), 24, k)
            actual, second_sats, _ = fixed_balanced_add_tree(partial, 24, k)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(first_sats + second_sats, expected_sats)


class SecondLayerGoldenTests(unittest.TestCase):
    def test_binary_trace_matches_scalar_tree_and_lif(self) -> None:
        from wangdao.fixed import fixed_balanced_add_tree
        exact, approximate = frozen_first_layer()
        spikes = np.random.default_rng(519).integers(0, 2, (3, 4, 48), dtype=np.uint8)
        spikes[0] = 0
        spikes[1] = 1
        for model, k in ((exact, 0), (approximate, 2)):
            trace = second_layer_trace(spikes, model, k)
            for patch_index in range(3):
                for channel in range(24):
                    membrane = 0
                    for step in range(4):
                        products = spikes[patch_index, step].astype(np.int64) * model.weights[1][channel]
                        summed, tree_sats, _ = fixed_balanced_add_tree(products, 24, 0)
                        current = int(summed) << (12 - int(model.weight_fraction_bits[1][channel]))
                        membrane, spike, saturated = lif_step(membrane, current, True, k)
                        self.assertEqual(trace[patch_index, step, channel].tolist(),
                            [(membrane & 0xFFFFFF) | (spike << 24), tree_sats, saturated])
        # Aligned currents have five or more zero low bits; k=2 membrane LOA
        # is exactly equal to addition for identical second-layer spike inputs.
        np.testing.assert_array_equal(second_layer_trace(spikes, exact, 0),
                                      second_layer_trace(spikes, approximate, 2))

    def test_reject_nonbinary_and_incorrect_shapes(self) -> None:
        exact, _ = frozen_first_layer()
        for spikes in (np.full((1, 4, 48), 2), np.full((1, 4, 48), -1), np.zeros((1, 48)), np.zeros((1, 4, 47))):
            with self.subTest(shape=spikes.shape), self.assertRaises(ValueError):
                second_layer_trace(spikes, exact, 0)

    def test_second_layer_frozen_weights_must_match(self) -> None:
        exact, approximate = frozen_first_layer()
        approximate.weights = tuple(weight.copy() for weight in approximate.weights)
        approximate.weights[1][0, 0] ^= 1
        with tempfile.TemporaryDirectory() as temporary:
            with patch("host_tools.rtl_verification.generate_rtl_vectors.frozen_first_layer", return_value=(exact, approximate)):
                with self.assertRaisesRegex(ValueError, "weights/scaling differ"):
                    generate_two_layers(Path(temporary), 42, 0)

    def test_two_layer_vectors_are_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "two_layer_manifest.json"
            path.write_text("{}", encoding="ascii")
            with self.assertRaises(FileExistsError):
                generate_two_layers(path.parent, 42, 0)
            self.assertEqual(path.read_text(), "{}")

    def test_second_layer_rom_and_intermediate_corruption_rejected(self) -> None:
        for filename in ("layer2_bank0.txt", "layer2_shifts.txt", "two_layer_intermediate.txt"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                manifest, files = validate_manifest(ROOT / "pl_design/test_vectors", "two_layers")
                for source in files:
                    shutil.copyfile(source, directory / source.name)
                (directory / "two_layer_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                (directory / filename).write_text("00\n", encoding="ascii")
                with self.assertRaisesRegex(ValueError, "hash mismatch"):
                    validate_manifest(directory, "two_layers")


if __name__ == "__main__":
    unittest.main()
