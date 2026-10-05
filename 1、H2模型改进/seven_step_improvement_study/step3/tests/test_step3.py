from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_step3 import _read_reused_stage, parser as step3_parser  # noqa: E402
from collect_step3_results import include as include_result  # noqa: E402
from step3_common import (  # noqa: E402
    CandidateCorpus,
    SEQUENCE_DENSITY_EPS,
    export_refiner_npz,
    load_refiner_npz,
    make_float_refiner,
    predict_quantized,
    quantized_layers_from_model,
)
from step3_diagnostics import PROBE_VARIANTS, probe_features  # noqa: E402
from step3_training import (  # noqa: E402
    CorrectedStratifiedSampler,
    FullPassSampler,
    PROGRESS_FIELDS,
    _save_checkpoint,
    candidate_bce,
    load_checkpoint_package,
)


def synthetic_record() -> dict:
    labels = np.asarray([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
    high = np.asarray([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.uint8)
    return {
        "file": "train_synthetic.npz",
        "features": np.tile(np.arange(32, dtype=np.uint8), (8, 1)),
        "labels": labels,
        "baseline_probability": np.linspace(0.1, 0.8, 8, dtype=np.float32),
        "windows": np.asarray([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.uint8),
        "high_density": high,
        "legacy_window_high_density": high,
        "window_rate_eps": np.asarray([25000, 25000, 22000, 22000, 5000, 5000, 4000, 4000], dtype=np.float32),
        "hard_negative": np.asarray([0, 1, 0, 0, 0, 1, 0, 0], dtype=np.uint8),
        "hard_positive": np.asarray([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8),
        "history": np.tile(np.arange(4, dtype=np.uint8), (8, 1)),
        "x": np.arange(8, dtype=np.uint16),
        "y": np.zeros(8, dtype=np.uint16),
        "t_ms": np.arange(8, dtype=np.int32),
        "sequence_rate_eps": np.asarray(25000.0),
        "event_count": np.asarray(8),
    }


class Step3Tests(unittest.TestCase):
    def test_new_density_boundary_is_frozen(self):
        self.assertEqual(SEQUENCE_DENSITY_EPS, 20000.0)

    def test_full_pass_has_no_omission_or_repeat(self):
        corpus = CandidateCorpus([synthetic_record()])
        sampler = FullPassSampler(corpus, 42)
        batches = sampler.batches(1, 3)
        report = sampler.report(batches)
        self.assertEqual(report["draws"], 8)
        self.assertEqual(report["unique_candidates"], 8)
        self.assertEqual(report["omitted"], 0)
        self.assertEqual(report["repeat_fraction"], 0.0)

    def test_corrected_sampler_has_disjoint_quotas_and_weights(self):
        corpus = CandidateCorpus([synthetic_record()])
        sampler = CorrectedStratifiedSampler(corpus, 42)
        batch = sampler.batches(1, 4096)[0]
        selected, weights = batch
        self.assertEqual(len(selected), 4096)
        high = corpus.high_density[selected].astype(bool)
        positive = corpus.labels[selected] == 1
        self.assertEqual(int((high & positive).sum()), 512)
        self.assertEqual(int((high & ~positive).sum()), 1536)
        self.assertEqual(int((~high & positive).sum()), 512)
        self.assertEqual(int((~high & ~positive).sum()), 1536)
        # All four synthetic strata contain 25% of the natural corpus.  The
        # sampled positive strata are 12.5%, negatives 37.5%.
        self.assertAlmostEqual(float(weights[high & positive].mean()), 2.0)
        self.assertAlmostEqual(float(weights[high & ~positive].mean()), 2.0 / 3.0, places=6)

    def test_importance_weighting_recovers_natural_stratum_mean(self):
        corpus = CandidateCorpus([synthetic_record()])
        sampler = CorrectedStratifiedSampler(corpus, 7)
        selected, weights = sampler.batches(3, 4096)[0]
        losses = np.where(corpus.labels[selected] == 1, 3.0, 1.0)
        corrected = float(np.mean(losses * weights))
        natural = float(np.mean(np.where(corpus.labels == 1, 3.0, 1.0)))
        self.assertAlmostEqual(corrected, natural, places=6)

    def test_bce_weight_changes_only_sample_contribution(self):
        raw = torch.tensor([[0.1, 0.2, 0.4, 0.3], [0.4, 0.1, 0.2, 0.6]])
        labels = torch.tensor([1.0, 0.0])
        base = candidate_bce(raw, labels)
        equal = candidate_bce(raw, labels, torch.ones(2))
        torch.testing.assert_close(base, equal, rtol=0, atol=0)
        self.assertNotEqual(float(base), float(candidate_bce(raw, labels, torch.tensor([2.0, 0.5]))))

    def test_exact_int8_export_round_trip(self):
        model = make_float_refiner(42)
        features = np.tile(np.arange(32, dtype=np.uint8), (16, 1))
        expected = predict_quantized(quantized_layers_from_model(model), features)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "refiner.npz"
            export_refiner_npz(path, model)
            actual = predict_quantized(load_refiner_npz(path), features)
        np.testing.assert_array_equal(expected, actual)

    def test_checkpoint_contains_optimizer_scheduler_and_zero_count(self):
        model = make_float_refiner(42)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-5)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _epoch: 1.0)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "zero.pt"
            _save_checkpoint(
                path, model, optimizer, scheduler,
                {"model_id": "A3-1", "seed": 42, "epoch": 0, "update_count": 0, "scheduler_step_count": 0},
            )
            package = load_checkpoint_package(path)
        self.assertIn("optimizer_state_dict", package)
        self.assertIn("scheduler_state_dict", package)
        self.assertEqual(package["metadata"]["update_count"], 0)

    def test_probe_feature_shapes(self):
        record = synthetic_record()
        bounds = (np.log1p(4000.0), np.log1p(25000.0))
        expected = {
            "base32": 32,
            "plus_density": 33,
            "plus_history": 36,
            "drop_sequence_time": 31,
            "drop_coordinates": 30,
        }
        self.assertEqual(set(expected), set(PROBE_VARIANTS))
        for variant, columns in expected.items():
            self.assertEqual(probe_features(record, variant, bounds).shape, (8, columns))

    def test_continuation_cli_requires_explicit_checkpoint_contract(self):
        args = step3_parser().parse_args([
            "train", "--model", "A3-2", "--output", "continued",
            "--resume-checkpoint", "epoch_020.pt", "--epoch-offset", "20",
            "--epochs", "10", "--run",
        ])
        self.assertEqual(args.epoch_offset, 20)
        self.assertEqual(args.epochs, 10)
        self.assertEqual(args.learning_rate, 1.0e-5)

    def test_resume_rejects_smoke_formal_mismatch_and_incomplete_diagnostic(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "summary.json"
            path.write_text(
                '{"status":"complete","smoke":true,"validation_used":false,"test_used":false}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "does not belong to formal mode"):
                _read_reused_stage(path, "A3-1", "complete", False)
            path.write_text(
                '{"status":"running","smoke":false,"validation_used":false,"test_used":false}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "expected status=complete"):
                _read_reused_stage(path, "D3-0", "complete", False)

    def test_progress_csv_contract_includes_both_hidden_layer_saturation(self):
        for layer in (1, 2):
            for region in ("zero", "one", "linear"):
                self.assertIn(f"hidden{layer}_{region}_fraction", PROGRESS_FIELDS)

    def test_continuations_are_inside_formal_result_tree(self):
        server = (ROOT / "README_SERVER.md").read_text(encoding="utf-8")
        self.assertIn("step3_formal_20261004/continuations/A3-1-seed42-continue10", server)
        self.assertIn("step3_formal_20261004/continuations/A3-2-seed42-continue10", server)
        self.assertIn("collect_step3_results.py runs/step3_formal_20261004", server)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            formal = Path(directory) / "step3_formal"
            continuation = formal / "continuations" / "A3-1-seed42-continue10" / "summary.json"
            continuation.parent.mkdir(parents=True)
            continuation.write_text("{}\n", encoding="utf-8")
            self.assertTrue(include_result(continuation, formal))

    def test_readme_commands_have_no_shell_continuation(self):
        for name in ("README.md", "README_SERVER.md"):
            path = ROOT / name
            if path.exists():
                for line in path.read_text(encoding="utf-8").splitlines():
                    self.assertFalse(line.rstrip().endswith("\\"), f"continued command in {name}: {line}")


if __name__ == "__main__":
    unittest.main()
