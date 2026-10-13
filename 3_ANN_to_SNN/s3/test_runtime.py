"""Numerical gates for S3 state-resident execution and matched loss."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).absolute().parent
sys.path.insert(0, str(HERE.parent / "s2"))
from core import ResearchSNN, SNNRunner  # noqa: E402
sys.path.insert(0, str(HERE))
from runtime import DeviceSNNRunner  # noqa: E402


class S3RuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.features = np.array([
            [25] * 36, [100] * 36, [40] * 36, [80] * 36,
            [20] * 36, [60] * 36], dtype=np.uint8)
        self.x = np.array([0, 4, 0, 4, 0, 4], dtype=np.int16)
        self.y = np.array([0, 0, 0, 0, 0, 0], dtype=np.int16)
        self.t = np.array([1, 1, 4, 6, 102, 103], dtype=np.float64)

    def test_prediction_and_persistent_state_match_s2(self):
        model = ResearchSNN().eval()
        old = SNNRunner(model, 0)
        new = DeviceSNNRunner(model, 0)
        with torch.no_grad():
            for segment in (slice(0, 4), slice(4, 6)):
                _, expected = old.run_window(self.features[segment], self.x[segment],
                                             self.y[segment], self.t[segment])
                actual = new.run_window(self.features[segment], self.x[segment],
                                        self.y[segment], self.t[segment])
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
            for previous, current in zip(old.state, new.state):
                np.testing.assert_allclose(current.detach().numpy(), previous, rtol=0, atol=1e-7)
        self.assertEqual(new.counters()["candidate_events"], 6)

    def test_weighted_loss_and_gradient_match_s2(self):
        old_model = ResearchSNN()
        new_model = ResearchSNN()
        new_model.load_state_dict(old_model.state_dict())
        target = np.array([0, 1, 0, 1], dtype=np.uint8)
        weight = torch.tensor([1.0, 3.0])
        old = SNNRunner(old_model, 0)
        new = DeviceSNNRunner(new_model, 0)
        expected, _ = old.run_window(self.features[:4], self.x[:4], self.y[:4],
                                     self.t[:4], target, weight)
        weighted, plain, mass, n = new.run_window(self.features[:4], self.x[:4],
                                                   self.y[:4], self.t[:4], target, weight)
        self.assertEqual(n, 4)
        self.assertAlmostEqual(float(weighted.detach()), float(expected.detach()), places=6)
        self.assertAlmostEqual(float(mass.detach()), 8.0, places=6)
        self.assertTrue(float(plain.detach()) > 0)
        expected.backward()
        weighted.backward()
        for a, b in zip(old_model.parameters(), new_model.parameters()):
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=1e-6)

    def test_window_reset_removes_only_neural_history(self):
        model = ResearchSNN().eval()
        reset = DeviceSNNRunner(model, 0, mode="window_reset")
        fresh = DeviceSNNRunner(model, 0, mode="window_reset")
        with torch.no_grad():
            reset.run_window(self.features[:4], self.x[:4], self.y[:4], self.t[:4])
            actual = reset.run_window(self.features[4:], self.x[4:], self.y[4:], self.t[4:])
            expected = fresh.run_window(self.features[4:], self.x[4:], self.y[4:], self.t[4:])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
