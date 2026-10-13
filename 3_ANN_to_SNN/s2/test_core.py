import unittest

import numpy as np
import torch

from core import ResearchSNN, SNNRunner, SurrogateSpike, THRESHOLD


class S2CoreTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = ResearchSNN()

    def test_surrogate_forward_and_gradient(self):
        value = torch.tensor([-0.25, 0.0, 1.5], requires_grad=True)
        spike = SurrogateSpike.apply(value)
        self.assertEqual(spike.tolist(), [0.0, 1.0, 1.0])
        spike.sum().backward()
        np.testing.assert_allclose(value.grad.numpy(), [0.75, 1.0, 0.0])

    def test_same_timestamp_one_spike_per_update_and_edge(self):
        with torch.no_grad():
            for layer in (self.model.fc1, self.model.fc2, self.model.fc3):
                layer.weight.zero_()
            self.model.fc1.weight[0, 0] = 1.0
        runner = SNNRunner(self.model, 0.0)
        features = np.zeros((2, 36), dtype=np.uint8)
        features[:, 0] = 255
        _, probability = runner.run_window(features, np.array([345, 345]), np.array([259, 259]),
                                           np.array([10.0, 10.0]))
        self.assertEqual(len(probability), 2)
        self.assertEqual(runner.counts["spikes_l1"], 2)
        self.assertAlmostEqual(float(runner.state[0][-1, 0]), 2 * (1.0 - THRESHOLD))
        self.assertEqual(runner.counts["positive_dt"], 1)
        cost = runner.cost()
        self.assertEqual(cost["refiner_dense_mac"], 2 * 3392)
        self.assertEqual(cost["refiner_effective_input_mac"], 2 * 48)
        self.assertEqual(cost["refiner_binary_ac"], 2 * 32)
        self.assertEqual(cost["voltage_and_trace_bytes"], 65 * 87 * 88 * 4)

    def test_lazy_decay_and_reset(self):
        with torch.no_grad():
            for layer in (self.model.fc1, self.model.fc2, self.model.fc3):
                layer.weight.zero_()
        runner = SNNRunner(self.model, 0.0)
        runner.state[0][0, 16] = 1.0
        runner.run_window(np.empty((0, 36), np.uint8), np.empty(0, np.int16),
                          np.empty(0, np.int16), np.empty(0))
        self.assertEqual(float(runner.state[0][0, 16]), 1.0)
        runner.run_window(np.zeros((1, 36), np.uint8), np.array([0]), np.array([0]),
                          np.array([25.0]))
        self.assertAlmostEqual(float(runner.state[0][0, 16]), float(np.exp(-1.0)), places=6)
        runner.reset(100.0)
        self.assertEqual(int(np.count_nonzero(runner.state[0])), 0)
        self.assertEqual(runner.counts["events"], 0)

    def test_readout_combines_trace_and_residual_membrane(self):
        with torch.no_grad():
            for layer in (self.model.fc1, self.model.fc2, self.model.fc3):
                layer.weight.zero_()
            self.model.trace_gain.fill_(2.0)
            self.model.mem_gain.fill_(3.0)
            self.model.readout_bias.fill_(0.1)
        state = (torch.zeros(1, 48), torch.zeros(1, 32),
                 torch.full((1, 4), 0.1), torch.full((1, 4), 0.2))
        logits, next_state, spikes = self.model.step(torch.zeros(1, 36), torch.zeros(1), state)
        self.assertAlmostEqual(float(logits[0, 0]), 2 * 0.2 + 3 * (0.1 / THRESHOLD) + 0.1)
        self.assertEqual(int(torch.count_nonzero(spikes[2])), 0)
        self.assertAlmostEqual(float(next_state[3][0, 0]), 0.2)

    def test_chunking_equivalence_and_gradient(self):
        features = np.zeros((3, 36), dtype=np.uint8)
        features[:, :4] = 255
        x = np.array([0, 4, 0]); y = np.array([0, 0, 0]); t = np.array([1.0, 1.0, 20.0])
        full = SNNRunner(self.model, 0.0)
        _, a = full.run_window(features, x, y, t)
        split = SNNRunner(self.model, 0.0)
        _, b0 = split.run_window(features[:2], x[:2], y[:2], t[:2])
        _, b1 = split.run_window(features[2:], x[2:], y[2:], t[2:])
        np.testing.assert_allclose(a, np.concatenate([b0, b1]), atol=1e-7)
        for left, right in zip(full.state, split.state):
            np.testing.assert_allclose(left, right, atol=1e-6)
        training = SNNRunner(self.model, 0.0)
        loss, _ = training.run_window(features, x, y, t, np.array([1, 0, 1]))
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in self.model.parameters()))


if __name__ == "__main__":
    unittest.main()
