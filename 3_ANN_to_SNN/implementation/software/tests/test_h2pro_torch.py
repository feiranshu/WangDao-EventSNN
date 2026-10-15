"""Numerical parity and gradient existence, without optimizer or training runs."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from wangdao.h2pro_snn.transport import MODES, SparseTransportLIF, TransportConfig
from wangdao.h2pro_snn.torch_cell import TransportLIF


class TorchTransportTests(unittest.TestCase):
    def test_dense_torch_matches_sparse_event_reference(self):
        config = TransportConfig(rows=2, columns=2, channels=4, local_channels=2)
        for mode in MODES:
            with self.subTest(mode=mode):
                rng = np.random.default_rng(7)
                dense, sparse = TransportLIF(config, mode), SparseTransportLIF(config, mode)
                state = torch.zeros((4, 4), dtype=torch.float64)
                timestamp = 0.
                for _ in range(30):
                    dt = float(rng.choice([0., 1., 10., 25.]))
                    timestamp += dt
                    currents = rng.uniform(-.5, 1.2, (4, 4))
                    inactive = int(rng.integers(4))
                    currents[inactive] = 0.
                    destinations = rng.integers(4, size=4)
                    pulses, state = dense(torch.from_numpy(currents), state, dt, torch.from_numpy(destinations))
                    out = sparse.step(timestamp, {i: currents[i] for i in range(4) if i != inactive},
                                      dict(enumerate(destinations)))
                    reference = np.zeros((4, 4), dtype=np.int64)
                    for cell, values in out.items():
                        reference[cell] = values
                    np.testing.assert_array_equal(pulses.detach().numpy(), reference)
                    np.testing.assert_allclose(state.detach().numpy(), sparse.snapshot(), atol=1e-12, rtol=1e-12)

    def test_gradient_exists_for_all_routes_without_parameter_update(self):
        config = TransportConfig(rows=1, columns=2, channels=2, local_channels=1)
        for mode in MODES:
            with self.subTest(mode=mode):
                cell = TransportLIF(config, mode)
                current = torch.tensor([[.3, .6], [.4, .2]], dtype=torch.float64, requires_grad=True)
                state = torch.full((2, 2), .2, dtype=torch.float64, requires_grad=True)
                spikes, residual = cell(current, state, 10., torch.tensor([1, 0]))
                (spikes.sum() + .2*residual.sum()).backward()
                self.assertTrue(torch.isfinite(current.grad).all())
                self.assertTrue(torch.isfinite(state.grad).all())
                self.assertGreater(current.grad.abs().sum().item(), 0.)

    def test_invalid_time_and_duplicate_destination_shape(self):
        cell = TransportLIF(TransportConfig(rows=1, columns=2, channels=2, local_channels=1))
        zero = torch.zeros((2, 2))
        for dt in (-1., float('nan')):
            with self.assertRaises(ValueError):
                cell(zero, zero, dt)
        with self.assertRaises(ValueError):
            cell(zero, zero, 1., torch.tensor([[0, 1], [0, 1]]))


if __name__ == '__main__':
    unittest.main()
