"""Transport order, charge conservation, physical time and stream boundaries."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1]))
from wangdao.h2pro_snn.transport import MODES, SparseTransportLIF, TransportConfig, mechanism_demo


def config(**kwargs):
    values = dict(rows=1, columns=3, channels=2, local_channels=1,
                  tau_short_ms=20., tau_long_ms=100., threshold=1.)
    values.update(kwargs)
    return TransportConfig(**values)


class TransportTests(unittest.TestCase):
    def test_weak_moving_evidence_counterexample(self):
        result = mechanism_demo()
        self.assertEqual(result['primary_metrics'], {'fixed': 0, 'spike_route': 0, 'membrane_route': 1})
        self.assertAlmostEqual(result['results']['membrane_route']['residual'][2][0], .084)
        self.assertTrue(result['supplied_routes_use_ground_truth'])

    def test_static_evidence_does_not_need_transport(self):
        totals = []
        for mode in MODES:
            model = SparseTransportLIF(config(channels=1, local_channels=0), mode)
            for k in range(3):
                model.step(k*10., {0: [.4]})
            totals.append(model.ledger.unit_spikes)
        self.assertEqual(totals, [1, 1, 1])

    def test_short_memory_stays_and_long_memory_moves(self):
        model = SparseTransportLIF(config(), 'membrane_route')
        model.step(0., {0: [.3, .6]})
        model.step(0., {}, {0: 1})
        np.testing.assert_allclose(model.snapshot(), [[.3, 0.], [0., .6], [0., 0.]])

    def test_collisions_sum_distinct_sources_without_copying(self):
        model = SparseTransportLIF(config(threshold=10.), 'membrane_route')
        model.step(0., {0: [0., .4], 2: [0., .6]})
        model.step(0., {}, {0: 1, 2: 1})
        np.testing.assert_allclose(model.snapshot()[:, 1], [0., 1., 0.])
        model.step(0., {}, {0: 1, 2: 1})
        self.assertAlmostEqual(model.snapshot().sum(), 1.)

    def test_cycles_use_simultaneous_source_snapshot(self):
        model = SparseTransportLIF(config(threshold=10.), 'membrane_route')
        model.step(0., {0: [0., .4], 1: [0., .6]})
        model.step(0., {}, {0: 1, 1: 0})
        np.testing.assert_allclose(model.snapshot()[:, 1], [.6, .4, 0.])

    def test_only_emitted_pulses_move_in_spike_control(self):
        model = SparseTransportLIF(config(), 'spike_route')
        pulses = model.step(0., {0: [0., 1.2]}, {0: 1})
        np.testing.assert_array_equal(pulses[1], [0, 1])
        self.assertAlmostEqual(model.snapshot()[0, 1], .2)
        self.assertEqual(model.snapshot()[1, 1], 0.)

    def test_every_unit_in_a_burst_is_counted(self):
        model = SparseTransportLIF(config())
        pulses = model.step(0., {0: [2.4, 3.2]})
        np.testing.assert_array_equal(pulses[0], [2, 3])
        self.assertEqual(model.ledger.unit_spikes, 5)
        np.testing.assert_allclose(model.snapshot()[0], [.4, .2])

    def test_zero_spike_updates_have_nonzero_cost(self):
        model = SparseTransportLIF(config())
        self.assertEqual(model.step(10., {0: [.1, .1]}), {})
        self.assertEqual(model.ledger.unit_spikes, 0)
        self.assertGreater(model.ledger.scalar_mul, 0)
        self.assertGreater(model.ledger.state_write_bytes, 0)

    def test_no_empty_grid_sweep_in_sparse_runtime(self):
        model = SparseTransportLIF(config(rows=20, columns=20))
        model.step(0., {5: [.2, .2]})
        before = model.ledger.state_scalar_updates
        model.step(1000., {})
        self.assertEqual(model.ledger.state_scalar_updates, before)
        np.testing.assert_allclose(model.snapshot()[5], [.2*np.exp(-50), .2*np.exp(-10)])

    def test_time_units_and_same_timestamp(self):
        model = SparseTransportLIF(config())
        model.step(0., {0: [.4, .4]})
        model.step(0., {0: [.1, .1]})
        np.testing.assert_allclose(model.snapshot()[0], [.5, .5])
        model.step(20., {})
        np.testing.assert_allclose(model.snapshot()[0], [.5*np.exp(-1), .5*np.exp(-.2)])

    def test_reset_clears_sequence_state_and_counters(self):
        model = SparseTransportLIF(config())
        model.step(5., {0: [2., 3.]})
        model.reset(100.)
        self.assertEqual(model.snapshot().sum(), 0.)
        self.assertEqual(model.ledger.unit_spikes, 0)
        self.assertEqual(model.watermark_ms, 100.)

    def test_invalid_input_is_transactional(self):
        model = SparseTransportLIF(config(), 'membrane_route')
        model.step(10., {0: [.2, .3]})
        before = model.snapshot().copy()
        for timestamp, current, routes in [(9., {}, {}), (np.nan, {}, {}),
                (11., {0: [np.nan, 0.]}, {}), (11., {}, {0: 2}), (11., {3: [0., 0.]}, {})]:
            with self.subTest(timestamp=timestamp), self.assertRaises(ValueError):
                model.step(timestamp, current, routes)
            np.testing.assert_array_equal(before, model.snapshot())
            self.assertEqual(model.watermark_ms, 10.)

    def test_linear_index_does_not_wrap_grid_edges(self):
        model = SparseTransportLIF(config(rows=2, columns=3), 'membrane_route')
        with self.assertRaises(ValueError):
            model.step(0., {}, {2: 3})

    def test_packet_grouping_preserves_outputs_state_and_cost(self):
        drives = [(0., {0: [.1, .4]}, {}), (10., {1: [.2, .4]}, {0: 1}),
                  (20., {2: [.2, .4]}, {1: 2}), (20., {2: [.3, .4]}, {})]
        results = []
        for chunks in ([drives], [drives[:1], drives[1:3], drives[3:]]):
            model = SparseTransportLIF(config(), 'membrane_route')
            outputs = []
            for chunk in chunks:
                for drive in chunk:
                    outputs.append({k: v.tolist() for k, v in model.step(*drive).items()})
            results.append((outputs, model.snapshot(), model.cost_report()))
        self.assertEqual(results[0][0], results[1][0])
        np.testing.assert_array_equal(results[0][1], results[1][1])
        self.assertEqual(results[0][2], results[1][2])

    def test_future_updates_do_not_mutate_published_spikes(self):
        model = SparseTransportLIF(config(), 'membrane_route')
        published = model.step(0., {0: [1.2, 1.3]})
        expected = published[0].copy()
        model.step(10., {0: [-.7, 2.4]}, {0: 1})
        np.testing.assert_array_equal(published[0], expected)

    def test_no_new_charge_from_transport(self):
        model = SparseTransportLIF(config(threshold=10.), 'membrane_route')
        model.step(0., {0: [-.2, .4], 1: [.3, -.1], 2: [0., .2]})
        before = np.abs(model.snapshot()).sum()
        model.step(10., {}, {0: 1, 1: 2})
        self.assertLessEqual(np.abs(model.snapshot()).sum(), before)


if __name__ == '__main__':
    unittest.main()
