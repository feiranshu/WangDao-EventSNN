import unittest

import numpy as np

from snn_accel.calibration import choose_density_hysteresis, event_count_candidates


class CalibrationTests(unittest.TestCase):
    def test_candidate_quantiles(self):
        values = np.array([1, 2, 3, 4, 100])
        self.assertEqual(event_count_candidates(values), {"p25": 2, "p50": 3, "p75": 4})

    def test_hysteresis_thresholds_are_ordered(self):
        self.assertEqual(choose_density_hysteresis(np.ones(4, dtype=int)), (1, 2))


if __name__ == "__main__":
    unittest.main()
