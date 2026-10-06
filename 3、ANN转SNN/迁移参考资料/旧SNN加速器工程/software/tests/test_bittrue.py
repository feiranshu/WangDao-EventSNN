import unittest

import numpy as np

from snn_accel.bittrue import approximate_add, exact_add, lif_step, required_accumulator_bits


class BitTrueTests(unittest.TestCase):
    def test_exact_saturation(self):
        np.testing.assert_array_equal(exact_add([120, -120], [20, -20], 8), [127, -128])

    def test_approximate_low_bits_are_or_without_carry(self):
        # 7 (0111) + 3 (0011), K=2: high 1+0, low 3|3 -> 0111 = 7.
        self.assertEqual(int(approximate_add(7, 3, 8, 2)), 7)
        self.assertEqual(int(approximate_add(120, 120, 8, 2)), 127)

    def test_lif_spike_reset_and_leak(self):
        result = lif_step(np.array([80, 30]), np.array([30, 0]), threshold=100, width=16, leak_shift=2)
        np.testing.assert_array_equal(result.spike, [False, False])
        np.testing.assert_array_equal(result.membrane, [90, 23])
        spike = lif_step(90, 20, threshold=100, width=16)
        self.assertTrue(bool(spike.spike))
        self.assertEqual(int(spike.membrane), 0)

    def test_width_calibration_is_bounded(self):
        self.assertEqual(required_accumulator_bits(0), 16)
        self.assertEqual(required_accumulator_bits(1 << 30), 24)


if __name__ == "__main__":
    unittest.main()

