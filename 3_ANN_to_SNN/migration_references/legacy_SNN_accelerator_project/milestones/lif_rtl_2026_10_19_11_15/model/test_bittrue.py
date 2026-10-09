import random
import unittest

from bittrue import (
    approximate_add,
    exact_add,
    int8_mac,
    lif_step,
    pot_accumulate,
    pot_contribution,
    signed_bounds,
)


class ArithmeticTests(unittest.TestCase):
    def test_exact_saturation(self):
        self.assertEqual(exact_add(120, 20, 8), (127, True))
        self.assertEqual(exact_add(-120, -20, 8), (-128, True))
        self.assertEqual(exact_add(10, -3, 8), (7, False))

    def test_loa_definition(self):
        self.assertEqual(approximate_add(7, 3, 8, 2), (7, False))
        self.assertEqual(approximate_add(120, 120, 8, 2), (127, True))

    def test_random_exact_add_never_wraps(self):
        rng = random.Random(24307140011)
        minimum, maximum = signed_bounds(20)
        for _ in range(5000):
            a = rng.randint(minimum, maximum)
            b = rng.randint(minimum, maximum)
            result, saturated = exact_add(a, b, 20)
            self.assertLessEqual(minimum, result)
            self.assertGreaterEqual(maximum, result)
            self.assertEqual(saturated, not minimum <= a + b <= maximum)

    def test_int8_mac(self):
        accumulator, saturated = int8_mac(0, 10, 3)
        self.assertEqual((accumulator, saturated), (30, False))
        accumulator, saturated = int8_mac(accumulator, -4, 2)
        self.assertEqual((accumulator, saturated), (22, False))


class PotAccumulatorTests(unittest.TestCase):
    def test_q8_contributions(self):
        self.assertEqual(
            pot_contribution(
                spike=True,
                weight_zero=False,
                weight_negative=False,
                exponent=-1,
                layer_scale=0,
            ),
            (128, False),
        )
        self.assertEqual(
            pot_contribution(
                spike=True,
                weight_zero=False,
                weight_negative=True,
                exponent=0,
                layer_scale=0,
            ),
            (-256, False),
        )

    def test_condition_suppresses_update(self):
        for spike, zero in ((False, False), (True, True), (False, True)):
            self.assertEqual(
                pot_accumulate(
                    123,
                    spike=spike,
                    weight_zero=zero,
                    weight_negative=False,
                    exponent=7,
                    layer_scale=0,
                ),
                (123, False),
            )

    def test_underflow_and_saturation_are_reported(self):
        self.assertEqual(
            pot_contribution(
                spike=True,
                weight_zero=False,
                weight_negative=False,
                exponent=-8,
                layer_scale=-8,
            ),
            (0, True),
        )
        value, saturated = pot_contribution(
            spike=True,
            weight_zero=False,
            weight_negative=False,
            exponent=7,
            layer_scale=8,
        )
        self.assertEqual(value, (1 << 19) - 1)
        self.assertTrue(saturated)


class LifTests(unittest.TestCase):
    def test_hard_reset_and_leak(self):
        first = lif_step(80, 30, threshold=100, width=16, leak_shift=2)
        self.assertEqual(first, first.__class__(90, False, False))
        fired = lif_step(90, 20, threshold=100, width=16, leak_shift=0)
        self.assertEqual(fired, fired.__class__(0, True, False))

    def test_subtractive_reset(self):
        fired = lif_step(90, 30, threshold=100, width=16, leak_shift=0, subtractive_reset=True)
        self.assertEqual(fired.membrane, 20)
        self.assertTrue(fired.spike)

    def test_soft_reset_has_priority(self):
        reset = lif_step(500, 500, threshold=100, width=16, soft_reset=True)
        self.assertEqual((reset.membrane, reset.spike), (0, False))

    def test_approximate_mode_differs_by_definition(self):
        exact = lif_step(0, 3, threshold=100, width=8, leak_shift=0)
        exact = lif_step(exact.membrane, 7, threshold=100, width=8, leak_shift=0)
        approximate = lif_step(0, 3, threshold=100, width=8, leak_shift=0, approximate=True, approx_lsb_bits=2)
        approximate = lif_step(
            approximate.membrane,
            7,
            threshold=100,
            width=8,
            leak_shift=0,
            approximate=True,
            approx_lsb_bits=2,
        )
        self.assertEqual(exact.membrane, 10)
        self.assertEqual(approximate.membrane, 7)


if __name__ == "__main__":
    unittest.main()

