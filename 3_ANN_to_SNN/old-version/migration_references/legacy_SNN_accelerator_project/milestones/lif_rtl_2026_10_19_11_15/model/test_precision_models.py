import math
import unittest

from precision_models import (
    bits_to_float,
    float_lif_step,
    float_to_bits,
    int8_lif_step,
    mitchell_multiply,
    mitchell_multiply_bits,
)


class MitchellMultiplierTests(unittest.TestCase):
    def test_binary32_known_values(self):
        self.assertEqual(mitchell_multiply_bits(0x3FC00000, 0x3FC00000, "fp32"), 0x40000000)
        self.assertEqual(mitchell_multiply_bits(0x40000000, 0x3F000000, "fp32"), 0x3F800000)
        self.assertEqual(mitchell_multiply_bits(0xBFC00000, 0x3FC00000, "fp32"), 0xC0000000)

    def test_binary16_known_values(self):
        self.assertEqual(mitchell_multiply_bits(0x3E00, 0x3E00, "fp16"), 0x4000)
        self.assertEqual(mitchell_multiply_bits(0x4000, 0x3800, "fp16"), 0x3C00)
        self.assertEqual(mitchell_multiply_bits(0xBE00, 0x3E00, "fp16"), 0xC000)

    def test_special_values_and_ftz(self):
        self.assertEqual(mitchell_multiply_bits(0x00000000, 0x7F800000, "fp32"), 0x7FC00000)
        self.assertEqual(mitchell_multiply_bits(0x0000, 0x7C00, "fp16"), 0x7E00)
        self.assertEqual(mitchell_multiply_bits(0x00000001, 0x3F800000, "fp32"), 0x00000000)
        self.assertEqual(mitchell_multiply_bits(0x7F7FFFFF, 0x40000000, "fp32"), 0x7F800000)

    def test_power_of_two_products_are_exact(self):
        for fmt in ("fp16", "fp32"):
            for a, b in ((0.5, 2.0), (4.0, 0.25), (-8.0, 0.5)):
                self.assertEqual(mitchell_multiply(a, b, fmt), a * b)

    def test_maximum_characteristic_error_point(self):
        for fmt in ("fp16", "fp32"):
            exact = 1.5 * 1.5
            approximate = mitchell_multiply(1.5, 1.5, fmt)
            self.assertEqual(approximate, 2.0)
            self.assertAlmostEqual((exact - approximate) / exact, 1.0 / 9.0)


class PrecisionLifTests(unittest.TestCase):
    def test_float_approximation_changes_candidate(self):
        for fmt in ("fp16", "fp32"):
            exact = float_lif_step(1.5, 0.0, alpha=1.5, threshold=3.0, fmt=fmt)
            approximate = float_lif_step(
                1.5, 0.0, alpha=1.5, threshold=3.0, fmt=fmt, approximate=True
            )
            self.assertEqual(exact.candidate, 2.25)
            self.assertEqual(approximate.candidate, 2.0)
            self.assertFalse(exact.spike)
            self.assertFalse(approximate.spike)

    def test_float_hard_reset(self):
        for fmt in ("fp16", "fp32"):
            result = float_lif_step(0.75, 0.25, alpha=1.0, threshold=1.0, fmt=fmt)
            self.assertTrue(result.spike)
            self.assertEqual(result.membrane, 0.0)

    def test_int8_exact_and_approximate(self):
        exact = int8_lif_step(3, 7, threshold=100, leak_shift=0)
        approximate = int8_lif_step(
            3, 7, threshold=100, leak_shift=0, approximate=True, approx_lsb_bits=2
        )
        self.assertEqual(exact.candidate, 10)
        self.assertEqual(approximate.candidate, 7)

    def test_bit_conversions(self):
        self.assertEqual(float_to_bits(1.5, "fp32"), 0x3FC00000)
        self.assertEqual(float_to_bits(1.5, "fp16"), 0x3E00)
        self.assertEqual(bits_to_float(0xBF800000, "fp32"), -1.0)
        self.assertTrue(math.isinf(bits_to_float(0x7C00, "fp16")))


if __name__ == "__main__":
    unittest.main()
