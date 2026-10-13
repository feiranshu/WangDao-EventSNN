import unittest

import numpy as np

from snn_accel.quantization import decode_signed_pot, quantize_signed_pot


class QuantizationTests(unittest.TestCase):
    def test_signed_power_of_two_round_trip(self):
        weights = np.array([0.0, 0.24, -0.51, 3.8], dtype=np.float32)
        codes = quantize_signed_pot(weights)
        quantized = decode_signed_pot(codes)
        np.testing.assert_array_equal(quantized, [0.0, 0.25, -0.5, 4.0])
        self.assertEqual(len(codes.packed_bytes()), len(weights))

    def test_packed_code_layout(self):
        codes = quantize_signed_pot(np.array([0.0, -0.5, 1.0]))
        packed = np.frombuffer(codes.packed_bytes(), dtype=np.uint8)
        self.assertTrue(packed[0] & 0x80)
        self.assertTrue(packed[1] & 0x40)
        self.assertEqual(packed[2] & 0x0F, 8)


if __name__ == "__main__":
    unittest.main()

