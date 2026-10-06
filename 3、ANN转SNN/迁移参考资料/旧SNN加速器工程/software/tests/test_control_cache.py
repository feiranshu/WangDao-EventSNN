import unittest

from snn_accel.cache_model import StateTileCacheModel
from snn_accel.mode_control import HysteresisController, PrecisionMode


class ControlAndCacheTests(unittest.TestCase):
    def test_hysteresis(self):
        control = HysteresisController(10, 20)
        self.assertEqual(control.update(5), PrecisionMode.APPROXIMATE)
        self.assertEqual(control.update(15), PrecisionMode.APPROXIMATE)
        self.assertEqual(control.update(25), PrecisionMode.EXACT)
        self.assertEqual(control.update(15), PrecisionMode.EXACT)

    def test_prefetch_produces_neighbor_hit(self):
        cache = StateTileCacheModel(width=64, height=64, tile_size=8, capacity_tiles=16)
        self.assertFalse(cache.access(16, 16, prefetch_3x3=True))
        self.assertTrue(cache.access(24, 16, prefetch_3x3=True))
        self.assertGreater(cache.stats.external_reads, 0)


if __name__ == "__main__":
    unittest.main()

