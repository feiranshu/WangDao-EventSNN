import unittest

import numpy as np

from snn_accel.aer import PacketType, unpack_packet
from snn_accel.events import EventTable, SliceConfig
from snn_accel.pipeline import preprocess


class SlicingTests(unittest.TestCase):
    def setUp(self):
        self.events = EventTable(
            x=np.array([10, 11, 12, 13, 14, 15]),
            y=np.array([20, 21, 22, 23, 24, 25]),
            t=np.array([0, 100, 200, 300, 400, 500]),
            p=np.array([0, 1, 0, 1, 0, 1]),
            label=np.array([0, 1, 1, 0, 1, 1]),
            name=np.array([0, 7, 7, 0, 9, 9]),
        )
        self.config = SliceConfig(target_events=3, min_duration_us=0, max_duration_us=1000, bbox_margin=1)

    def test_bounded_count_and_targets(self):
        result = preprocess(self.events, self.config)
        self.assertEqual([item.stop - item.start for item in result.slices], [3, 3])
        first = result.slices[0]
        self.assertEqual(len(first.targets), 1)
        self.assertEqual((first.targets[0].x_min, first.targets[0].x_max), (10, 13))
        self.assertTrue(result.tlast[-1])
        self.assertEqual(unpack_packet(result.packets[-1]).packet_type, PacketType.SLICE_END)

    def test_labels_do_not_change_boundaries(self):
        baseline = preprocess(self.events, self.config)
        changed = EventTable(
            x=self.events.x,
            y=self.events.y,
            t=self.events.t,
            p=self.events.p,
            label=1 - self.events.label,
            name=np.arange(len(self.events)),
        )
        other = preprocess(changed, self.config)
        self.assertEqual([(s.start, s.stop) for s in baseline.slices], [(s.start, s.stop) for s in other.slices])
        np.testing.assert_array_equal(baseline.packets, other.packets)

    def test_fixed_time_mode(self):
        config = SliceConfig(mode="fixed_time", fixed_duration_us=199, max_duration_us=1000)
        result = preprocess(self.events, config)
        self.assertEqual([item.stop - item.start for item in result.slices], [2, 2, 2])


if __name__ == "__main__":
    unittest.main()

