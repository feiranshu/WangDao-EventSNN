import tempfile
import unittest
from pathlib import Path

import numpy as np

from snn_accel.events import load_ev_uav_npz


class EventLoadingTests(unittest.TestCase):
    def test_timestamp_unit_conversion_and_stable_sort(self):
        matrix = np.array([
            [1, 2, 2.0, 1, 0, 0],
            [3, 4, 1.0, 0, 1, 7],
        ])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "events.npz"
            np.savez(path, ev=matrix)
            events = load_ev_uav_npz(path, timestamp_unit="ms")
        np.testing.assert_array_equal(events.t, [1000, 2000])
        np.testing.assert_array_equal(events.x, [3, 1])


if __name__ == "__main__":
    unittest.main()
