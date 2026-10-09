"""Measurement integrity and directed mask-address coverage for first-layer skipping."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))
from host_tools.rtl_verification.generate_rtl_vectors import zero_group_cases
from host_tools.rtl_verification.run_rtl_simulation import check_cycle_measurements


class ZeroSkipTests(unittest.TestCase):
    def test_mask_address_and_lane_coverage(self) -> None:
        cases, names = zero_group_cases()
        self.assertEqual(len(cases), 138)
        self.assertEqual(len(set(names)), len(cases))
        for index, case in enumerate(cases[:128]):
            expected = (index // 2) * 8 + (7 if index % 2 else 0)
            self.assertEqual(np.flatnonzero(case).tolist(), [expected])
        for lane, case in enumerate(cases[128:136]):
            self.assertEqual(np.flatnonzero(case).tolist(), list(range(lane, 512, 8)))
        self.assertEqual(np.count_nonzero(cases[-1]), 256)
        np.testing.assert_array_equal(cases[-1] + cases[-2], np.full((4, 128), 255))

    def test_measured_cycles_use_actual_input_groups(self) -> None:
        values = np.zeros((3, 512), dtype=np.uint8)
        values[1] = 255
        values[2, -1] = 1
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "inputs.txt"
            path.write_text("\n".join(" ".join(f"{v:02X}" for v in row) for row in values) + "\n", encoding="ascii")
            for top, baseline in (("tb_first_layer_engine", 7425), ("tb_proposal_engine", 9601)):
                for skip in (False, True):
                    expected = [baseline - (zero * 48 if skip else 0) for zero in (64, 0, 63)]
                    output = "\n".join(f"WD_CYCLE patch={i} cycles={v}" for i, v in enumerate(expected)) + "\n"
                    result = check_cycle_measurements(output, path, top, 3, skip)
                    self.assertEqual(result["measured_total_cycles"], sum(expected))
                    self.assertEqual([r["zero_groups"] for r in result["rows"]], [64, 0, 63])
                    for broken in ("", output + output.splitlines()[0] + "\n",
                                   output + "WD_CYCLE malformed\n",
                                   "\n".join(output.splitlines()[:-1]),
                                   output.replace("patch=1", "patch=0"),
                                   output.replace(f"cycles={expected[0]}", f"cycles={expected[0] + 1}"),
                                   "\n".join(reversed(output.splitlines()))):
                        with self.assertRaises(ValueError):
                            check_cycle_measurements(broken, path, top, 3, skip)


if __name__ == "__main__":
    unittest.main()
