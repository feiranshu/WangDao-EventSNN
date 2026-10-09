"""Composite window mapping, extended captures and replay isolation checks."""
import copy
import io
import json
from contextlib import redirect_stderr
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "model_runtime"))
from host_tools.rtl_verification.replay_rtl_windows import WINDOWS, main, validate_outputs, window_slices
from wangdao.rtl_bridge import ProposalCapture, read_proposal_capture


class MultiwindowTests(unittest.TestCase):
    def test_long_formal_run_requires_explicit_user_execution(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "experiments/workspace") as temporary:
            folder = Path(temporary)
            pilot = folder / "pilot.json"
            pilot.write_text(json.dumps({"status": "pilot", "experiment_id": "RTL-MULTIWINDOW-PILOT",
                                         "estimated_formal_seconds": 700}), encoding="utf-8")
            argv = ["replay_rtl_windows.py", "--mode", "formal", "--run-directory", str(folder / "new"),
                    "--output", str(folder / "result.json"), "--pilot-result", str(pilot)]
            with patch.object(sys, "argv", argv), redirect_stderr(io.StringIO()) as stream:
                with self.assertRaises(SystemExit) as caught:
                    main()
            self.assertEqual(caught.exception.code, 2)
            self.assertIn("user should execute", stream.getvalue())
            self.assertFalse((folder / "new").exists())

    def test_contiguous_complete_windows_and_repeat(self) -> None:
        rows = [dict(file=f, window=w, patch_start=10 + i * 336, patches=336) for i, (f, w) in enumerate(WINDOWS)]
        parts = window_slices(rows, 1690)
        self.assertEqual((parts[0].start, parts[-1].stop), (10, 1690))
        self.assertEqual(WINDOWS[0], WINDOWS[-1])
        for key, value in (("patch_start", 345), ("patches", 335), ("window", 80)):
            bad = copy.deepcopy(rows)
            bad[1][key] = value
            with self.assertRaises(ValueError):
                window_slices(bad, 1690)
        with self.assertRaises(ValueError):
            window_slices(rows, 1691)

    def test_large_capture_and_window_boundary_corruption(self) -> None:
        count = 682 * 16
        lines = ["WD_CAPTURE version=1 patches=682 channels=16"]
        lines += [f"{i // 16} {i % 16} {i + 1} 00000001 0 0 FFFFFFFF 0 0" for i in range(count)]
        lines.append(f"WD_CAPTURE_END records={count}")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "capture.txt"
            path.write_text("\n".join(lines) + "\n", encoding="ascii")
            result = read_proposal_capture(path, 682)
            self.assertEqual(result.logits_q.shape, (2, 682, 16))
            self.assertTrue(np.all(result.logits_q[1] == -1))
            lines[346 * 16 + 1] = lines[346 * 16]
            path.write_text("\n".join(lines) + "\n", encoding="ascii")
            with self.assertRaises(ValueError):
                read_proposal_capture(path, 682)
            with self.assertRaises(ValueError):
                read_proposal_capture(path, 2049)

    def test_capture_variants_and_golden_isolation_reject_mismatch(self) -> None:
        rows = [dict(file="train_093.npz", window=70, patch_start=10 + i * 336, patches=336) for i in range(2)]
        manifest = {"configuration": {"windows": rows, "patches": 682}}
        value = np.zeros((2, 682, 16), dtype=np.int64)
        capture = ProposalCapture(value, value.copy(), value.copy(), np.zeros((682, 16), dtype=np.int64))
        cache = {("train_093.npz", 70): (None, [np.zeros((336, 16, 6), dtype=np.int64)])}
        metadata = {"proposal_threshold": .51, "event_threshold": .28}
        def route(_events, _window, logits, *_args):
            return {"logits_q": logits, "event_indices": np.array([], dtype=np.int64),
                    "candidates": np.array([], dtype=np.int64), "refiner_saturations": (0, 0, 0)}
        with patch("host_tools.rtl_verification.replay_rtl_windows.route_window", side_effect=route):
            variants = {name: copy.deepcopy(capture) for name in ("fixed", "skip", "stressed")}
            result = validate_outputs(manifest, variants, {"train_093.npz": None}, None, metadata, cache)
            self.assertEqual(len(result), 2)
            variants["stressed"].logits_q[0, 346, 0] = 1
            with self.assertRaisesRegex(ValueError, "actual capture mismatch"):
                validate_outputs(manifest, variants, {"train_093.npz": None}, None, metadata, cache)
            for variant in variants.values():
                variant.logits_q[0, 346, 0] = 1
            with self.assertRaisesRegex(ValueError, "pipeline mismatch"):
                validate_outputs(manifest, variants, {"train_093.npz": None}, None, metadata, cache)


if __name__ == "__main__":
    unittest.main()
