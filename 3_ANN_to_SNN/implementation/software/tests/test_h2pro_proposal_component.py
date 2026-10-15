import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).absolute().parents[2]


class ComponentCalibrationTests(unittest.TestCase):
    def test_fast_components_and_histogram_frontier_match_original(self):
        run = subprocess.run([sys.executable, '-B', str(ROOT/'software/scripts/h2pro_proposal_component.py'), 'check'],
                             cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                             env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(run.returncode, 0, run.stderr+run.stdout)
        report = json.loads(run.stdout)
        self.assertEqual(report['component_masks_exact'], 60)
        self.assertTrue(report['histogram_frontier_exact'])
        self.assertFalse(report['training_started'])


if __name__ == '__main__':
    unittest.main()
