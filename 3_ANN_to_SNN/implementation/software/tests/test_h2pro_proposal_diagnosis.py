from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from wangdao.h2pro_proposal_diagnosis_v1.common import load_bundle, save_bundle, lock_directory

ROOT = Path(__file__).absolute().parents[2]


class ProposalDiagnosisTests(unittest.TestCase):
    def test_isolated_interventions_against_unchanged_pipeline(self):
        run = subprocess.run([sys.executable, '-B', str(ROOT/'software/scripts/h2pro_proposal_diagnose.py'), 'check'],
                             cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                             env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(run.returncode, 0, run.stderr+run.stdout[-2000:])
        report = json.loads(run.stdout)
        self.assertEqual(report['status'], 'passed')
        self.assertGreaterEqual(report['check_count'], 11)
        self.assertFalse(report['training_started'])

    def test_bundle_integrity_and_exact_prediction_storage(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results/runs') as name:
            stem = Path(name)/'point'
            arrays = {'candidate': np.array([True, False]), 'probabilities': np.array([.7, 0], np.float32)}
            save_bundle(stem, arrays, {'role': 'train_main'})
            actual, receipt = load_bundle(stem)
            for key in arrays:
                np.testing.assert_array_equal(actual[key], arrays[key])
            self.assertEqual(receipt['role'], 'train_main')
            stem.with_suffix('.npz').write_bytes(stem.with_suffix('.npz').read_bytes()+b'corrupt')
            with self.assertRaises(ValueError):
                load_bundle(stem)

    def test_distinct_diagnosis_cannot_adopt_old_transfer_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results/runs') as name:
            with lock_directory(name, {'version': 'old'}):
                pass
            with self.assertRaises(ValueError):
                with lock_directory(name, {'version': 'diagnosis'}):
                    pass


if __name__ == '__main__':
    unittest.main()
