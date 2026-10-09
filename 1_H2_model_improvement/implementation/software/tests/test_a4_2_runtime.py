from __future__ import annotations

import json
import hashlib
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.a4_2_runtime import (  # noqa: E402
    A4_2_DIMENSIONS, A4_2_EPOCH, A4_2_MODEL_ID, A4_2_SEED, A4_2_THRESHOLD,
    load_a4_2_refiner,
)


WEIGHTS = ROOT / "software" / "weights" / "A4-2_seed42_epoch008_int8.npz"
EXPECTED_INT8_SHA256 = "d7d42a3876c37497bdcd5206e6f35a5f949b0230dde0675fd128f989b70d05c4"


class A42RuntimeTests(unittest.TestCase):
    def test_final_int8_artifact_hash(self):
        self.assertEqual(hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(), EXPECTED_INT8_SHA256)

    def test_final_refiner_identity_and_dimensions(self):
        refiner = load_a4_2_refiner(WEIGHTS)
        dimensions = [refiner.layers[0][0].shape[0]]
        dimensions.extend(layer[0].shape[1] for layer in refiner.layers)
        self.assertEqual(tuple(dimensions), A4_2_DIMENSIONS)
        self.assertEqual(refiner.threshold, A4_2_THRESHOLD)
        self.assertEqual(refiner.metadata["model_id"], A4_2_MODEL_ID)
        self.assertEqual(refiner.metadata["seed"], A4_2_SEED)
        self.assertEqual(refiner.metadata["epoch"], A4_2_EPOCH)

    def test_deployment_decision_matches_runtime(self):
        decision = json.loads((ROOT / "results" / "final_A4-2" / "deployment_decision.json").read_text(encoding="utf-8"))
        self.assertEqual(decision["model"]["model_id"], A4_2_MODEL_ID)
        self.assertEqual(decision["model"]["seed"], A4_2_SEED)
        self.assertEqual(decision["model"]["epoch"], A4_2_EPOCH)
        self.assertEqual(decision["model"]["threshold"], A4_2_THRESHOLD)
        self.assertEqual(decision["status"], "accepted_for_deployment")


if __name__ == "__main__":
    unittest.main()

