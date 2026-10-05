"""Step-4 shallow Refiner contracts; no dataset or protected role access."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wangdao.h2_students import (
    build_student,
    complexity_summary,
    export_student_artifact,
    load_student_artifact,
    load_training_checkpoint,
    save_training_checkpoint,
    student_objective,
    temperature_kl,
)


class H2StudentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_complexity_matches_preregistered_32_feature_counts(self):
        r1 = complexity_summary(build_student("S-R1", "A1", 42))
        r2 = complexity_summary(build_student("S-R2", "A1", 42))
        self.assertEqual((r1["total_weights"], r1["serial_weight_layers"]), (8832, 5))
        self.assertEqual((r2["total_weights"], r2["serial_weight_layers"]), (9984, 5))
        self.assertAlmostEqual(r1["reduction_vs_h2_11904"], 0.25806451612903225)

    def test_same_seed_gives_same_direct_and_kd_initialization(self):
        direct = build_student("S-R1", "A3", 44)
        distilled = build_student("S-R1", "A3", 44)
        for left, right in zip(direct.parameters(), distilled.parameters()):
            torch.testing.assert_close(left, right)

    def test_temperature_kl_uses_detached_teacher_and_has_student_gradient(self):
        student = torch.randn(7, 4, requires_grad=True)
        teacher = torch.randn(7, 4, requires_grad=True)
        loss = temperature_kl(student, teacher, temperature=2.0)
        loss.backward()
        self.assertGreater(float(student.grad.abs().sum()), 0.0)
        self.assertIsNone(teacher.grad)
        self.assertAlmostEqual(float(temperature_kl(teacher.detach(), teacher.detach())), 0.0, places=6)

    def test_student_objective_direct_and_kd_share_task_term(self):
        raw = torch.randn(8, 4, requires_grad=True)
        teacher = torch.randn(8, 4)
        labels = torch.tensor([0, 1, 0, 1, 1, 0, 0, 1], dtype=torch.float32)
        direct = student_objective(raw, labels)
        kd = student_objective(raw, labels, teacher, temperature=2.0, kd_weight=0.2)
        torch.testing.assert_close(direct["task"], kd["task"])
        torch.testing.assert_close(direct["total"], direct["task"])
        self.assertGreaterEqual(float(kd["kd"].detach()), 0.0)

    def test_versioned_artifact_and_checkpoint_round_trip(self):
        model = build_student("S-R1", "A1", 42)
        metadata = {"student_variant": "S-R1", "feature_variant": "A1", "seed": 42,
                    "smoke": True, "test_used": False, "validation_used": False}
        inputs = torch.randint(0, 256, (5, 32), dtype=torch.uint8)
        expected = model(inputs).detach()
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            artifact = base / "student.npz"
            loaded = export_student_artifact(model, artifact, metadata)
            torch.testing.assert_close(expected, loaded.model(inputs))
            torch.testing.assert_close(expected, load_student_artifact(artifact).model(inputs))
            checkpoint = base / "student.pt"
            save_training_checkpoint(checkpoint, model, metadata)
            restored, restored_metadata = load_training_checkpoint(checkpoint)
            self.assertEqual(restored_metadata, metadata)
            torch.testing.assert_close(expected, restored(inputs))

    def test_artifact_rejects_inconsistent_dimensions(self):
        model = build_student("S-R1", "A1", 42)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.npz"
            with self.assertRaises(ValueError):
                export_student_artifact(model, path,
                                        {"student_variant": "S-R2", "feature_variant": "A1"})


if __name__ == "__main__":
    unittest.main()
