import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from snn_accel.bittrue import approximate_add


@unittest.skipIf(torch is None, "PyTorch is optional")
class TrainingTests(unittest.TestCase):
    def test_approximate_add_ste_matches_reference_and_has_gradient(self):
        from snn_accel.training import approximate_add_ste

        a = torch.tensor([7.0, 120.0], requires_grad=True)
        b = torch.tensor([3.0, 120.0], requires_grad=True)
        output = approximate_add_ste(a, b, 8, 2)
        np.testing.assert_array_equal(output.detach().numpy(), approximate_add([7, 120], [3, 120], 8, 2))
        output.sum().backward()
        np.testing.assert_array_equal(a.grad.numpy(), [1.0, 1.0])
        np.testing.assert_array_equal(b.grad.numpy(), [1.0, 1.0])

    def test_distillation_terms(self):
        from snn_accel.training import distillation_loss

        detection = torch.tensor(2.0, requires_grad=True)
        student = torch.tensor([[1.0, 0.0]], requires_grad=True)
        teacher = torch.tensor([[0.0, 1.0]])
        terms = distillation_loss(
            detection_loss=detection,
            student_logits=student,
            teacher_logits=teacher,
            kd_weight=0.5,
            temperature=2.0,
        )
        self.assertGreater(float(terms.total.detach()), 1.0)
        terms.total.backward()
        self.assertIsNotNone(student.grad)


if __name__ == "__main__":
    unittest.main()

