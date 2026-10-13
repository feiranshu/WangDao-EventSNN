"""Check diagnostic masks, readout bounds and noninterference with learning."""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnose_temporal_snn import code_audit, masks, snapshot
from wangdao.temporal_segmentation import TemporalSegmenter
from wangdao.temporal_study import update


class DiagnosisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_masks_mixed_polarity_and_temporal_support(self):
        indices = torch.tensor([[0, 0, 0, 20, 20], [0, 0, 1, 20, 20], [1, 0, 0, 40, 40]])
        region = masks(indices, torch.tensor([1., 0., 0.]), 1, False)
        self.assertTrue(region["mixed_cells"][0, 0, 5, 5])
        self.assertFalse(region["target_only_cells"].any())
        self.assertTrue(region["background_only_cells"][1, 0, 10, 10])
        self.assertTrue(region["empty_current_cells"][1, 0, 5, 5])
        support = region["causal_event_support"]
        self.assertTrue(support[1, 0, 0, 5, 5])
        self.assertEqual(int(support[0, 1].sum()), 9)
        self.assertEqual(int(support[0, 2].sum()), 25)
        self.assertFalse(support[0, 2, 0, 10, 10])

    def test_phase_code_oracle_retains_duplicates(self):
        indices = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 1, 0, 0]])
        spike = torch.zeros(4, 1, 8, 65, 87)
        result = code_audit(spike, indices, torch.tensor([1., 0., 0.]))
        self.assertEqual(result["target_events"]["zero_code_fraction"], 1.)
        self.assertEqual(result["phase_code_oracle"]["oracle_event_iou"], .5)
        spike[0, 0, 0, 0, 0] = .1
        with self.assertRaises(ValueError):
            code_audit(spike, indices, torch.tensor([1., 0., 0.]))

    def test_observation_does_not_change_next_weight_update(self):
        x = torch.zeros(4, 1, 64, 65, 87)
        x[:, :, :4, 5, 5] = .5
        indices = torch.tensor([[0, 0, 0, 20, 20], [0, 0, 1, 20, 20], [3, 0, 0, 20, 20]])
        batch = x, indices, torch.tensor([1., 0., 1.])
        model, control = TemporalSegmenter("Local-SNN"), TemporalSegmenter("Local-SNN")
        diagnostic = snapshot(model, batch, torch.device("cpu"))
        self.assertTrue(diagnostic["observer_equivalence"])
        self.assertLessEqual(diagnostic["readout"]["optimistic_task_loss_fixed_head"], diagnostic["task_loss"])
        bounds = np.array(diagnostic["readout"]["possible_logit_bounds_per_phase"])
        weights = model.head.weight.detach().flatten(1).numpy()
        assert model.head.bias is not None
        bias = model.head.bias.detach().numpy()
        codes = ((np.arange(256)[:, None] >> np.arange(8)) & 1)
        all_outputs = codes @ weights.T + bias
        np.testing.assert_allclose(bounds[:, 0], all_outputs.min(0), atol=1e-6)
        np.testing.assert_allclose(bounds[:, 1], all_outputs.max(0), atol=1e-6)
        for m in (model, control):
            update(m, batch, torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4), torch.device("cpu"))
        for a, b in zip(model.parameters(), control.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
