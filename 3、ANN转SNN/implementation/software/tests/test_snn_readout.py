"""Readout single-factor controls preserve causality and original defaults."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.study_snn_readout import decide
from scripts.train_snn_readout import run_variant
from wangdao.temporal_segmentation import TemporalSegmenter
from wangdao.temporal_study import SequenceStore, checkpoint_load, update


class ReadoutTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_scaled_initialization_changes_only_head_weight(self):
        a = TemporalSegmenter("Local-SNN")
        b = TemporalSegmenter("Local-SNN", head_std=8 ** -.5)
        for key in a.state_dict():
            if key != "head.weight":
                torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
        torch.testing.assert_close(b.head.weight, a.head.weight * (8 ** -.5 / .01))
        self.assertEqual(sum(p.numel() for p in b.parameters()), 1952)

    def test_membrane_is_causal_leaky_sum_with_single_bias(self):
        instant = TemporalSegmenter("Local-SNN")
        model = TemporalSegmenter("Local-SNN", readout="membrane")
        x = torch.zeros(4, 1, 64, 65, 87)
        x[0, :, :, 5, 5] = 1
        with torch.no_grad():
            plain = instant(x)
            output = model(x)
            assert model.head.bias is not None
            bias = F.pixel_shuffle(model.head.bias[None, :, None, None].expand(1, 32, 65, 87), 4)[..., :346]
            accumulated = torch.zeros_like(bias)
            for t in range(4):
                accumulated = .875 * accumulated + plain.logits[t] - bias
                torch.testing.assert_close(output.logits[t], accumulated + bias, rtol=1e-5, atol=1e-6)
            state = None
            values = []
            for step in x:
                result = model.step(step, state)
                state = result.state
                values.append(result.logits)
            torch.testing.assert_close(output.logits, torch.stack(values), rtol=0, atol=0)
            changed = x.clone()
            changed[2:] = 1
            torch.testing.assert_close(model(changed).logits[:2], output.logits[:2], rtol=0, atol=0)
            model(changed)
            torch.testing.assert_close(model(x).logits, output.logits, rtol=0, atol=0)
            self.assertEqual(len(output.state), 4)
            self.assertEqual(output.state[-1].numel(), 180960)
            with self.assertRaises(ValueError):
                model.step(x[0], plain.state)

    def test_empty_steps_do_not_accumulate_prior_and_batch_isolation(self):
        model = TemporalSegmenter("Dilated-SNN", readout="membrane")
        with torch.no_grad():
            empty = model(torch.zeros(4, 2, 64, 65, 87))
            for step in empty.logits:
                torch.testing.assert_close(step, empty.logits[0], rtol=0, atol=0)
            self.assertEqual(int(torch.count_nonzero(empty.state[-1])), 0)
            x = torch.zeros(4, 1, 64, 65, 87)
            x[:, :, 0, 5, 5] = 1
            together = model(torch.cat((x, torch.zeros_like(x)), dim=1))
            torch.testing.assert_close(together.logits[:, :1], model(x).logits, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(together.logits[:, 1], empty.logits[:, 1], rtol=0, atol=0)

    def test_admission_requires_fit_and_probe_and_full_run(self):
        def row(initial, final, probe):
            return {"initial_fit": {"total_loss": initial},
                    "final_fit": {"total_loss": final, "task_loss": final, "layers": [{"global_rate": .1}] * 3},
                    "final_probe": {"task_loss": probe}}
        data = {"Local-SNN": {"baseline": row(.6, .54, .55), "scaled_init": row(.7, .5, .5)}}
        self.assertEqual(decide(data, False)["admitted_short_check_candidates"], [])
        self.assertEqual(decide(data, True)["admitted_short_check_candidates"], ["Local-SNN/scaled_init"])
        worse = copy.deepcopy(data)
        worse["Local-SNN"]["scaled_init"]["final_probe"]["task_loss"] = .56
        self.assertEqual(decide(worse, True)["admitted_short_check_candidates"], [])

    def test_membrane_training_resume_restores_optimizer_and_selection(self):
        x = torch.zeros(4, 1, 64, 65, 87)
        x[:, :, :4, 5, 5] = 1
        batch = x, torch.tensor([[0, 0, 0, 20, 20], [3, 0, 1, 20, 20]]), torch.tensor([1., 0.])
        evaluation = {"event": {"iou": .5, "f1": 2 / 3}, "threshold": .2, "seconds": 0.0}
        store = Mock(spec=SequenceStore)
        store.stats = {}
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()), \
             patch("scripts.train_snn_readout.batches", side_effect=lambda *args: iter([batch])), \
             patch("scripts.train_snn_readout.evaluate_model", return_value=evaluation):
            root = Path(temp)
            def run(name):
                return run_variant(root / name, "membrane", store, [("synthetic", 0)], (.1, .1),
                                   torch.device("cpu"), {"synthetic": True}, pilot=True)
            calls = [0]
            def interrupt(*args):
                calls[0] += 1
                if calls[0] == 2:
                    raise RuntimeError("interrupted")
                return update(*args)
            with patch("scripts.train_snn_readout.update", side_effect=interrupt):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run("resumed")
            resumed = run("resumed")
            run("continuous")
            a = checkpoint_load(root / "resumed/membrane/last.pt", torch.device("cpu"))
            b = checkpoint_load(root / "continuous/membrane/last.pt", torch.device("cpu"))
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], rtol=0, atol=0)
            self.assertEqual(resumed["best"]["epoch"], 1)
            self.assertEqual(resumed["status"], "ok")


if __name__ == "__main__":
    unittest.main()
