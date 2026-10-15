"""Direction 4 numeric, data-isolation and interrupted-runtime regressions."""
from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from typing import Any

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from wangdao.h2pro_direct_v1.core import forward, load_layers, probabilities
from wangdao.h2pro_distill_v1.checks import synthetic_resume
from wangdao.h2pro_distill_v1.common import Config, IMPLEMENTATION, atomic_json, lock_directory, read_json
from wangdao.h2pro_distill_v1.data import ResidentData
from wangdao.h2pro_distill_v1.evaluation import evaluate, export, score
from wangdao.h2pro_distill_v1.model import CountSTE, R2T8, class_logits, losses, update
from wangdao.h2pro_distill_v1.runtime import (checkpoint, initial_state, restore, setup_device, train)
from wangdao.h2pro_longtrain_v1.storage import PauseRequested, load


def toy_data(config):
    data: Any = ResidentData.__new__(ResidentData)
    data.device = torch.device('cpu')
    data.identity = {'synthetic_fixture': 'distill_runtime_v1', 'config': asdict(config)}
    data.stats = {'synthetic_only': True}
    data.roles, data.sequences = {}, {}
    rng = np.random.default_rng(712)
    for role in ('train_main', 'train_dev'):
        seqs, xs, ys, ts = [], [], [], []
        offset = 0
        for i, n in enumerate((63, 68)):
            x = rng.integers(0, 256, (n, 36), dtype=np.uint8)
            t, _ = forward(x, load_layers())
            y = rng.integers(0, 2, n+3, dtype=np.uint8)
            y[-3:] = [1, 1, 0]
            c = np.r_[np.ones(n, bool), np.zeros(3, bool)]
            row = {'sequence': f'synthetic_{role}_{i}', 'role': role, 'events': n+3,
                   'high_density': bool(i), 'candidate_events': n, 'target_events': int(y.sum()),
                   'candidate_targets': int(y[c].sum()), 'proposal_sparse_mac': 40, 'proposal_dense_mac': 80}
            seqs.append({'row': row, 'labels': y, 'candidate': c, 'lo': offset, 'hi': offset+n})
            offset += n
            xs.append(x); ys.append(y[c]); ts.append(t)
        data.roles[role] = {k: torch.from_numpy(np.concatenate(v))
                            for k, v in [('features', xs), ('labels', ys), ('teacher', ts)]}
        data.sequences[role] = seqs
    return data


def prepare_toy(directory, data):
    model = R2T8()
    result, _, _ = evaluate(model, data, 32)
    teacher = data.roles['train_dev']['teacher'].numpy()
    ann = score(data, teacher, np.zeros((len(teacher), 2), np.int64))
    with lock_directory(directory, data.identity) as out:
        atomic_json(out/'ready.json', {'status': 'passed', 'identity': data.identity,
                                     'initial_student': result, 'frozen_ann': ann})


class Direction4Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_device('cpu')
        cls.layers = load_layers()
        cls.features = np.random.default_rng(71).integers(0, 256, (257, 36), dtype=np.uint8)

    def temporary(self):
        return tempfile.TemporaryDirectory(dir=IMPLEMENTATION/'results/runs')

    def test_initial_forward_and_counts_are_original_r2_t8(self):
        model = R2T8()
        raw, spikes = model(torch.from_numpy(self.features))
        reference, cost = forward(self.features, self.layers, 'r2', 8, 'spike_add')
        np.testing.assert_array_equal(raw.detach().numpy(), reference)
        self.assertEqual(int((spikes*torch.tensor([32, 4])).sum()), cost.executed_synaptic_ac)
        self.assertEqual([p.numel() for p in model.parameters()], [2304, 2048, 128])

    def test_full_uint8_encoder_domain_and_derivative(self):
        x = torch.arange(256, dtype=torch.float64, requires_grad=True)
        counts = CountSTE.apply(x)
        self.assertIsNotNone(counts)
        assert counts is not None
        np.testing.assert_array_equal(counts.detach().numpy(), (np.arange(256)*8+127)//255)
        counts.sum().backward()
        torch.testing.assert_close(x.grad, torch.full_like(x, 8/255), rtol=0, atol=0)

    def test_chunks_and_reset_do_not_change_predictions(self):
        model = R2T8()
        x = torch.from_numpy(self.features)
        raw, _ = model(x)
        parts = torch.cat([model(part)[0] for part in x.split(23)])
        self.assertTrue(torch.equal(raw, parts))
        empty, spikes = model(torch.empty((0, 36), dtype=torch.uint8))
        self.assertEqual(tuple(empty.shape), (0, 4))
        self.assertEqual(tuple(spikes.shape), (0, 2))

    def test_supervision_and_kd_match_explicit_definition(self):
        config = Config()
        raw = torch.tensor([[1., 2., 3., 4.], [2., 1., 3., -4.]], requires_grad=True)
        teacher = torch.tensor([[4., 2., 3., 1.], [5., 1., 2., 1.]], requires_grad=True)
        labels = torch.tensor([1, 0])
        total, ce, kd = losses(raw, labels, teacher, config)
        s, t = class_logits(raw), class_logits(teacher)
        logp, logq = torch.log_softmax(s/2, 1), torch.log_softmax(t/2, 1)
        expected = (logq.exp()*(logq-logp)).sum(1).mean()*4
        torch.testing.assert_close(kd, expected)
        torch.testing.assert_close(total, .5*ce+.5*kd)
        total.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(raw.grad)

    def test_only_original_student_weights_update(self):
        model = R2T8()
        data = toy_data(Config())
        original = [p.detach().clone() for p in model.parameters()]
        scales = [s.copy() for _, s in model.export_layers()]
        update(model, torch.optim.Adam(model.parameters(), lr=.0001, foreach=False),
               data.batch('train_main', slice(None)), Config())
        self.assertTrue(all(not torch.equal(a, b) for a, b in zip(original, model.parameters())))
        for (_, scale), expected in zip(model.export_layers(), scales):
            np.testing.assert_array_equal(scale, expected)
        self.assertTrue(all(not b.requires_grad for b in model.buffers()))

    def test_valid_edge_weight_bin_still_has_training_gradient(self):
        model = R2T8()
        with torch.no_grad():
            model.weights[0][0, 0] = 127.25*getattr(model, 'scale0')[0]
        weight = model.integer_weight(0)
        self.assertEqual(float(weight[0, 0]), 127.)
        weight[0, 0].backward()
        gradient = model.weights[0].grad
        assert gradient is not None
        self.assertNotEqual(float(gradient[0, 0]), 0.)

    def test_updated_export_is_still_original_kernel_compatible(self):
        model = R2T8()
        data = toy_data(Config())
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, foreach=False)
        for _ in range(3):
            update(model, optimizer, data.batch('train_main', slice(None)), Config())
        with self.temporary() as temp:
            path = Path(temp)/'student.npz'
            export(path, model)
            with np.load(path) as z:
                layers = tuple((z[f'layer{i}_weight'], z[f'layer{i}_scale']) for i in range(3))
            raw, _ = model(torch.from_numpy(self.features))
            expected, _ = forward(self.features, layers, 'r2', 8, 'spike_add')
            np.testing.assert_array_equal(raw.detach().numpy(), expected)

    def test_metrics_keep_proposal_rejects_and_correct_cost_scope(self):
        data = toy_data(Config())
        result, raw, spikes = evaluate(R2T8(), data, 32)
        self.assertEqual(result['overall']['events'], 137)
        self.assertEqual(result['overall']['candidate_events'], 131)
        self.assertEqual(result['overall']['proposal_fn'], 4)
        c = result['cost']
        self.assertEqual(c['inference_equivalent_ac'], int((spikes*[32, 4]).sum()))
        self.assertEqual(c['inference_equivalent_mac_sparse_proposal'], 80+131*2304)
        self.assertEqual(c['executed_reference_refiner_mac'], 131*4480)
        self.assertEqual(c['executed_reference_refiner_ac'], 0)
        self.assertEqual(result['high_density']['events'], 71)

    def test_uniform_order_is_complete_reproducible_and_resumable(self):
        data = toy_data(Config())
        order = data.order(1, 42)
        self.assertTrue(torch.equal(order, data.order(1, 42)))
        self.assertFalse(torch.equal(order, data.order(2, 42)))
        self.assertEqual(sorted(order.tolist()), list(range(131)))
        self.assertTrue(torch.equal(torch.cat([order[:64], data.order(1, 42)[64:]]), order))

    def test_synthetic_minibatch_resume_adam_is_exact(self):
        with self.temporary() as temp:
            r = synthetic_resume(torch.device('cpu'), temp)
        self.assertTrue(r['adam_and_weights_bit_exact'])

    def test_checkpoint_rejects_different_identity(self):
        with self.temporary() as temp:
            model = R2T8()
            opt = torch.optim.Adam(model.parameters())
            path = Path(temp)/'last.pt'
            checkpoint(path, model, opt, initial_state(), {'kind': 'one'})
            with self.assertRaisesRegex(ValueError, 'identity'):
                restore(path, model, opt, {'kind': 'two'})

    def test_atomic_failed_save_keeps_committed_checkpoint(self):
        with self.temporary() as temp:
            model = R2T8(); opt = torch.optim.Adam(model.parameters())
            path = Path(temp)/'last.pt'
            checkpoint(path, model, opt, initial_state(), {})
            original = path.read_bytes()
            with patch('wangdao.h2pro_longtrain_v1.storage.os.replace', side_effect=OSError('synthetic disk failure')):
                with self.assertRaises(OSError):
                    checkpoint(path, model, opt, initial_state(), {})
            self.assertEqual(path.read_bytes(), original)

    def test_end_to_end_mid_epoch_resume_and_total_epoch_semantics(self):
        config = Config(batch_size=32)
        data = toy_data(config)
        with self.temporary() as temp:
            a, b = Path(temp)/'full', Path(temp)/'resumed'
            prepare_toy(a, data); prepare_toy(b, data)
            train(None, a, config, 2, 1, 'cpu', _test_data=data)
            paused = train(None, b, config, 2, 1, 'cpu', _test_data=data, _test_pause_after=2)
            self.assertEqual(paused['status'], 'paused')
            self.assertEqual(paused['cursor']['batch'], 2)
            train(None, b, config, 2, 1, 'cpu', _test_data=data)
            sa, sb = load(a/'last.pt'), load(b/'last.pt')
            self.assertEqual(sa['state']['global_step'], 10)
            self.assertEqual(sb['state']['global_step'], 10)
            for k in sa['model']:
                self.assertTrue(torch.equal(sa['model'][k], sb['model'][k]))
            for ea, eb in zip(sa['state']['history'], sb['state']['history']):
                self.assertEqual(ea['evaluation']['overall'], eb['evaluation']['overall'])
                self.assertEqual(ea['training']['loss_sums'], eb['training']['loss_sums'])
            (b/'training.json').unlink()
            result = train(None, b, config, 2, 1, 'cpu', _test_data=data)
            self.assertEqual(result['global_step'], 10)
            self.assertEqual(result['completed_epochs'], 2)
            self.assertEqual(len(read_json(b/'training.json')['epochs']), 2)

    def test_evaluation_interruption_does_not_retrain_epoch(self):
        config = Config(batch_size=32)
        data = toy_data(config)
        with self.temporary() as temp:
            out = Path(temp)/'run'
            prepare_toy(out, data)
            with patch('wangdao.h2pro_distill_v1.runtime.evaluate', side_effect=PauseRequested()):
                result = train(None, out, config, 1, 10, 'cpu', _test_data=data)
            self.assertEqual(result['cursor']['phase'], 'evaluate')
            self.assertEqual(result['global_step'], 5)
            result = train(None, out, config, 1, 10, 'cpu', _test_data=data)
            self.assertEqual(result['global_step'], 5)
            self.assertEqual(result['completed_epochs'], 1)

    def test_pause_marker_saves_latest_update_and_is_consumed_on_resume(self):
        config = Config(batch_size=32)
        data = toy_data(config)
        with self.temporary() as temp:
            out = Path(temp)/'run'; prepare_toy(out, data)
            def request_after_update(model, optimizer, batch, cfg):
                result = update(model, optimizer, batch, cfg)
                (out/'PAUSE').write_text('synthetic user pause')
                return result
            with patch('wangdao.h2pro_distill_v1.runtime.update', side_effect=request_after_update):
                result = train(None, out, config, 1, 100, 'cpu', _test_data=data)
            self.assertEqual(result['status'], 'paused')
            self.assertEqual(load(out/'last.pt')['state']['global_step'], 1)
            result = train(None, out, config, 1, 100, 'cpu', _test_data=data)
            self.assertEqual(result['global_step'], 5)
            self.assertFalse((out/'PAUSE').exists())

    def test_nonfinite_failure_does_not_commit_bad_weights(self):
        config = Config(batch_size=32)
        data = toy_data(config)
        with self.temporary() as temp:
            out = Path(temp)/'run'; prepare_toy(out, data)
            calls = []
            def failing(model, optimizer, batch, cfg):
                calls.append(1)
                if len(calls) == 2:
                    with torch.no_grad():
                        model.weights[0][0, 0] = float('nan')
                    raise FloatingPointError('synthetic nonfinite fault')
                return update(model, optimizer, batch, cfg)
            with patch('wangdao.h2pro_distill_v1.runtime.update', side_effect=failing):
                with self.assertRaises(FloatingPointError):
                    train(None, out, config, 1, 1, 'cpu', _test_data=data)
            good = load(out/'last.pt')
            self.assertEqual(good['state']['global_step'], 1)
            self.assertTrue(all(bool(torch.isfinite(v).all()) for v in good['model'].values()))
            result = train(None, out, config, 1, 1, 'cpu', _test_data=data)
            self.assertEqual(result['global_step'], 5)
            self.assertEqual(len(list(out.glob('failure_*.json'))), 1)

    def test_killed_process_releases_lock_and_committed_resume_is_exact(self):
        config = Config(batch_size=32)
        data = toy_data(config)
        with self.temporary() as temp:
            full, stopped = Path(temp)/'full', Path(temp)/'killed'
            prepare_toy(full, data); prepare_toy(stopped, data)
            train(None, full, config, 2, 1, 'cpu', _test_data=data)
            code = '''import sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'software'))
from software.tests.test_h2pro_distill import toy_data
from wangdao.h2pro_distill_v1.common import Config
import wangdao.h2pro_distill_v1.runtime as runtime
config=Config(batch_size=32)
data=toy_data(config)
original=runtime.update
def slow_update(*args):
    time.sleep(.15)
    return original(*args)
runtime.update=slow_update
runtime.train(None,Path(sys.argv[1]),config,2,1,'cpu',_test_data=data)
'''
            with (Path(temp)/'child.log').open('w') as log:
                child = subprocess.Popen([sys.executable, '-B', '-c', code, str(stopped)],
                                         cwd=IMPLEMENTATION, stdout=log, stderr=log)
                try:
                    deadline = time.monotonic()+20
                    while time.monotonic() < deadline:
                        progress = stopped/'progress.json'
                        if progress.exists() and read_json(progress).get('global_step', 0) >= 2:
                            break
                        if child.poll() is not None:
                            self.fail('Synthetic child exited before fault injection')
                        time.sleep(.03)
                    else:
                        self.fail('Synthetic child did not reach checkpoint')
                    child.kill()
                    child.wait(timeout=10)
                finally:
                    if child.poll() is None:
                        child.kill(); child.wait(timeout=10)
            self.assertGreaterEqual(load(stopped/'last.pt')['state']['global_step'], 2)
            train(None, stopped, config, 2, 1, 'cpu', _test_data=data)
            a, b = load(full/'last.pt'), load(stopped/'last.pt')
            self.assertEqual(b['state']['global_step'], 10)
            for key in a['model']:
                self.assertTrue(torch.equal(a['model'][key], b['model'][key]))

    def test_unrelated_directory_cannot_be_adopted(self):
        with self.temporary() as temp:
            path = Path(temp); (path/'old.pt').write_text('preserve')
            with self.assertRaises(ValueError):
                with lock_directory(path, {'new': 1}):
                    pass
            self.assertEqual((path/'old.pt').read_text(), 'preserve')

    def test_invalid_config_and_input_rejected(self):
        for config in (Config(batch_size=0), Config(learning_rate=float('nan')), Config(kd_alpha=2)):
            with self.assertRaises(ValueError):
                config.validate()
        with self.assertRaises(ValueError):
            R2T8()(torch.zeros(3, 36))


if __name__ == '__main__':
    unittest.main()
