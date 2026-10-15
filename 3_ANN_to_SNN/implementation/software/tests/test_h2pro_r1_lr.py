"""Synthetic-only paired LR updates, source equivalence, and crash recovery."""
from dataclasses import asdict
from copy import deepcopy
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

from software.tests.test_h2pro_r1_train import toy_data
from wangdao.h2pro_r1_train_v1.common import Config, IMPLEMENTATION, read_json, atomic_json, lock_directory
from wangdao.h2pro_r1_train_v1.model import R1T16, update
from wangdao.h2pro_r1_train_v1.evaluation import evaluate
from wangdao.h2pro_r1_train_v1.runtime import setup_device
from wangdao.h2pro_longtrain_v1.storage import all_rng, set_rng, load, save_atomic, PauseRequested
from wangdao.h2pro_r1_lr_v1.common import (
    ARMS, LEARNING_RATES, SOURCE_EPOCH, digest, sha256, arm_identity,
    initialized_payload, verify_prepared, independent_output,
)
from wangdao.h2pro_r1_lr_v1.checks import initialize_arms, check, synthetic_resume
from wangdao.h2pro_r1_lr_v1.comparison import compare
from wangdao.h2pro_r1_lr_v1.runtime import train_arm, train_suite, status, validate_cursor


def fixture():
    config = Config(batch_size=32)
    data = toy_data(config)
    torch.manual_seed(42)
    model = R1T16(); opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0., foreach=False)
    for _ in range(2):
        update(model, opt, data.batch('train_main', slice(0, 32)), config)
    baseline, _, _ = evaluate(model, data, 32)
    origin = {'model': deepcopy(model.state_dict()), 'optimizer': deepcopy(opt.state_dict()),
              'rng': all_rng(), 'state': {'global_step': 2}}
    identity = {'synthetic_only': True, 'config': asdict(config)}
    return config, data, model, baseline, origin, identity


def prepare(directory, fx):
    config, data, model, baseline, origin, identity = fx
    with lock_directory(directory, identity):
        hashes = initialize_arms(directory, identity, origin, baseline, model)
        immutable = [f'{a}/{n}' for a in ARMS for n in ('identity.json', 'initial.pt', 'initial_int8.npz')]
        ready = {'status': 'passed', 'identity': identity, 'initial_student': baseline,
                 'frozen_ann': baseline, 'original_r1_before_training': baseline,
                 'initial_hashes': hashes, 'immutable_files_sha256': {n: sha256(directory/n) for n in immutable}}
        atomic_json(directory/'ready.json', ready)
        atomic_json(directory/'suite_status.json', {'status': 'ready_not_trained'})
    return ready


class PairedLRTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_device('cpu')

    def setUp(self):
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.fx = fixture()
        self.config, self.data, self.model, self.baseline, self.origin, self.identity = self.fx

    def temporary(self):
        temp = tempfile.TemporaryDirectory(dir=IMPLEMENTATION/'results/runs')
        assert Path(temp.name).resolve().parent == (IMPLEMENTATION/'results/runs').resolve()
        return temp

    def suite(self, path, target=2):
        return train_suite(suite=path, additional_epochs=target, checkpoint_batches=1,
                           device_name='cpu', _test_context=(self.identity, self.config, self.data))

    def arm(self, path, arm, ready, target=2, **kwargs):
        return train_arm(self.data, path, arm, self.identity, ready, self.config, target, 1, **kwargs)

    def assert_same_committed(self, a, b):
        for field in ('model', 'optimizer', 'rng'):
            self.assertEqual(digest(a[field]), digest(b[field]), field)
        self.assertEqual(a['state']['global_step'], b['state']['global_step'])
        for ea, eb in zip(a['state']['history'], b['state']['history']):
            self.assertEqual(ea['training']['loss_sums'], eb['training']['loss_sums'])
            self.assertEqual(ea['evaluation']['overall'], eb['evaluation']['overall'])

    def test_initialization_preserves_source_and_adam_except_lr(self):
        before = digest(self.origin)
        with self.temporary() as temp:
            path = Path(temp); ready = prepare(path, self.fx)
            for arm in ARMS:
                p = load(path/arm/'initial.pt')
                self.assertEqual(digest(p['model']), digest(self.origin['model']))
                self.assertEqual(digest(p['optimizer']['state']), digest(self.origin['optimizer']['state']))
                self.assertEqual(digest(p['rng']), digest(self.origin['rng']))
                self.assertEqual(p['optimizer']['param_groups'][0]['lr'], LEARNING_RATES[arm])
                self.assertEqual(p['state']['global_step'], 2)
                self.assertEqual(p['state']['best_epoch'], 0)
            self.assertEqual(ready['initial_hashes']['control'], ready['initial_hashes']['low_lr'])
        self.assertEqual(digest(self.origin), before)

    def test_synthetic_preflight_has_independent_optimizer_tensor_storage(self):
        before = digest(self.origin)
        with self.temporary() as temp:
            result = synthetic_resume(torch.device('cpu'), Path(temp), self.origin,
                                      self.identity, self.baseline, self.config)
        self.assertTrue(all(result[a]['model_adam_rng_bit_exact'] for a in ARMS))
        self.assertEqual(digest(self.origin), before)

    def test_mid_batch_resume_is_exact_for_both_learning_rates(self):
        with self.temporary() as temp:
            full, resumed = Path(temp)/'full', Path(temp)/'resumed'
            ready = prepare(full, self.fx); prepare(resumed, self.fx)
            for arm in ARMS:
                self.arm(full, arm, ready)
                paused = self.arm(resumed, arm, ready, _test_pause_after=2)
                self.assertEqual(paused['status'], 'paused')
                self.assertEqual(paused['cursor']['batch'], 2)
                self.arm(resumed, arm, ready)
                self.assert_same_committed(load(full/arm/'last.pt'), load(resumed/arm/'last.pt'))
                self.assertEqual(load(resumed/arm/'last.pt')['state']['global_step'], 12)

    def test_control_matches_original_update_and_absolute_order(self):
        with self.temporary() as temp:
            path = Path(temp); ready = prepare(path, self.fx)
            self.arm(path, 'control', ready)
            model = R1T16(); model.load_state_dict(self.origin['model'])
            opt = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0., foreach=False)
            opt.load_state_dict(self.origin['optimizer']); set_rng(self.origin['rng'])
            for epoch in (160, 161):
                order = self.data.order(epoch, 42)
                for lo in range(0, 131, 32):
                    update(model, opt, self.data.batch('train_main', order[lo:lo+32]), self.config)
            saved = load(path/'control/last.pt')
            self.assertEqual(digest(model.state_dict()), digest(saved['model']))
            self.assertEqual(digest(opt.state_dict()), digest(saved['optimizer']))
            self.assertEqual(digest(all_rng()), digest(saved['rng']))

    def test_arms_share_first_batch_but_learning_rate_changes_updates(self):
        with self.temporary() as temp:
            path = Path(temp); ready = prepare(path, self.fx); seen = []
            def record(model, opt, batch, config):
                seen.append((config.learning_rate, digest(batch['features'])))
                return update(model, opt, batch, config)
            with patch('wangdao.h2pro_r1_lr_v1.runtime.update', side_effect=record):
                self.arm(path, 'control', ready, 1); self.arm(path, 'low_lr', ready, 1)
            self.assertEqual([x[1] for x in seen[:5]], [x[1] for x in seen[5:]])
            self.assertEqual([x[0] for x in seen], [1e-4]*5 + [3e-5]*5)
            self.assertNotEqual(digest(load(path/'control/last.pt')['model']), digest(load(path/'low_lr/last.pt')['model']))

    def test_stage_order_comparisons_and_complete_epoch_retention(self):
        with self.temporary() as temp:
            path = Path(temp); prepare(path, self.fx); calls = []
            def record(*args, **kwargs):
                calls.append((args[2], args[6])); return train_arm(*args, **kwargs)
            with patch('wangdao.h2pro_r1_lr_v1.runtime.MILESTONES', (1, 2, 3)), \
                    patch('wangdao.h2pro_r1_lr_v1.runtime.train_arm', side_effect=record):
                result = self.suite(path, 3)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(calls, [(a, n) for n in (1, 2, 3) for a in ARMS])
            for n in (1, 2, 3):
                comparison = read_json(path/f'comparison_added_{n:03}.json')
                self.assertEqual(comparison['effective_epoch'], 159+n)
                for arm in ARMS:
                    self.assertEqual(len(comparison['arms'][arm]['learning_trend']), n)
                    saved = load(path/arm/'epochs'/f'added_{n:03}.pt')
                    self.assertEqual(saved['state']['global_step'], 2+5*n)
                    self.assertEqual(saved['state']['cursor']['epoch'], n+1)
                    self.assertEqual(len(saved['state']['history']), n)

    def test_whole_suite_pause_stops_before_second_arm_and_resumes(self):
        with self.temporary() as temp:
            path = Path(temp); prepare(path, self.fx)
            def request(model, opt, batch, cfg):
                result = update(model, opt, batch, cfg)
                (path/'PAUSE').write_text('synthetic request')
                return result
            with patch('wangdao.h2pro_r1_lr_v1.runtime.update', side_effect=request):
                result = self.suite(path, 1)
            self.assertEqual(result['status'], 'paused')
            self.assertEqual(load(path/'control/last.pt')['state']['global_step'], 3)
            self.assertEqual(load(path/'low_lr/last.pt')['state']['global_step'], 2)
            result = self.suite(path, 1)
            self.assertEqual(result['status'], 'complete')
            self.assertFalse((path/'PAUSE').exists())

    def test_evaluation_pause_does_not_repeat_updates(self):
        with self.temporary() as temp:
            path = Path(temp); ready = prepare(path, self.fx)
            with patch('wangdao.h2pro_r1_lr_v1.runtime.evaluate', side_effect=PauseRequested()):
                result = self.arm(path, 'low_lr', ready, 1)
            self.assertEqual(result['cursor']['phase'], 'evaluate')
            self.assertEqual(result['global_step'], 7)
            with patch('wangdao.h2pro_r1_lr_v1.runtime.update', side_effect=AssertionError('must not retrain')):
                self.arm(path, 'low_lr', ready, 1)
            self.assertEqual(load(path/'low_lr/last.pt')['state']['global_step'], 7)

    def test_nonfinite_failure_keeps_valid_last_and_suite_failure_status(self):
        with self.temporary() as temp:
            path = Path(temp); prepare(path, self.fx); calls = []
            def fail(model, opt, batch, cfg):
                calls.append(1)
                if len(calls) == 2:
                    with torch.no_grad():
                        model.weights[0][0, 0] = float('nan')
                    raise FloatingPointError('synthetic nonfinite fault')
                return update(model, opt, batch, cfg)
            with patch('wangdao.h2pro_r1_lr_v1.runtime.update', side_effect=fail):
                with self.assertRaises(FloatingPointError):
                    self.suite(path, 1)
            self.assertEqual(status(path)['workflow_status'], 'failed')
            saved = load(path/'control/last.pt')
            self.assertEqual(saved['state']['global_step'], 3)
            self.assertTrue(all(bool(torch.isfinite(v).all()) for v in saved['model'].values()))
            self.assertEqual(load(path/'low_lr/last.pt')['state']['global_step'], 2)
            self.assertEqual(self.suite(path, 1)['status'], 'complete')

    def test_committed_last_repairs_history_after_write_failure(self):
        with self.temporary() as temp:
            path = Path(temp); prepare(path, self.fx)
            def fail_history(target, value):
                if target.name == 'training.json':
                    raise OSError('synthetic history write fault')
                return atomic_json(target, value)
            with patch('wangdao.h2pro_r1_lr_v1.runtime.atomic_json', side_effect=fail_history):
                with self.assertRaises(OSError):
                    self.suite(path, 1)
            self.assertEqual(len(load(path/'control/last.pt')['state']['history']), 1)
            result = self.suite(path, 1)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(read_json(path/'control/training.json')['epochs']), 1)

    def test_repeating_completed_budget_never_adds_steps_or_changes_comparison(self):
        with self.temporary() as temp:
            path = Path(temp); ready = prepare(path, self.fx)
            self.suite(path, 1)
            first = (path/'comparison_added_001.json').read_bytes()
            self.suite(path, 2)
            self.assertEqual(compare(path, 1, ready), read_json(path/'comparison_added_001.json'))
            self.assertEqual(first, (path/'comparison_added_001.json').read_bytes())
            states = {a: sha256(path/a/'last.pt') for a in ARMS}
            with patch('wangdao.h2pro_r1_lr_v1.runtime.update', side_effect=AssertionError('extra update')):
                self.suite(path, 2)
            self.assertEqual(states, {a: sha256(path/a/'last.pt') for a in ARMS})
            for arm in ARMS:
                self.assertEqual(load(path/arm/'last.pt')['state']['global_step'], 12)

    def test_comparison_requires_both_arms_and_contains_cost_memory_and_reference(self):
        with self.temporary() as temp:
            path = Path(temp); ready = prepare(path, self.fx)
            self.arm(path, 'control', ready, 1)
            with self.assertRaises((ValueError, FileNotFoundError)):
                compare(path, 1, ready)
            self.arm(path, 'low_lr', ready, 1)
            report = compare(path, 1, ready)
            for arm in ARMS:
                self.assertIn('memory', report['arms'][arm])
                self.assertIn('cost', report['arms'][arm]['endpoint'])
                self.assertIn('hd_recall', report['arms'][arm]['last_window'])
                self.assertIn('best_added_epoch', report['arms'][arm])
            self.assertEqual(report['low_lr_minus_control_cost']['mac'], 0)

    def test_changed_lr_in_checkpoint_is_rejected(self):
        with self.temporary() as temp:
            path = Path(temp); ready = prepare(path, self.fx)
            p = load(path/'low_lr/last.pt'); p['optimizer']['param_groups'][0]['lr'] = 1e-4
            save_atomic(path/'low_lr/last.pt', p)
            with self.assertRaisesRegex(ValueError, 'learning rate'):
                self.arm(path, 'low_lr', ready, 1)

    def test_immutable_preparation_tampering_is_rejected(self):
        with self.temporary() as temp:
            path = Path(temp); prepare(path, self.fx)
            (path/'control/initial.pt').write_bytes(b'synthetic corruption')
            with self.assertRaisesRegex(ValueError, 'Immutable'):
                verify_prepared(path, self.identity)

    def test_bad_cursor_is_rejected(self):
        p = initialized_payload(self.origin, arm_identity(self.identity, 'control'), self.baseline)
        p['state']['global_step'] += 1
        with self.assertRaisesRegex(ValueError, 'coverage'):
            validate_cursor(p['state'], self.data, self.config)

    def test_preflight_reuse_does_not_reload_real_data_or_reset_trained_last(self):
        with self.temporary() as temp:
            path = Path(temp); ready = prepare(path, self.fx)
            self.suite(path, 1)
            before = {a: sha256(path/a/'last.pt') for a in ARMS}
            with patch('wangdao.h2pro_r1_lr_v1.checks.inputs', return_value=(self.identity, self.origin, self.config, {})), \
                    patch('wangdao.h2pro_r1_lr_v1.checks.ResidentData', side_effect=AssertionError('must reuse')):
                result = check(suite=path, device_name='cpu')
            self.assertEqual(result, ready)
            self.assertEqual(before, {a: sha256(path/a/'last.pt') for a in ARMS})

    def test_invalid_budget_and_source_output_overlap(self):
        for count in (0, -1, True):
            with self.assertRaises(ValueError):
                self.suite(Path('unused'), count)
        source = IMPLEMENTATION/'results/runs/h2pro_r1t16_train_v1_s42'
        for out in (source, source/'child', source.parent):
            with self.assertRaises(ValueError):
                independent_output(out, source)

    def test_killed_process_releases_locks_and_recovers_exactly(self):
        with self.temporary() as temp:
            full, stopped = Path(temp)/'full', Path(temp)/'stopped'
            prepare(full, self.fx); prepare(stopped, self.fx)
            self.suite(full, 2)
            code = '''import sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'software'))
from software.tests.test_h2pro_r1_train import toy_data
from wangdao.h2pro_r1_train_v1.common import Config,read_json
import wangdao.h2pro_r1_lr_v1.runtime as runtime
config=Config(batch_size=32);data=toy_data(config);path=Path(sys.argv[1])
identity=read_json(path/'identity.json');original=runtime.update
def slow(*args):
 time.sleep(.15)
 return original(*args)
runtime.update=slow
runtime.train_suite(suite=path,additional_epochs=2,checkpoint_batches=1,device_name='cpu',_test_context=(identity,config,data))
'''
            with (Path(temp)/'child.log').open('w') as log:
                child = subprocess.Popen([sys.executable, '-B', '-c', code, str(stopped)], cwd=IMPLEMENTATION,
                    stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                try:
                    deadline = time.monotonic()+30
                    while time.monotonic() < deadline:
                        p = read_json(stopped/'control/progress.json')
                        if p.get('additional_updates', 0) >= 2:
                            break
                        if child.poll() is not None:
                            self.fail('Synthetic child exited before checkpoint')
                        time.sleep(.03)
                    else:
                        self.fail('Synthetic child did not commit')
                    child.kill(); child.wait(timeout=10)
                finally:
                    if child.poll() is None:
                        child.kill(); child.wait(timeout=10)
            self.suite(stopped, 2)
            for arm in ARMS:
                self.assert_same_committed(load(full/arm/'last.pt'), load(stopped/arm/'last.pt'))


if __name__ == '__main__':
    unittest.main()
