"""Numerical/loss and synthetic-only continuation, pause and crash regressions."""
from copy import deepcopy
from dataclasses import asdict
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

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

from software.tests.test_h2pro_r1_train import toy_data
from wangdao.h2pro_r1_train_v1.model import R1T16,losses,update as old_update
from wangdao.h2pro_r1_train_v1.runtime import setup_device
from wangdao.h2pro_longtrain_v1.storage import all_rng,set_rng,load,save_atomic,PauseRequested
from wangdao.h2pro_calibration_v1.core import forward,Encoder
from wangdao.h2pro_multit_v1.model import R1Multi,PrefixCountSTE,joint_losses,update
from wangdao.h2pro_multit_v1.common import (
    Config,IMPLEMENTATION,ARMS,STEPS,PROTOCOL,EXIT_POLICY,digest,sha256,read_json,atomic_json,
    lock_directory,arm_identity,initialized_payload,independent_output,verify_prepared)
from wangdao.h2pro_multit_v1.evaluation import evaluate,export,load_export
from wangdao.h2pro_multit_v1.checks import initialize_arms,synthetic_resume,check
from wangdao.h2pro_multit_v1.runtime import train_arm,train_suite,validate_cursor,status
from wangdao.h2pro_multit_v1.comparison import compare


def fixture(device='cpu'):
    config=Config(batch_size=32);data=toy_data(config);data.device=torch.device(device)
    for role in data.roles:
        data.roles[role]={k:v.to(data.device) for k,v in data.roles[role].items()}
    torch.manual_seed(42)
    model=R1Multi().to(data.device);opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=0.,foreach=False)
    for _ in range(2):update(model,opt,data.batch('train_main',slice(0,32)),config,'control')
    baseline,_,_=evaluate(model,data,32)
    origin=dict(model=deepcopy(model.state_dict()),optimizer=deepcopy(opt.state_dict()),rng=all_rng(),state=dict(global_step=2))
    identity=dict(synthetic_only=True,config=asdict(config))
    return config,data,model,baseline,origin,identity


def prepare(path,fx):
    config,data,model,baseline,origin,identity=fx
    with lock_directory(path,identity):
        hashes=initialize_arms(path,identity,origin,baseline,model)
        immutable=[f'{a}/{n}' for a in ARMS for n in ('identity.json','initial.pt','initial_int8.npz')]
        ready=dict(status='passed',identity=identity,initial_student=baseline,frozen_ann=baseline,
                   original_r1_before_training=baseline,initial_hashes=hashes,
                   immutable_files_sha256={n:sha256(path/n) for n in immutable})
        atomic_json(path/'ready.json',ready);atomic_json(path/'suite_status.json',dict(status='ready_not_trained'))
    return ready


class MultiTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):setup_device('cpu')

    def setUp(self):
        self.stack=contextlib.ExitStack();self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self.fx=fixture();self.config,self.data,self.model,self.baseline,self.origin,self.identity=self.fx

    def temp(self):return tempfile.TemporaryDirectory(dir=IMPLEMENTATION/'results/runs')

    def arm(self,path,arm,ready,target=2,**kwargs):
        return train_arm(self.data,path,arm,self.identity,ready,self.config,target,1,**kwargs)

    def suite(self,path,target=2):
        return train_suite(suite=path,additional_epochs=target,checkpoint_batches=1,device_name='cpu',
                           _test_context=(self.identity,self.config,self.data))

    def same(self,a,b):
        for k in ('model','optimizer','rng'):self.assertEqual(digest(a[k]),digest(b[k]),k)
        self.assertEqual(a['state']['global_step'],b['state']['global_step'])
        for ea,eb in zip(a['state']['history'],b['state']['history']):
            for k in ('loss_sums','prefix_loss_sums'):self.assertEqual(ea['training'][k],eb['training'][k])
            for k in ('fixed','adaptive'):self.assertEqual(ea['evaluation'][k],eb['evaluation'][k])

    def test_original_structure_state_and_no_added_parameters(self):
        self.assertEqual(list(R1T16().state_dict()),list(R1Multi().state_dict()))
        self.assertEqual([p.numel() for p in self.model.parameters()],[2304,2048,128])
        self.assertEqual(PROTOCOL['roles'],['train_main','train_dev'])
        self.assertFalse(PROTOCOL['ac_regularization']);self.assertFalse(PROTOCOL['two_step_exit'])

    def test_full_uint8_count_domain_and_step_scaled_derivative(self):
        for t in STEPS:
            x=torch.arange(256,dtype=torch.float64,requires_grad=True)
            c=PrefixCountSTE.apply(x,t);c.sum().backward()
            np.testing.assert_array_equal(c.detach().numpy(),(np.arange(256)*t+191)//255)
            torch.testing.assert_close(x.grad,torch.full_like(x,t/255),rtol=0,atol=0)

    def test_every_prefix_matches_original_explicit_binary_kernel(self):
        x=self.data.roles['train_main']['features'];raw,n=self.model.prefixes(x)
        for i,t in enumerate(STEPS):
            expected,c=forward(x.numpy(),self.model.export_layers(),'r1',t,(Encoder(255,191),Encoder()),'spike_add')
            np.testing.assert_array_equal(raw[i].detach().numpy(),expected)
            self.assertEqual(int(n[i].sum())*32,c.executed_synaptic_ac)

    def test_prefix_batching_and_empty_input(self):
        x=self.data.roles['train_main']['features'];raw,n=self.model.prefixes(x)
        parts=[self.model.prefixes(v) for v in x.split(19)]
        self.assertTrue(torch.equal(raw,torch.cat([p[0] for p in parts],1)))
        self.assertTrue(torch.equal(n,torch.cat([p[1] for p in parts],1)))
        empty,_=self.model.prefixes(x[:0]);self.assertEqual(tuple(empty.shape),(3,0,4))

    def test_joint_loss_is_mean_and_teacher_detached(self):
        raw=torch.randn(3,7,4,requires_grad=True);teacher=torch.randn(7,4,requires_grad=True)
        labels=torch.tensor([0,1,0,0,1,1,0]);terms,per=joint_losses(raw,labels,teacher,self.config)
        explicit=torch.stack([torch.stack(losses(r,labels,teacher,self.config)) for r in raw])
        torch.testing.assert_close(per,explicit,rtol=0,atol=0)
        torch.testing.assert_close(terms,explicit.sum(0)/3,rtol=0,atol=0)
        terms[0].backward();self.assertIsNone(teacher.grad)
        self.assertTrue(all(bool(torch.count_nonzero(g)) for g in raw.grad))

    def test_t16_prefix_loss_and_gradient_equal_original(self):
        x=self.data.batch('train_main',slice(0,32));old=R1T16();old.load_state_dict(self.origin['model'])
        a,_=old(x['features']);b,_=self.model.prefixes(x['features'])
        self.assertTrue(torch.equal(a,b[2]))
        old.zero_grad();self.model.zero_grad()
        losses(a,x['labels'],x['teacher'],self.config)[0].backward()
        losses(b[2],x['labels'],x['teacher'],self.config)[0].backward()
        for p,q in zip(old.parameters(),self.model.parameters()):torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)

    def test_shared_graph_gradient_matches_three_independent_losses(self):
        batch=self.data.batch('train_main',slice(0,32));self.model.zero_grad()
        raw,_=self.model.prefixes(batch['features'])
        joint_losses(raw,batch['labels'],batch['teacher'],self.config)[0][0].backward()
        expected=[]
        for i in range(3):
            separate=R1Multi();separate.load_state_dict(self.origin['model'])
            r,_=separate.prefixes(batch['features'])
            (losses(r[i],batch['labels'],batch['teacher'],self.config)[0]/3).backward()
            expected.append([p.grad.clone() for p in separate.parameters()])
        for j,p in enumerate(self.model.parameters()):
            torch.testing.assert_close(p.grad,sum(row[j] for row in expected),rtol=2e-6,atol=1e-7)

    def test_updated_export_remains_all_prefix_compatible(self):
        batch=self.data.batch('train_main',slice(0,32));scales=[s.copy() for _,s in self.model.export_layers()]
        opt=torch.optim.Adam(self.model.parameters(),lr=1e-4,foreach=False)
        update(self.model,opt,batch,self.config,'multit')
        with self.temp() as tmp:
            p=Path(tmp)/'weights.npz';export(p,self.model);layers=load_export(p)
            raw,n=self.model.prefixes(batch['features'])
            for i,t in enumerate(STEPS):
                y,c=forward(batch['features'].numpy(),layers,'r1',t,(Encoder(255,191),Encoder()),'spike_add')
                np.testing.assert_array_equal(raw[i].detach().numpy(),y)
                self.assertEqual(int(n[i].sum())*32,c.executed_synaptic_ac)
            for (_,s),old in zip(layers,scales):np.testing.assert_array_equal(s,old)

    def test_exact_same_master_full_adam_and_rng_initialization(self):
        before=digest(self.origin)
        with self.temp() as tmp:
            path=Path(tmp);ready=prepare(path,self.fx)
            for arm in ARMS:
                p=load(path/arm/'initial.pt')
                for k in ('model','optimizer','rng'):self.assertEqual(digest(p[k]),digest(self.origin[k]))
                self.assertEqual(p['state']['best_adaptive_epoch'],0)
            self.assertEqual(ready['initial_hashes']['control'],ready['initial_hashes']['multit'])
        self.assertEqual(digest(self.origin),before)

    def test_synthetic_checkpoint_resume_does_not_alias_source_adam(self):
        before=digest(self.origin)
        with self.temp() as tmp:
            r=synthetic_resume(torch.device('cpu'),Path(tmp),self.origin,self.identity,self.baseline,self.config)
        self.assertTrue(all(r[a]['model_adam_rng_bit_exact'] for a in ARMS))
        self.assertEqual(digest(self.origin),before)

    def test_mid_batch_resume_both_arms_matches_uninterrupted(self):
        with self.temp() as tmp:
            full,resumed=Path(tmp)/'full',Path(tmp)/'resumed';ready=prepare(full,self.fx);prepare(resumed,self.fx)
            for arm in ARMS:
                self.arm(full,arm,ready)
                p=self.arm(resumed,arm,ready,_test_pause_after=2)
                self.assertEqual(p['status'],'paused');self.assertEqual(p['cursor']['batch'],2)
                self.arm(resumed,arm,ready)
                self.same(load(full/arm/'last.pt'),load(resumed/arm/'last.pt'))

    def test_control_reproduces_original_optimizer_and_absolute_order(self):
        with self.temp() as tmp:
            path=Path(tmp);ready=prepare(path,self.fx);self.arm(path,'control',ready)
            model=R1T16();model.load_state_dict(self.origin['model'])
            opt=torch.optim.Adam(model.parameters(),lr=1e-4,foreach=False)
            opt.load_state_dict(deepcopy(self.origin['optimizer']));set_rng(self.origin['rng'])
            for epoch in (160,161):
                order=self.data.order(epoch,42)
                for lo in range(0,131,32):old_update(model,opt,self.data.batch('train_main',order[lo:lo+32]),self.config)
            saved=load(path/'control/last.pt')
            self.assertEqual(digest(model.state_dict()),digest(saved['model']))
            self.assertEqual(digest(opt.state_dict()),digest(saved['optimizer']))
            self.assertEqual(digest(all_rng()),digest(saved['rng']))

    def test_arms_use_identical_batches_but_different_objective_updates(self):
        with self.temp() as tmp:
            path=Path(tmp);ready=prepare(path,self.fx);seen=[]
            def record(model,opt,batch,cfg,arm):
                seen.append((arm,digest(batch)));return update(model,opt,batch,cfg,arm)
            with patch('wangdao.h2pro_multit_v1.runtime.update',side_effect=record):
                for a in ARMS:self.arm(path,a,ready,1)
            self.assertEqual([x[1] for x in seen[:5]],[x[1] for x in seen[5:]])
            self.assertNotEqual(digest(load(path/'control/last.pt')['model']),digest(load(path/'multit/last.pt')['model']))

    def test_suite_stage_order_predictions_and_comparisons(self):
        with self.temp() as tmp:
            path=Path(tmp);prepare(path,self.fx);calls=[]
            def record(*a,**kw):calls.append((a[2],a[6]));return train_arm(*a,**kw)
            with patch('wangdao.h2pro_multit_v1.runtime.MILESTONES',(1,2)),patch('wangdao.h2pro_multit_v1.runtime.train_arm',side_effect=record):
                self.assertEqual(self.suite(path)['status'],'complete')
            self.assertEqual(calls,[(a,n) for n in (1,2) for a in ARMS])
            for n in (1,2):
                c=read_json(path/f'comparison_added_{n:03}.json')
                self.assertEqual(set(c['multit_minus_control_pp']),{'T4','T8','T16','adaptive'})
                for arm in ARMS:
                    p=load(path/arm/'epochs'/f'added_{n:03}.pt')
                    self.assertEqual(p['state']['global_step'],2+5*n)
                    self.assertTrue((path/arm/'epochs'/f'added_{n:03}_predictions.npz').is_file())
                    self.assertIn('best_adaptive_added_epoch',c['arms'][arm])

    def test_whole_suite_pause_then_resume(self):
        with self.temp() as tmp:
            path=Path(tmp);prepare(path,self.fx)
            def request(*a):
                r=update(*a);(path/'PAUSE').write_text('synthetic pause');return r
            with patch('wangdao.h2pro_multit_v1.runtime.update',side_effect=request):
                self.assertEqual(self.suite(path,1)['status'],'paused')
            self.assertEqual(load(path/'control/last.pt')['state']['global_step'],3)
            self.assertEqual(load(path/'multit/last.pt')['state']['global_step'],2)
            self.assertEqual(self.suite(path,1)['status'],'complete')

    def test_evaluation_pause_never_repeats_training_updates(self):
        with self.temp() as tmp:
            path=Path(tmp);ready=prepare(path,self.fx)
            with patch('wangdao.h2pro_multit_v1.runtime.evaluate',side_effect=PauseRequested()):
                p=self.arm(path,'multit',ready,1)
            self.assertEqual(p['cursor']['phase'],'evaluate')
            with patch('wangdao.h2pro_multit_v1.runtime.update',side_effect=AssertionError('No repeat training')):
                self.arm(path,'multit',ready,1)
            self.assertEqual(load(path/'multit/last.pt')['state']['global_step'],7)

    def test_nonfinite_failure_keeps_valid_committed_checkpoint(self):
        with self.temp() as tmp:
            path=Path(tmp);prepare(path,self.fx);calls=[]
            def fail(*a):
                calls.append(1)
                if len(calls)==2:
                    with torch.no_grad():a[0].weights[0][0,0]=float('nan')
                    raise FloatingPointError('synthetic fault')
                return update(*a)
            with patch('wangdao.h2pro_multit_v1.runtime.update',side_effect=fail):
                with self.assertRaises(FloatingPointError):self.suite(path,1)
            saved=load(path/'control/last.pt');self.assertEqual(saved['state']['global_step'],3)
            self.assertTrue(all(bool(torch.isfinite(v).all()) for v in saved['model'].values()))
            self.assertEqual(status(path)['workflow_status'],'failed')
            self.assertEqual(self.suite(path,1)['status'],'complete')

    def test_completed_budget_reuse_and_history_repair(self):
        with self.temp() as tmp:
            path=Path(tmp);prepare(path,self.fx)
            def fail(target,value):
                if target.name=='training.json':raise OSError('synthetic write fault')
                return atomic_json(target,value)
            with patch('wangdao.h2pro_multit_v1.runtime.atomic_json',side_effect=fail):
                with self.assertRaises(OSError):self.suite(path,1)
            self.suite(path,1);before={a:sha256(path/a/'last.pt') for a in ARMS}
            with patch('wangdao.h2pro_multit_v1.runtime.update',side_effect=AssertionError('Already complete')):
                self.suite(path,1)
            self.assertEqual(before,{a:sha256(path/a/'last.pt') for a in ARMS})

    def test_preflight_reuse_never_resets_trained_checkpoint(self):
        with self.temp() as tmp:
            path=Path(tmp);ready=prepare(path,self.fx);self.suite(path,1)
            before={a:sha256(path/a/'last.pt') for a in ARMS}
            with patch('wangdao.h2pro_multit_v1.checks.inputs',return_value=(self.identity,self.origin,self.config,{})), \
                 patch('wangdao.h2pro_multit_v1.checks.ResidentData',side_effect=AssertionError('No real read')):
                self.assertEqual(check(suite=path,device_name='cpu'),ready)
            self.assertEqual(before,{a:sha256(path/a/'last.pt') for a in ARMS})

    def test_identity_cursor_optimizer_and_output_safety(self):
        with self.temp() as tmp:
            path=Path(tmp);ready=prepare(path,self.fx)
            p=load(path/'multit/last.pt');p['optimizer']['param_groups'][0]['lr']=3e-5
            save_atomic(path/'multit/last.pt',p)
            with self.assertRaisesRegex(ValueError,'learning rate'):self.arm(path,'multit',ready,1)
            (path/'control/initial.pt').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):verify_prepared(path,self.identity)
        p=initialized_payload(self.origin,arm_identity(self.identity,'control'),self.baseline)
        p['state']['global_step']+=1
        with self.assertRaises(ValueError):validate_cursor(p['state'],self.data,self.config)
        for n in (0,-1,True):
            with self.assertRaises(ValueError):self.suite(Path('unused'),n)
        with self.assertRaises(ValueError):independent_output(IMPLEMENTATION/'results/runs',IMPLEMENTATION/'results/runs/source')

    def test_killed_process_releases_lock_and_restores_exact_state(self):
        with self.temp() as tmp:
            full,stopped=Path(tmp)/'full',Path(tmp)/'stopped';prepare(full,self.fx);prepare(stopped,self.fx)
            self.suite(full,2)
            code='''import sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'software'))
from software.tests.test_h2pro_r1_train import toy_data
from wangdao.h2pro_multit_v1.common import Config,read_json
import wangdao.h2pro_multit_v1.runtime as rt
cfg=Config(batch_size=32);data=toy_data(cfg);p=Path(sys.argv[1]);ident=read_json(p/'identity.json');old=rt.update
def slow(*a):
 time.sleep(.15)
 return old(*a)
rt.update=slow
rt.train_suite(suite=p,additional_epochs=2,checkpoint_batches=1,device_name='cpu',_test_context=(ident,cfg,data))
'''
            with (Path(tmp)/'child.log').open('w') as log:
                child=subprocess.Popen([sys.executable,'-B','-c',code,str(stopped)],cwd=IMPLEMENTATION,
                    stdout=log,stderr=log,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                try:
                    deadline=time.monotonic()+30
                    while time.monotonic()<deadline:
                        if read_json(stopped/'control/progress.json').get('additional_updates',0)>=2:break
                        if child.poll() is not None:self.fail('Synthetic child stopped before checkpoint')
                        time.sleep(.03)
                    else:self.fail('No synthetic checkpoint')
                    child.kill();child.wait(timeout=10)
                finally:
                    if child.poll() is None:child.kill();child.wait(timeout=10)
            self.suite(stopped,2)
            for arm in ARMS:self.same(load(full/arm/'last.pt'),load(stopped/arm/'last.pt'))


if __name__=='__main__':unittest.main()
