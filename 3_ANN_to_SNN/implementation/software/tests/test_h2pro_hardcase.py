"""Direction5 objective, paired initialization, isolation and recovery regressions.

All optimizer updates in this file use synthetic random features/labels only.
"""
from __future__ import annotations

import copy
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from wangdao.h2pro_direct_v1.core import forward, load_layers
from wangdao.h2pro_distill_v1.data import ResidentData
from wangdao.h2pro_distill_v1.model import R2T8, losses, update as original_update
from wangdao.h2pro_distill_v1.evaluation import evaluate
from wangdao.h2pro_distill_v1.runtime import setup_device
from wangdao.h2pro_longtrain_v1.storage import load, save_atomic, all_rng, PauseRequested
from wangdao.h2pro_hardcase_v1.common import (Config, IMPLEMENTATION, ARMS, atomic_json,
    lock_directory, arm_identity, digest, sha256, save_arrays, read_json)
from wangdao.h2pro_hardcase_v1.objective import categories, weighted_losses, update
from wangdao.h2pro_hardcase_v1 import runtime
from wangdao.h2pro_hardcase_v1.runtime import initialized_payload, train_arm, attach_hard_categories


def toy_data():
    data=ResidentData.__new__(ResidentData)
    data.device=torch.device('cpu'); data.roles={}; data.sequences={}; data.stats={'synthetic_only':True}
    rng=np.random.default_rng(915)
    for role in ('train_main','train_dev'):
        xs,ys,teachers,seqs=[],[],[],[]; offset=0
        for i,n in enumerate((21,26)):
            x=rng.integers(0,256,(n,36),dtype=np.uint8)
            teacher,_=forward(x,load_layers())
            labels=rng.integers(0,2,n+3,dtype=np.uint8); labels[-3:]=[1,1,0]
            candidate=np.r_[np.ones(n,bool),np.zeros(3,bool)]
            row={'sequence':f'synthetic_{role}_{i}','role':role,'events':n+3,'high_density':bool(i),
                 'candidate_events':n,'candidate_targets':int(labels[candidate].sum()),
                 'target_events':int(labels.sum()),'proposal_sparse_mac':40,'proposal_dense_mac':80}
            seqs.append({'row':row,'labels':labels,'candidate':candidate,'lo':offset,'hi':offset+n})
            offset+=n; xs.append(x); ys.append(labels[candidate]); teachers.append(teacher)
        data.roles[role]={k:torch.from_numpy(np.concatenate(v)) for k,v in
                          [('features',xs),('labels',ys),('teacher',teachers)]}
        data.sequences[role]=seqs
    return data


def toy_suite(directory,config):
    data=toy_data(); model=R2T8()
    optimizer=torch.optim.Adam(model.parameters(),lr=config.learning_rate,weight_decay=0.,foreach=False)
    # Nonzero moments and step counter make the inheritance check substantive.
    for _ in range(3):
        original_update(model,optimizer,data.batch('train_main',slice(None)),config)
    origin={'model':copy.deepcopy(model.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
            'rng':all_rng(),'state':{'global_step':3}}
    baseline,_,_=evaluate(model,data,config.batch_size)
    identity={'synthetic_suite':True,'config':config.__dict__}
    with lock_directory(directory,identity):
        category=np.where(data.roles['train_main']['labels'].numpy()==1,1,2).astype(np.uint8)
        category[::3]=0
        save_arrays(directory/'main_hard_categories.npz',category=category)
        ready={'status':'passed','identity':identity,'initial_student':baseline,'frozen_ann':baseline,
               'hard_categories_sha256':sha256(directory/'main_hard_categories.npz')}
        atomic_json(directory/'ready.json',ready)
        for arm in ARMS:
            aid=arm_identity(identity,arm,ready['hard_categories_sha256'])
            with lock_directory(directory/arm,aid):
                payload=initialized_payload(origin,aid,baseline)
                for name in ('initial.pt','last.pt','best.pt'):
                    save_atomic(directory/arm/name,payload)
                atomic_json(directory/arm/'progress.json',{'status':'ready_not_trained'})
    attach_hard_categories(data,directory,ready)
    return data,identity,ready,origin


class Direction5Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setup_device('cpu')
        cls.config=Config(batch_size=16)

    def temporary(self):
        return tempfile.TemporaryDirectory(dir=IMPLEMENTATION/'results/runs')

    def test_hard_positive_negative_and_teacher_wrong_exclusion(self):
        def raw(pred):
            return np.array([[0,0,2,2] if p else [2,2,0,0] for p in pred],np.float32)
        y=np.array([1,0,1,0,1,0],np.uint8)
        c=categories(raw([1,0,0,1,1,0]),raw([0,1,0,1,1,0]),y,role='train_main')
        np.testing.assert_array_equal(c,[1,2,0,0,0,0])
        with self.assertRaisesRegex(ValueError,'Main-only'):
            categories(raw([1]*6),raw([0]*6),y,role='train_dev')

    def test_weighted_objective_matches_integer_replication_and_gradients(self):
        raw=torch.tensor([[1.,2.,3.,4.],[5.,3.,2.,-1.],[1.,0.,-2.,3.]],requires_grad=True)
        teacher=torch.tensor([[3.,2.,1.,0.],[2.,1.,4.,3.],[4.,2.,1.,0.]],requires_grad=True)
        y=torch.tensor([1,0,1]); w=torch.tensor([2.,1.,2.])
        actual=weighted_losses(raw,y,teacher,w,self.config)
        alternate=raw.detach().clone().requires_grad_(True)
        ids=torch.tensor([0,0,1,2,2])
        expected=losses(alternate[ids],y[ids],teacher[ids],self.config)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
        actual[0].backward(); expected[0].backward()
        torch.testing.assert_close(raw.grad,alternate.grad,atol=1e-6,rtol=1e-6)
        self.assertIsNone(teacher.grad)

    def test_uniform_weights_preserve_original_loss_and_gradient(self):
        raw=torch.randn(7,4,requires_grad=True); teacher=torch.randn(7,4); y=torch.arange(7)%2
        copy_raw=raw.detach().clone().requires_grad_(True)
        a=weighted_losses(raw,y,teacher,torch.ones(7),self.config)
        b=losses(copy_raw,y,teacher,self.config)
        torch.testing.assert_close(torch.stack(a),torch.stack(b))
        a[0].backward(); b[0].backward(); torch.testing.assert_close(raw.grad,copy_raw.grad)

    def test_control_update_is_bit_exact_to_original(self):
        data=toy_data(); batch=data.batch('train_main',slice(None))
        a,b=R2T8(),R2T8()
        oa=torch.optim.Adam(a.parameters(),lr=1e-4,foreach=False)
        ob=torch.optim.Adam(b.parameters(),lr=1e-4,foreach=False)
        for _ in range(3):
            update(a,oa,batch,self.config,'control')
            original_update(b,ob,batch,self.config)
        self.assertEqual(digest(a.state_dict()),digest(b.state_dict()))
        self.assertEqual(digest(oa.state_dict()),digest(ob.state_dict()))

    def test_same_nonzero_adam_master_weights_rng_and_absolute_permutation(self):
        with self.temporary() as temp:
            out=Path(temp)/'suite'; data,identity,ready,origin=toy_suite(out,self.config)
            for arm in ARMS:
                payload=load(out/arm/'initial.pt')
                for k in ('model','optimizer','rng'):
                    self.assertEqual(digest(payload[k]),digest(origin[k]))
                self.assertEqual(payload['state']['global_step'],3)
            expected=np.random.default_rng(np.random.SeedSequence([42,89])).permutation(47)
            np.testing.assert_array_equal(data.order(89,42).numpy(),expected)

    def test_both_arms_batch_resume_are_bit_exact_and_masks_are_fixed(self):
        for arm in ARMS:
            with self.subTest(arm=arm), self.temporary() as temp:
                root=Path(temp); full=root/'full'; paused=root/'paused'
                data,identity,ready,_=toy_suite(full,self.config)
                other,other_id,other_ready,_=toy_suite(paused,self.config)
                category_hash=sha256(paused/'main_hard_categories.npz')
                train_arm(data,full,arm,identity,ready,self.config,2,1)
                r=train_arm(other,paused,arm,other_id,other_ready,self.config,2,1,_test_pause_after=2)
                self.assertEqual(r['status'],'paused')
                train_arm(other,paused,arm,other_id,other_ready,self.config,2,1)
                a,b=load(full/arm/'last.pt'),load(paused/arm/'last.pt')
                for key in ('model','optimizer'):
                    self.assertEqual(digest(a[key]),digest(b[key]))
                self.assertNotEqual(digest(b['model']),digest(load(paused/arm/'initial.pt')['model']))
                self.assertEqual(b['state']['global_step'],9)
                self.assertEqual(len(b['state']['history']),2)
                self.assertEqual(category_hash,sha256(paused/'main_hard_categories.npz'))

    def test_evaluation_pause_reuses_main_selection_and_does_not_repeat_updates(self):
        with self.temporary() as temp:
            out=Path(temp)/'suite'; data,identity,ready,_=toy_suite(out,self.config)
            with patch.object(runtime,'evaluate',side_effect=PauseRequested()):
                r=train_arm(data,out,'hard_weighted',identity,ready,self.config,1,1)
            self.assertEqual(r['cursor']['phase'],'evaluate')
            self.assertEqual(r['global_step'],6)
            selection=out/'hard_weighted/milestones/added_001_main.json'; before=sha256(selection)
            train_arm(data,out,'hard_weighted',identity,ready,self.config,1,1)
            self.assertEqual(load(out/'hard_weighted/last.pt')['state']['global_step'],6)
            self.assertEqual(sha256(selection),before)
            validation=read_json(out/'hard_weighted/milestones/added_001_dev.json')
            self.assertLess(read_json(selection)['created_at_utc'],validation['validation_started_at_utc'])

    def test_failure_does_not_replace_committed_weights(self):
        with self.temporary() as temp:
            out=Path(temp)/'suite'; data,identity,ready,_=toy_suite(out,self.config); calls=[]
            original=runtime.update
            def fail(model,*args):
                calls.append(1)
                if len(calls)==2:
                    with torch.no_grad():model.weights[0][0,0]=float('nan')
                    raise FloatingPointError('synthetic fault')
                return original(model,*args)
            with patch.object(runtime,'update',side_effect=fail):
                with self.assertRaises(FloatingPointError):
                    train_arm(data,out,'hard_weighted',identity,ready,self.config,1,1)
            saved=load(out/'hard_weighted/last.pt')
            self.assertEqual(saved['state']['global_step'],4)
            self.assertTrue(all(bool(torch.isfinite(t).all()) for t in saved['model'].values()))
            train_arm(data,out,'hard_weighted',identity,ready,self.config,1,1)
            self.assertEqual(load(out/'hard_weighted/last.pt')['state']['global_step'],6)

    def test_completed_checkpoint_repairs_missing_history_without_updates(self):
        with self.temporary() as temp:
            out=Path(temp)/'suite'; data,identity,ready,_=toy_suite(out,self.config)
            train_arm(data,out,'control',identity,ready,self.config,1,1)
            before=digest(load(out/'control/last.pt'))
            (out/'control/training.json').unlink()
            with patch.object(runtime,'update',side_effect=AssertionError('must not update')):
                train_arm(data,out,'control',identity,ready,self.config,1,1)
            self.assertEqual(len(read_json(out/'control/training.json')['epochs']),1)
            self.assertEqual(before,digest(load(out/'control/last.pt')))

    def test_corrupt_cursor_and_changed_mask_rejected(self):
        with self.temporary() as temp:
            out=Path(temp)/'suite'; data,identity,ready,_=toy_suite(out,self.config)
            saved=load(out/'control/last.pt'); saved['state']['cursor']['batch']=2
            save_atomic(out/'control/last.pt',saved)
            with self.assertRaisesRegex(ValueError,'cursor'):
                train_arm(data,out,'control',identity,ready,self.config,1,1)
            save_arrays(out/'main_hard_categories.npz',category=np.zeros(47,np.uint8))
            with self.assertRaisesRegex(ValueError,'mask changed'):
                attach_hard_categories(data,out,ready)

    def test_comparison_requires_both_arms_and_reuses_identical_report(self):
        with self.temporary() as temp:
            out=Path(temp)/'suite'; data,identity,ready,_=toy_suite(out,self.config)
            train_arm(data,out,'control',identity,ready,self.config,1,1)
            with self.assertRaises(ValueError):runtime.compare(out,1,ready)
            train_arm(data,out,'hard_weighted',identity,ready,self.config,1,1)
            result=runtime.compare(out,1,ready)
            self.assertIn('control_035',result['weighted_minus_control_pp'])
            before=sha256(out/'comparison_added_001.json')
            runtime.compare(out,1,ready)
            self.assertEqual(before,sha256(out/'comparison_added_001.json'))

    def test_schedule_balances_5_10_20_and_pause_stops_before_other_arm(self):
        for stop in (False,True):
            with self.subTest(stop=stop),self.temporary() as temp:
                out=Path(temp)/'suite'; data,identity,ready,origin=toy_suite(out,self.config)
                calls=[]; comparisons=[]
                def arm_call(data_arg,suite,arm,ident,ready_arg,config,target,*args):
                    calls.append((arm,target))
                    if stop:(suite/'PAUSE').write_text('synthetic request')
                    return {'status':'complete'}
                with patch.object(runtime,'inputs',return_value=(identity,origin,self.config)), \
                     patch.object(runtime,'ResidentData',return_value=data), \
                     patch.object(runtime,'train_arm',side_effect=arm_call), \
                     patch.object(runtime,'compare',side_effect=lambda suite,target,r: comparisons.append(target)):
                    result=runtime.train_suite(Path(temp)/'cache',out,Path(temp)/'source',Path(temp)/'diag',
                                               20,1,'cpu')
                self.assertEqual(calls,[('control',5)] if stop else [(a,e) for e in (5,10,20) for a in ARMS])
                self.assertEqual(comparisons,[] if stop else [5,10,20])
                self.assertEqual(result['status'],'paused' if stop else 'complete')

    def test_abrupt_process_kill_resumes_weights_and_adam_exactly(self):
        with self.temporary() as temp:
            root=Path(temp); full=root/'full'; killed=root/'killed'
            data,identity,ready,_=toy_suite(full,self.config)
            other,other_id,other_ready,_=toy_suite(killed,self.config)
            train_arm(data,full,'hard_weighted',identity,ready,self.config,2,1)
            code='''import sys,time
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/'software'))
sys.path.insert(0,str(Path.cwd()/'software/tests'))
from test_h2pro_hardcase import toy_data
from wangdao.h2pro_hardcase_v1.common import Config,read_json
from wangdao.h2pro_hardcase_v1 import runtime
runtime.setup_device('cpu')
suite=Path(sys.argv[1]); data=toy_data(); ready=read_json(suite/'ready.json')
runtime.attach_hard_categories(data,suite,ready)
original=runtime.update
def slow(*args):
    time.sleep(.25)
    return original(*args)
runtime.update=slow
runtime.train_arm(data,suite,'hard_weighted',ready['identity'],ready,Config(batch_size=16),2,1)
'''
            with (root/'child.log').open('w') as stream:
                child=subprocess.Popen([sys.executable,'-B','-c',code,str(killed)],cwd=IMPLEMENTATION,
                                       stdout=stream,stderr=stream)
                try:
                    deadline=time.monotonic()+30
                    while time.monotonic()<deadline:
                        p=read_json(killed/'hard_weighted/progress.json')
                        if p.get('additional_updates',0)>=2:break
                        if child.poll() is not None:self.fail('Synthetic child ended before kill')
                        time.sleep(.03)
                    else:self.fail('Synthetic child did not commit')
                    child.kill(); child.wait(timeout=10)
                finally:
                    if child.poll() is None:child.kill(); child.wait(timeout=10)
            train_arm(other,killed,'hard_weighted',other_id,other_ready,self.config,2,1)
            a,b=load(full/'hard_weighted/last.pt'),load(killed/'hard_weighted/last.pt')
            for key in ('model','optimizer'):
                self.assertEqual(digest(a[key]),digest(b[key]))


if __name__=='__main__':
    unittest.main()
