"""Numerics, full-event coverage, atomic recovery; updates use random fixtures only."""
import contextlib
import copy
import io
import os
from pathlib import Path
import signal
import sys
import unittest
from unittest import mock

for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from dynamic_ann.common import Config,FLOAT_SOURCE,DEFAULT_CACHE,read
from dynamic_ann.model import (Refiner,assert_initial_export,logits,supervised,optimizer_for,
                               update,exact_forward,exact_layers_on_device)
from dynamic_ann.data import verify_cache
from dynamic_ann.evaluation import score,evaluate
from dynamic_ann.runtime import (setup_device,initial_state,checkpoint,restore,validate_state,
                                 run_epochs,train)
from dynamic_ann.storage import (atomic_checkpoint,load_checkpoint,process_lock,PauseControl,
                                 PauseRequested,rng_state,seed_all)
from dynamic_ann.synthetic import synthetic_data,fixture_directory,equal_tree,resume_check,NeverPause
from dynamic_ann.preservation import verified_concurrent_changes
from wangdao.h2.numpy_model import quantized_mlp

CPU=setup_device('cpu')


class Numerics(unittest.TestCase):
    def setUp(self):
        seed_all(71)
        self.model=Refiner()
        self.x=torch.from_numpy(np.random.default_rng(8).integers(0,256,(2053,36),dtype=np.uint8))

    def test_original_weights_structure_float_forward(self):
        old=torch.load(FLOAT_SOURCE,map_location='cpu',weights_only=True)['state_dict']
        equal_tree(self.model.state_dict(),old)
        self.assertEqual(sum(p.numel() for p in self.model.parameters()),4480)
        self.assertEqual(set(old),{'old.weight','new.weight','hidden.weight','output.weight'})
        x=self.x.float()/255.
        other=torch.cat((x[:,:2],x[:,3:]),dim=1)
        hidden=F.hardtanh(F.linear(other,old['old.weight'])+F.linear(x[:,2:3],old['new.weight']),0,1)
        hidden=F.hardtanh(F.linear(hidden,old['hidden.weight']),0,1)
        torch.testing.assert_close(self.model(self.x),F.linear(hidden,old['output.weight']),rtol=0,atol=0)
        self.assertTrue(assert_initial_export(self.model))

    def test_int8_forward_matches_original_int32_backend(self):
        layers=self.model.export_layers()
        expected=quantized_mlp(self.x.numpy(),layers)
        actual=exact_forward(self.x,exact_layers_on_device(layers,CPU)).numpy()
        np.testing.assert_array_equal(actual,expected)

    def test_original_bce_and_ce_objective_and_gradients(self):
        raw=self.model(self.x);scores=logits(raw)
        y=torch.arange(len(raw))%2
        ce=F.cross_entropy(scores,y)
        old=F.binary_cross_entropy(torch.softmax(scores,dim=1)[:,1],y.float())
        torch.testing.assert_close(ce,old,rtol=1e-6,atol=1e-7)
        a=torch.autograd.grad(ce,tuple(self.model.parameters()),retain_graph=True)
        b=torch.autograd.grad(old,tuple(self.model.parameters()))
        for x,z in zip(a,b):torch.testing.assert_close(x,z,rtol=1e-4,atol=2e-6)

    def test_export_range_follows_weights_without_clipping_training(self):
        before=self.model.export_layers()
        with torch.no_grad():self.model.output.weight.mul_(4)
        after=self.model.export_layers()
        np.testing.assert_array_equal(after[-1][0],before[-1][0])
        np.testing.assert_array_equal(after[-1][1],before[-1][1]*4)
        raw=self.model(self.x)
        self.assertTrue(torch.isfinite(raw).all())

    def test_synthetic_update_all_original_matrices(self):
        before=copy.deepcopy(self.model.state_dict())
        optimizer=optimizer_for(self.model,Config())
        result=update(self.model,optimizer,self.x,torch.arange(len(self.x))%2)
        self.assertEqual(result['candidate_events'],len(self.x))
        for key,value in self.model.state_dict().items():self.assertFalse(torch.equal(value,before[key]),key)
        self.assertEqual(len(optimizer.state),4)


class CoverageAndCost(unittest.TestCase):
    def test_different_candidates_same_source_budget_and_complete_coverage(self):
        a=synthetic_data(CPU,'fixed100');b=synthetic_data(CPU,'dynamic2000')
        cfg=Config(source_batch_size=64)
        self.assertNotEqual(a.candidate_count,b.candidate_count)
        self.assertEqual(a.batches(cfg),b.batches(cfg))
        order=a.order(1,42)
        torch.testing.assert_close(order,b.order(1,42),rtol=0,atol=0)
        self.assertFalse(torch.equal(order,a.order(2,42)))
        for d in (a,b):
            seen=[];source=[];n=0
            for start in range(0,d.source_count,64):
                ids=order[start:start+64];source.extend(ids.tolist())
                mapped=d.roles['train_main']['mapping'][ids]
                seen.extend(mapped[mapped>=0].tolist())
                x,y=d.batch(ids);self.assertEqual(len(x),len(y));n+=len(x)
            self.assertEqual(sorted(source),list(range(d.source_count)))
            self.assertEqual(sorted(seen),list(range(d.candidate_count)))
            self.assertEqual(n,d.candidate_count)

    def test_proposal_rejected_targets_in_metrics_and_cost(self):
        d=synthetic_data(CPU);n=len(d.roles['train_dev']['features'])
        result=score(d,np.ones(n,np.float32))
        target=sum(int(s['full_labels'].sum()) for s in d.sequences['train_dev'])
        targets_in_candidates=int(d.roles['train_dev']['labels'].sum())
        self.assertEqual(result['overall']['events'],98)
        self.assertEqual(result['overall']['fn'],target-targets_in_candidates)
        self.assertEqual(result['overall']['tp'],targets_in_candidates)
        self.assertGreater(result['overall']['fn'],0)
        self.assertEqual(result['cost']['mac_sparse_proposal'],2*16+n*4480)
        self.assertEqual(result['cost']['mac_dense_proposal'],2*32+n*4480)
        self.assertEqual(result['cost']['ac'],0)
        d.sequences['train_dev'][0]['row']['arms']['fixed100']['cost']['proposal_history_extra_sparse_mac']=79
        self.assertEqual(score(d,np.ones(n,np.float32))['cost']['mac_sparse_proposal'],result['cost']['mac_sparse_proposal']+79)

    def test_full_real_cache_roles_counts_hashes(self):
        m=verify_cache(DEFAULT_CACHE)
        self.assertEqual(set(m['roles']),{'train_main','train_dev'})
        self.assertEqual(sum(len(v) for v in m['roles'].values()),89)
        self.assertEqual(m['stats']['train_main']['source_events'],7154631)
        self.assertEqual(m['stats']['train_dev']['source_events'],1221626)
        self.assertEqual(m['stats']['train_main']['candidates'],{'fixed100':2219864,'dynamic2000':3143348})
        self.assertEqual(m['stats']['train_dev']['candidates'],{'fixed100':438966,'dynamic2000':625310})
        self.assertEqual((7154631+32767)//32768,219)
        self.assertFalse(m['reserved_sequences_read'])
        self.assertEqual(m['real_data_optimizer_steps'],0)


class Recovery(unittest.TestCase):
    def test_concurrent_edit_record_cannot_hide_unreviewed_change(self):
        with self.assertRaises(ValueError):verified_concurrent_changes(['unknown'],{'files':{}},{})

    def test_concurrent_edit_record_cannot_exempt_ann_dependency(self):
        p=str(FLOAT_SOURCE)
        with self.assertRaises(ValueError):
            verified_concurrent_changes([p],{'files':{p:{'before':{},'after':{}}}},{p:{}})

    def test_production_loop_pause_and_epoch_commit_power_loss_exact(self):
        with contextlib.redirect_stdout(io.StringIO()):r=resume_check(CPU)
        self.assertTrue(r['model_adam_rng_exact'])
        self.assertTrue(r['epoch_commit_power_loss_exact'])
        self.assertTrue(r['epoch_export_exact'])

    def test_wrong_identity_and_arm_refused(self):
        p=fixture_directory('wrong_identity')/'last.pt';m=Refiner();o=optimizer_for(m,Config())
        checkpoint(p,m,o,{'value':1},{'tag':'original'},'fixed100')
        with self.assertRaises(ValueError):restore(p,m,o,{'tag':'different'},'fixed100')
        with self.assertRaises(ValueError):restore(p,m,o,{'tag':'original'},'dynamic2000')

    def test_invalid_cursor_refused(self):
        d=synthetic_data(CPU);m=Refiner();cfg=Config(source_batch_size=64)
        initial,_,_=evaluate(m,d,64);state=initial_state(initial)
        validate_state(state,d,cfg)
        for field,value in [('batch',1),('epoch',2),('phase','evaluate')]:
            bad=copy.deepcopy(state);bad['cursor'][field]=value
            with self.assertRaises(ValueError):validate_state(bad,d,cfg)

    def test_nonfinite_failure_retains_committed_checkpoint(self):
        p=fixture_directory('nonfinite');d=synthetic_data(CPU);m=Refiner();cfg=Config(source_batch_size=64)
        o=optimizer_for(m,cfg);initial,_,_=evaluate(m,d,64);state=initial_state(initial)
        checkpoint(p/'last.pt',m,o,state,{'test':'nonfinite'},'fixed100')
        before=(p/'last.pt').read_bytes()
        with torch.no_grad():m.output.weight.fill_(float('nan'))
        with contextlib.redirect_stdout(io.StringIO()),self.assertRaises(FloatingPointError):
            run_epochs(p,'fixed100',m,o,state,{'test':'nonfinite'},d,cfg,initial,NeverPause(),1,1)
        self.assertEqual((p/'last.pt').read_bytes(),before)
        self.assertEqual(len(list(p.glob('failure_*.json'))),1)

    def test_atomic_failed_replace_keeps_previous_checkpoint(self):
        p=fixture_directory('atomic')/'last.pt';atomic_checkpoint(p,{'step':5})
        with mock.patch('dynamic_ann.storage.os.replace',side_effect=OSError('injected synthetic disk error')):
            with self.assertRaises(OSError):atomic_checkpoint(p,{'step':6})
        self.assertEqual(load_checkpoint(p),{'step':5})

    def test_run_lock_exclusive_and_released(self):
        p=fixture_directory('lock')
        with process_lock(p):
            with self.assertRaises(RuntimeError):
                with process_lock(p):pass
        with process_lock(p):pass

    def test_pause_file_and_signal_are_deferred_to_boundary(self):
        p=fixture_directory('pause');old=signal.getsignal(signal.SIGINT)
        with PauseControl(p) as c:
            c.check();signal.getsignal(signal.SIGINT)(signal.SIGINT,None)
            self.assertTrue(c.requested)
            with self.assertRaises(PauseRequested):c.check()
        self.assertIs(signal.getsignal(signal.SIGINT),old)
        (p/'PAUSE').write_text('test',encoding='utf-8')
        with PauseControl(p) as c:
            with self.assertRaises(PauseRequested):c.check()

    def test_pause_stops_entire_paired_suite(self):
        # Real suite orchestrator with synthetic loader/loop stub: no real update.
        p=fixture_directory('suite_pause');cfg=Config(source_batch_size=64)
        d=synthetic_data(CPU);initial,_,_=evaluate(Refiner(),d,64)
        from dynamic_ann.common import atomic_json,sha
        m=Refiner();o=optimizer_for(m,cfg)
        atomic_checkpoint(p/'initial.pt',dict(identity={'test':True},model=m.state_dict(),optimizer=o.state_dict(),rng=rng_state()))
        atomic_json(p/'ready.json',dict(identity={'test':True},status='passed',initial_checkpoint_sha256=sha(p/'initial.pt'),
            arms={'fixed100':{'initial':initial},'dynamic2000':{'initial':initial}}))
        with mock.patch('dynamic_ann.runtime.verify_cache',return_value={}),mock.patch('dynamic_ann.runtime.run_identity',return_value={'test':True}),\
             mock.patch('dynamic_ann.runtime.ResidentData',return_value=d) as loader,mock.patch('dynamic_ann.runtime.run_epochs',return_value='paused') as loop:
            train(DEFAULT_CACHE,p,cfg,1,1,'cpu')
        self.assertEqual(loader.call_count,1);self.assertEqual(loop.call_count,1)
        self.assertFalse((p/'dynamic2000'/'last.pt').exists())


if __name__=='__main__':unittest.main(verbosity=2)
