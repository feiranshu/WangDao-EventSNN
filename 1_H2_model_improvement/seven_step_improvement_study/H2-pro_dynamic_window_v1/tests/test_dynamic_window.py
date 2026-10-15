"""Behavioral checks: causal boundaries, exact baseline, physical history and evidence."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import json
import sys
from contextlib import contextmanager
from uuid import uuid4
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.core import (Policy, BoundaryController, schedule, exact_mlp, SequenceContext,
                      EVENT_DTYPE, ANN_ROOT, SNN_ROOT, counts, metrics, POLICIES)
from src.workflow import (save, read, sha, digest, verified_record, select, load_sequence,
                          run_stage, manifests, RAW, FEATURES)
from wangdao.h2.numpy_model import quantized_mlp
from wangdao.h2_pro.causal_feature_builder import base_features


@contextmanager
def test_directory():
    # Keep tiny evidence fixtures inside the authorized workspace. Windows
    # sandbox system-temp ACLs can prohibit reopening a freshly created folder.
    root=Path(__file__).resolve().parents[1]/'results/checks/fixtures'
    path=root/str(uuid4())
    path.mkdir(parents=True)
    yield path


class DynamicWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw,feat=manifests()
        row=raw['roles']['train_main'][0]
        frow=next(x for x in feat['roles']['train_main'] if x['sequence']==row['sequence'])
        cls.fixture=load_sequence('train_main',row,frow)
        e,_,_,_,_,meta=cls.fixture
        cls.short=e[e['t']<meta['t0_ms']+400].copy()
        cls.t0=meta['t0_ms']

    def test_online_half_open_boundaries_and_empty_bins(self):
        times=np.array([0.,24.999,25.,25.,99.99,100.,199.9])
        rows=schedule(times,0.,2,Policy('q',event_budget=2))
        self.assertEqual(rows,[(0,1,2),(1,2,2),(2,4,1),(4,8,2)])
        self.assertEqual(schedule(np.array([]),0.,2,Policy('q',event_budget=2)),[(0,4,0),(4,8,0)])

    def test_controller_never_uses_future_counts(self):
        a=BoundaryController(Policy('a',event_budget=10));b=BoundaryController(Policy('a',event_budget=10))
        self.assertEqual([a.advance(n) for n in (3,4,8)],[b.advance(n) for n in (3,4,8)])
        self.assertEqual(a.advance(0),(3,4,0));self.assertEqual(b.advance(1000),(3,4,1000))

    def test_invalid_policies_and_input_rejected(self):
        for p in (Policy('x'),Policy('x',25,100),Policy('x',15),Policy('x',0,-1),Policy('x',25,0,'unknown')):
            with self.assertRaises(ValueError):p.validate()
        with self.assertRaises(ValueError):BoundaryController(Policy('x',25)).advance(-1)
        bad=self.short.copy();bad['t'][0]=np.nan
        with self.assertRaises(ValueError):SequenceContext(bad,self.t0,4)

    def test_exact_blas_mlp_matches_frozen_int32_and_costs(self):
        context=SequenceContext(self.short,self.t0,4);rng=np.random.default_rng(42)
        for layers in (context.frozen.proposal.proposal_layers,context.frozen.refiner_layers):
            x=rng.integers(0,256,(2053,layers[0][0].shape[0]),dtype=np.uint8)
            a,cost=exact_mlp(x,layers)
            np.testing.assert_array_equal(a,quantized_mlp(x,layers))
            self.assertEqual(cost['dense_mac'],len(x)*sum(w.size for w,s in layers))
            self.assertLessEqual(cost['sparse_mac'],cost['dense_mac'])
            v=x;sparse=0
            for i,(w,s) in enumerate(layers):
                sparse+=np.count_nonzero(v)*w.shape[1]
                raw=v.astype(np.int32)@w.astype(np.int32)
                y=raw.astype(np.float64)*s.astype(np.float64)[None,:]/255.
                if i+1<len(layers):v=np.rint(np.clip(y,0,1)*255).astype(np.uint8)
            self.assertEqual(cost['sparse_mac'],sparse)

    def test_fixed100_all_real_events_and_features_exact(self):
        e,y,c,p,f,m=self.fixture
        a=SequenceContext(e,m['t0_ms'],m['closed_windows']).run(Policy('fixed100',100))
        self.assertGreater(len(f),0)
        for actual,expected in ((a['candidate'],c),(a['probabilities'],p),(a['features'],f)):
            np.testing.assert_array_equal(actual,expected)

    def test_fixed100_direct_stream_and_packetization(self):
        context=SequenceContext(self.short,self.t0,4)
        f=context.frozen;f.reset(self.t0)
        last=0;published=[]
        for dt in (25.,50.,75.,100.,125.,200.,400.):
            end=int(np.searchsorted(self.short['t'],self.t0+dt,side='left'))
            f.push(self.short[last:end],self.t0+dt);last=end
            published+=f.flush_completed_windows()
        a=context.run(Policy('fixed100',100))
        self.assertEqual(len(published),4)
        np.testing.assert_array_equal(a['probabilities'],np.concatenate([w.probabilities for w in published]))
        np.testing.assert_array_equal(a['features'],np.concatenate([w.features for w in published]))

    def test_disabled_quota_and100ms_normalization_equal_baseline(self):
        c=SequenceContext(self.short,self.t0,4);base=c.run(Policy('base',100))
        for p in (Policy('fixed',100,0,'rate100'),Policy('disabled',0,10**9),Policy('disabled',0,10**9,'rate100')):
            a=c.run(p)
            for k in ('candidate','probabilities','features','windows'):np.testing.assert_array_equal(a[k],base[k])
            self.assertEqual(a['cost'],base['cost'])

    def test_no_future_event_leak_into_closed_subwindow(self):
        a=SequenceContext(self.short,self.t0,4)
        changed=self.short.copy();future=changed['t']>=self.t0+225
        changed['x'][future]=(changed['x'][future]+100)%346
        changed['y'][future]=(changed['y'][future]+100)%260
        b=SequenceContext(changed,self.t0,4)
        for scale in ('raw','rate100'):
            first=a.slice(8,9,scale);second=b.slice(8,9,scale)
            for k in ('candidate','probabilities','features'):np.testing.assert_array_equal(first[k],second[k])
        # An already-closed earlier anchor IS used as context.
        self.assertIn((4,8,'raw'),a.proposals)
        self.assertNotIn((8,12,'raw'),a.proposals)

    def test_labels_cannot_enter_inference(self):
        e=np.empty(len(self.short),dtype=list(EVENT_DTYPE.descr)+[('label','i1')])
        for k in EVENT_DTYPE.names:e[k]=self.short[k]
        e['label']=0;a=SequenceContext(e,self.t0,4).run(Policy('q',0,100))
        e['label']=1;b=SequenceContext(e,self.t0,4).run(Policy('q',0,100))
        for k in ('candidate','probabilities','features','windows'):np.testing.assert_array_equal(a[k],b[k])

    def test_physical_history_keeps_two_full_50ms_lag_bins(self):
        # 25 ms subwindow starting at125ms needs the entire0..50ms lag2 bin.
        e=np.zeros(7,EVENT_DTYPE);e['x']=100;e['y']=100;e['p']=1
        e['t']=[1.,15.,30.,60.,110.,126.,145.]
        c=SequenceContext(e,0.,2);base=c.base(5,6)
        all_past=base_features(e,5,0.)
        np.testing.assert_array_equal(base,all_past)
        truncated=base_features(e[2:],3,0.)
        self.assertFalse(np.array_equal(base,truncated))

    def test_extra_anchor_computation_is_not_free(self):
        c=SequenceContext(self.short,self.t0,4)
        base=c.run(Policy('base',100));a=c.run(Policy('short',25))
        self.assertEqual(a['cost']['proposal_history_extra_sparse_mac'],base['cost']['proposal_current_sparse_mac'])
        self.assertEqual(a['cost']['proposal_history_extra_dense_mac'],base['cost']['proposal_current_dense_mac'])
        self.assertEqual(a['cost']['proposal_anchor_reused'],0)
        self.assertEqual(a['cost']['refiner_mac'],int(a['candidate'].sum())*4480)
        self.assertEqual(sum(a['windows'][:,3]-a['windows'][:,2]),len(self.short))
        self.assertLess(a['wait_ms']['mean'],base['wait_ms']['mean'])

    def test_metrics_include_rejected_targets(self):
        c=counts([1,1,1,0],[.9,0.,.2,.5],[1,0,1,1]);m=metrics(c)
        self.assertEqual((c['tp'],c['fp'],c['fn'],c['proposal_fn'],c['refiner_fn']),(1,1,2,1,1))
        self.assertEqual(m['iou'],.25);self.assertEqual(m['recall'],1/3)

    def test_resume_verifies_predictions_without_rewriting(self):
        with test_directory() as tmp:
            out=Path(tmp);d=out/'train_main';d.mkdir();payload=d/'train_000.npz'
            np.savez_compressed(payload,p=np.array([.1,.2]))
            r=dict(role='train_main',sequence='train/train_000.npz',identity_sha256='id',
                   policies={'fixed100':{}},payload=payload.name,payload_sha256=sha(payload))
            path=d/'train_000.json';save(path,r)
            before=(sha(path),path.stat().st_mtime_ns,sha(payload),payload.stat().st_mtime_ns)
            self.assertEqual(verified_record(out,'train_main',r['sequence'],'id',['fixed100']),r)
            save(path,r)
            self.assertEqual(before,(sha(path),path.stat().st_mtime_ns,sha(payload),payload.stat().st_mtime_ns))
            with self.assertRaises(ValueError):verified_record(out,'train_main',r['sequence'],'changed',['fixed100'])
            payload.write_bytes(b'corrupted')
            with self.assertRaises(ValueError):verified_record(out,'train_main',r['sequence'],'id',['fixed100'])

    def test_selection_keeps_tradeoffs_even_below_baseline(self):
        p={}
        for i,policy in enumerate(POLICIES):
            p[policy.name]=dict(metrics=dict(iou=.99 if i==0 else .5+i*.001,precision=.8),
                               high_density=dict(recall=.5 if i==0 else .7-i*.001),cost=dict(mac_sparse_proposal=100+i))
        s=select({'policies':p})
        self.assertEqual(s['choices']['best_iou_raw'],'adaptive2000_raw')
        self.assertEqual(s['choices']['best_iou_rate100'],'adaptive2000_rate100')
        self.assertEqual(s['choices']['best_high_density_recall'],'adaptive500_raw')
        self.assertIn('adaptive500_raw',s['dev_policies'])

    def test_reserved_roles_and_unsealed_dev_blocked(self):
        with self.assertRaises(ValueError):load_sequence('test',{}, {})
        with test_directory() as tmp:
            with self.assertRaises(ValueError):run_stage(tmp,'train_dev',1)


if __name__=='__main__':unittest.main(verbosity=2)
