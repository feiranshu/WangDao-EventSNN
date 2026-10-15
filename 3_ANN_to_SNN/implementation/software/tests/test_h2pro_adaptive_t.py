import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from wangdao.h2pro_adaptive_t_v1.core import (
    Policy,choose,should_exit,trace_reference,incremental,ledger,STEPS)
from wangdao.h2pro_adaptive_t_v1.common import split_main,sealed_json,atomic_npz,DESIGN
from wangdao.h2pro_adaptive_t_v1.selection import select_main,score,subset,rank,policy_id
from wangdao.h2pro_adaptive_t_v1.experiment import require_seal,seal_main,get_trace
from wangdao.h2pro_calibration_v1.core import forward,Encoder,probabilities
from wangdao.h2pro_r1_train_v1.evaluation import load_export
from wangdao.h2pro_adaptive_t_v1.common import SOURCE,sha256


def layers(seed=19):
    rng=np.random.default_rng(seed)
    return tuple((rng.integers(-7,8,(a,b),dtype=np.int8),
                  rng.uniform(.003,.04,b).astype(np.float32)) for a,b in ((36,64),(64,32),(32,4)))


def synthetic_bundle():
    rng=np.random.default_rng(42);seqs=[];offset=0
    for i in range(8):
        row=dict(sequence=f'train/{i}.npz',role='train_main',events=7,target_events=3,
                 candidate_events=5,candidate_targets=2,high_density=i>=4,
                 proposal_sparse_mac=10,proposal_dense_mac=12)
        seqs.append(dict(row=row,lo=offset,hi=offset+5));offset+=5
    return dict(role='train_main',sequences=seqs,labels=np.tile([0,1,0,1,0],8),
                p=rng.uniform(0,1,(3,40)).astype(np.float32),
                spikes=np.stack([np.full(40,t*20,dtype=np.int64) for t in STEPS]))


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.layers=layers();self.x=np.random.default_rng(4).integers(0,256,(73,36),dtype=np.uint8)

    def test_every_prefix_exact_original_binary_kernel(self):
        raw,spikes=trace_reference(self.x,self.layers)
        for i,t in enumerate(STEPS):
            expected,c=forward(self.x,self.layers,'r1',t,(Encoder(255,191),Encoder()),'spike_add')
            np.testing.assert_array_equal(raw[i],expected)
            self.assertEqual(int(spikes[i].sum())*32,c.executed_synaptic_ac)

    def test_incremental_preserves_states_and_matches_prefixes(self):
        raw,spikes=trace_reference(self.x,self.layers)
        for policy in (Policy(.1,.8),Policy(.3,.6),Policy(.1,.8,'stable8'),Policy()):
            y,s,ix,actual=incremental(self.x,self.layers,policy,batch_size=23)
            expected=choose(np.stack([probabilities(r) for r in raw]),policy)
            np.testing.assert_array_equal(ix,expected)
            np.testing.assert_array_equal(y,raw[ix,np.arange(len(ix))])
            np.testing.assert_array_equal(s,spikes[ix,np.arange(len(ix))])
            counts=ledger(ix,s,policy)
            for k,v in actual.items():self.assertEqual(v,counts[k],k)

    def test_no_exit_exact_t16_and_three_readouts(self):
        y,s,ix,c=incremental(self.x,self.layers,Policy())
        expected,_=forward(self.x,self.layers,'r1',16,(Encoder(255,191),Encoder()),'spike_add')
        np.testing.assert_array_equal(y,expected);self.assertTrue(np.all(ix==2))
        self.assertEqual(c['readout_mac'],len(self.x)*3*128)
        self.assertEqual(c['input_projection_mac'],len(self.x)*2304)

    def test_all_early_and_stable8(self):
        ls=tuple((np.ones((a,b),np.int8) if i<2 else np.zeros((a,b),np.int8),np.full(b,.01,np.float32))
                 for i,(a,b) in enumerate(((36,64),(64,32),(32,4))))
        x=np.full((7,36),255,np.uint8)
        for mode,t in (('confidence',4),('stable8',8)):
            y,s,ix,c=incremental(x,ls,Policy(None,.4,mode))
            np.testing.assert_array_equal(y,forward(x,ls,'r1',t,(Encoder(255,191),Encoder()),'spike_add')[0])
            self.assertTrue(np.all(ix==STEPS.index(t)));self.assertTrue(np.all(s>0))
            self.assertEqual(c['active_step_rows'],7*t)

    def test_future_predictions_cannot_change_earlier_exits(self):
        p=np.array([[.001,.5,.5],[.9,.99,.5],[.9,.01,.9]],np.float32)
        policy=Policy(.01,.95);first=choose(p,policy)
        self.assertEqual(first.tolist(),[0,1,2])
        p[1:,0]=.5;p[2,:]=1-p[2,:]
        np.testing.assert_array_equal(choose(p,policy),first)

    def test_class_agreement_for_stable8(self):
        p=np.array([[.8,.1,.1],[.001,.99,.001],[.6,.2,.7]],np.float32)
        self.assertEqual(choose(p,Policy(.01,.95,'stable8')).tolist(),[2,2,1])

    def test_gate_boundaries_and_forced_final(self):
        p=np.array([.1,.2,.8,.9],np.float32)
        self.assertEqual(should_exit(p,4,Policy(.1,.8)).tolist(),[True,False,True,True])
        self.assertTrue(should_exit(p,16,Policy()).all())

    def test_independent_hand_cost(self):
        c=ledger(np.array([0,1,2],np.uint8),np.array([5,9,16]),Policy(.1,.9))
        self.assertEqual(c['input_projection_mac'],3*2304)
        self.assertEqual(c['readout_mac'],(1+2+3)*128)
        self.assertEqual(c['synaptic_ac'],30*32)
        self.assertEqual(c['if_adds'],(4+8+16)*64)
        self.assertEqual(c['gate_threshold_comparisons'],(3+2)*2)
        self.assertEqual(c['decode_scale_mul'],3*64+6*36)
        self.assertEqual(ledger(np.array([2]*3),np.array([5,9,16]),fixed=True)['readout_mac'],3*128)

    def test_batch_partition_invariant(self):
        p=Policy(.05,.7)
        a=incremental(self.x,self.layers,p,7);b=incremental(self.x,self.layers,p,73)
        for x,y in zip(a[:3],b[:3]):np.testing.assert_array_equal(x,y)
        self.assertEqual(a[3],b[3])

    def test_original_parameters_and_features_unchanged(self):
        x=self.x.copy();ls=copy.deepcopy(self.layers)
        incremental(self.x,self.layers,Policy(.1,.8));trace_reference(self.x,self.layers)
        np.testing.assert_array_equal(x,self.x)
        for a,b in zip(ls,self.layers):
            for x,y in zip(a,b):np.testing.assert_array_equal(x,y)

    def test_empty_and_invalid_inputs(self):
        y,s,ix,c=incremental(np.zeros((0,36),np.uint8),self.layers,Policy())
        self.assertEqual(y.shape,(0,4));self.assertEqual(c['synaptic_ac'],0)
        for p in (Policy(.35,.8),Policy(.1,.35),Policy(float('nan'),.8),Policy(.1,.8,'bad')):
            with self.assertRaises(ValueError):p.validate()
        with self.assertRaises(ValueError):incremental(self.x.astype(np.float32),self.layers,Policy())
        with self.assertRaises(ValueError):trace_reference(self.x,self.layers,batch_size=0)
        with self.assertRaises(ValueError):choose(np.full((3,2),np.nan),Policy())

    def test_real_epoch159_export_all_prefixes_without_training(self):
        path=SOURCE/'epochs/epoch_159_int8.npz';before=sha256(path);ls=load_export(path)
        raw,spikes=trace_reference(self.x,ls)
        for i,t in enumerate(STEPS):
            expected,c=forward(self.x,ls,'r1',t,(Encoder(255,191),Encoder()),'spike_add')
            np.testing.assert_array_equal(raw[i],expected)
            self.assertEqual(int(spikes[i].sum())*32,c.executed_synaptic_ac)
        self.assertEqual(before,sha256(path))


class SelectionAndRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_root=(SOURCE.parent/'h2pro_adaptive_t_v1_checks_20261015/tmp').resolve()
        self.temp_root.mkdir(parents=True,exist_ok=True)
        self.assertTrue(SOURCE.parent.resolve() in self.temp_root.parents)

    def test_full_event_denominator_and_rejects(self):
        b=synthetic_bundle();v=score(b,fixed_step=16,details=True)
        self.assertEqual(v['overall']['events'],56)
        self.assertEqual(v['overall']['target_events'],24)
        self.assertEqual(v['overall']['proposal_fn'],8)
        self.assertEqual(v['overall']['fn'],24-v['overall']['tp'])
        self.assertEqual(v['cost']['inference_equivalent_mac_sparse_proposal'],80+40*2432)
        self.assertEqual(v['cost']['synaptic_ac'],40*320*32)

    def test_split_deterministic_disjoint_and_stratified(self):
        b=synthetic_bundle();rs=[s['row'] for s in b['sequences']]
        a=split_main(rs);self.assertEqual(a,split_main(list(reversed(rs))))
        self.assertFalse(set(a['fit'])&set(a['validation']))
        self.assertEqual(len(a['fit'])+len(a['validation']),8)
        self.assertEqual(subset(b,a['validation'])['p'].shape[1],10)

    def test_dev_selection_rejected(self):
        b=synthetic_bundle();b['role']='train_dev'
        with self.assertRaises(ValueError):select_main(b)
        with self.assertRaises(ValueError):split_main([dict(role='train_dev')])

    def test_objective_identical_in_fit_and_validation(self):
        b=synthetic_bundle()
        with patch.dict(DESIGN,negative_grid=[None,.1],positive_grid=[None,.9]):
            s=select_main(b)
        validation=subset(b,s['split']['validation']);baseline=score(validation,fixed_step=16)
        for name,coefficient in DESIGN['objectives'].items():
            key=max(s['validation_candidates'],key=lambda k:rank(s['validation_candidates'][k],coefficient,baseline))
            self.assertEqual(s['selected'][name]['policy_id'],key)
        self.assertFalse(s['dev_used_for_selection']);self.assertFalse(s['weights_updated'])

    def test_selection_immune_to_unavailable_teacher(self):
        b=synthetic_bundle()
        with patch.dict(DESIGN,negative_grid=[None,.1],positive_grid=[None,.9]):
            s1=select_main(b);b['teacher']=np.full((40,4),np.nan);s2=select_main(b)
        self.assertEqual(s1,s2)

    def test_seal_required_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            self.assertEqual(Path(d).resolve().parent,self.temp_root)
            run=Path(d)
            with self.assertRaises(ValueError):require_seal(run)
            sealed_json(run/'identity.json',{'test':1})
            selection=dict(role='train_main',dev_used_for_selection=False)
            seal_main(run,selection,[])
            self.assertEqual(require_seal(run),selection)
            (run/'selection.json').write_text('{}')
            with self.assertRaises(ValueError):require_seal(run)

    def test_completed_json_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            self.assertEqual(Path(d).resolve().parent,self.temp_root)
            p=Path(d)/'f.json';sealed_json(p,{'a':1});before=p.read_bytes()
            sealed_json(p,{'a':1});self.assertEqual(p.read_bytes(),before)
            with self.assertRaises(ValueError):sealed_json(p,{'a':2})

    def test_trace_reuse_avoids_inference_and_checks_hash(self):
        with tempfile.TemporaryDirectory(dir=self.temp_root) as d:
            self.assertEqual(Path(d).resolve().parent,self.temp_root)
            root=Path(d);cache=root/'cache';cache.mkdir();run=root/'run'
            x=np.zeros((3,36),np.uint8);y=np.array([0,1,0,0],np.uint8);cand=np.array([1,1,0,1],bool)
            atomic_npz(cache/'a.npz',features=x,labels=y,candidate=cand,teacher=np.zeros((3,4),np.float32))
            row=dict(sequence='train/a.npz',role='train_main',events=4,target_events=1,candidate_events=3,
                     candidate_targets=1,cache_file='a.npz',cache_sha256=sha256(cache/'a.npz'))
            first=get_trace(run,cache,row,layers(),'cpu',8)
            path=run/'traces/train_main/a.npz';before=path.read_bytes()
            with patch('wangdao.h2pro_adaptive_t_v1.experiment.trace_reference',side_effect=AssertionError('must reuse')):
                second=get_trace(run,cache,row,layers(),'cpu',8)
            np.testing.assert_array_equal(first[1],second[1]);self.assertEqual(path.read_bytes(),before)
            path.write_bytes(b'corrupt')
            with self.assertRaises(ValueError):get_trace(run,cache,row,layers(),'cpu',8)


if __name__=='__main__':unittest.main()
