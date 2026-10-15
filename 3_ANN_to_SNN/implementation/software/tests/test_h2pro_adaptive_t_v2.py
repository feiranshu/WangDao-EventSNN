import copy,json,tempfile,unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np
from wangdao.h2pro_adaptive_t_v2.core import Policy,CONTROL,choose,should_exit,incremental,ledger
from wangdao.h2pro_adaptive_t_v2.common import SOURCE,REFERENCE,DEFAULT_CACHE,DESIGN,NEGATIVE,POSITIVE,sha256,atomic_npz,sealed_json
from wangdao.h2pro_adaptive_t_v2.scoring import score,subset,rank,policy_id
from wangdao.h2pro_adaptive_t_v2.search import histogram,all_statistics,grid_codes,policy_from_code,check_direct,select_main
from wangdao.h2pro_adaptive_t_v2 import experiment
from wangdao.h2pro_adaptive_t_v1.core import Policy as OldPolicy,incremental as old_incremental,trace_reference,ledger as old_ledger
from wangdao.h2pro_calibration_v1.core import forward,Encoder,probabilities
from wangdao.h2pro_r1_train_v1.evaluation import load_export


def layers():
    rng=np.random.default_rng(19)
    return tuple((rng.integers(-7,8,(a,b),dtype=np.int8),rng.uniform(.003,.04,b).astype(np.float32))
                 for a,b in ((36,64),(64,32),(32,4)))


def synthetic():
    rng=np.random.default_rng(42);p=rng.uniform(0,1,(3,200)).astype(np.float32)
    edges=np.array([0,.01,.3,.35,.4,.99,1],np.float32)
    boundary=np.concatenate([edges,np.nextafter(edges,np.float32(0)),np.nextafter(edges,np.float32(1))])
    p[:2,:len(boundary)]=boundary
    seqs=[];y=np.tile([0,1,1,0,0],40).astype(np.uint8)
    for i in range(8):
        row=dict(sequence=f'train/train_{i:03d}.npz',role='train_main',events=30,target_events=12,candidate_events=25,
                 candidate_targets=10,high_density=i>=4,proposal_sparse_mac=10,proposal_dense_mac=20)
        seqs.append(dict(row=row,lo=i*25,hi=(i+1)*25))
    return dict(role='train_main',sequences=seqs,p=p,labels=y,
                spikes=np.cumsum(rng.integers(0,150,(3,200),dtype=np.int64),axis=0))


@contextmanager
def workspace_tmp():
    root=(SOURCE.parent/'h2pro_adaptive_t_v2_checks_20261015/tmp').resolve()
    assert SOURCE.parent.resolve() in root.parents
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as d:
        path=Path(d).resolve();assert path.parent==root
        yield path


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.layers=layers();self.x=np.random.default_rng(51).integers(0,256,(101,36),dtype=np.uint8)

    def test_shared_control_exact_old_execution_and_entire_ledger(self):
        for neg,pos in ((.01,.99),(None,None),(.1,None),(None,.9)):
            new=incremental(self.x,self.layers,Policy(neg,pos,neg,pos),batch_size=29)
            old=old_incremental(self.x,self.layers,OldPolicy(neg,pos),batch_size=29)
            for a,b in zip(new[:3],old[:3]):np.testing.assert_array_equal(a,b)
            self.assertEqual(new[3],old[3])
            self.assertEqual(ledger(new[2],new[1],Policy(neg,pos,neg,pos)),old_ledger(old[2],old[1],OldPolicy(neg,pos)))

    def test_independent_prefixes_actual_spike_add_and_cost(self):
        raw,spikes=trace_reference(self.x,self.layers)
        pp=np.stack([probabilities(r) for r in raw])
        for policy in (Policy(.001,.999,.1,.6),Policy(None,None,.3,.4),Policy(.3,.4,None,None)):
            y,s,ix,measured=incremental(self.x,self.layers,policy,17)
            np.testing.assert_array_equal(ix,choose(pp,policy))
            np.testing.assert_array_equal(y,raw[ix,np.arange(len(ix))])
            np.testing.assert_array_equal(s,spikes[ix,np.arange(len(ix))])
            expected=ledger(ix,s,policy)
            for k,v in measured.items():self.assertEqual(v,expected[k])

    def test_gate_step_specific_boundaries(self):
        p=np.array([.01,np.nextafter(np.float32(.01),np.float32(1)),.1,.9,.99],np.float32)
        policy=Policy(.01,.99,.1,.9)
        self.assertEqual(should_exit(p,4,policy).tolist(),[True,False,False,False,True])
        self.assertEqual(should_exit(p,8,policy).tolist(),[True,True,True,True,True])
        self.assertTrue(should_exit(p,16,policy).all())

    def test_future_outputs_cannot_change_earlier_exits(self):
        p=np.array([[.001,.4,.4],[.5,.95,.4],[.8,.2,.4]],np.float32);policy=Policy(.01,.99,.1,.9)
        self.assertEqual(choose(p,policy).tolist(),[0,1,2])
        p[1:,0]=1-p[1:,0];p[2,:]=1-p[2,:]
        self.assertEqual(choose(p,policy).tolist(),[0,1,2])

    def test_chunk_order_invariance(self):
        policy=Policy(.03,.95,.1,.9);a=incremental(self.x,self.layers,policy,101)
        b=incremental(self.x[::-1].copy(),self.layers,policy,13)
        for left,right in zip(a[:3],b[:3]):np.testing.assert_array_equal(left,right[::-1])
        self.assertEqual(a[3],b[3])

    def test_no_exit_reproduces_original_t16(self):
        y,s,ix,c=incremental(self.x,self.layers,Policy())
        expected,cost=forward(self.x,self.layers,'r1',16,(Encoder(255,191),Encoder()),'spike_add')
        np.testing.assert_array_equal(y,expected);self.assertTrue((ix==2).all())
        self.assertEqual(int(s.sum())*32,cost.executed_synaptic_ac)
        self.assertEqual(c['input_projection_mac'],len(self.x)*2304)
        self.assertEqual(c['readout_mac'],len(self.x)*3*128)

    def test_source159_export_and_inputs_unchanged(self):
        path=SOURCE/'epochs/epoch_159_int8.npz';before=sha256(path);ls=load_export(path)
        saved=[(w.copy(),s.copy()) for w,s in ls];original=self.x.copy()
        raw,spikes=trace_reference(self.x,ls);policy=Policy(.001,.99,.03,.95)
        y,n,ix,_=incremental(self.x,ls,policy,31)
        np.testing.assert_array_equal(y,raw[ix,np.arange(len(ix))]);np.testing.assert_array_equal(n,spikes[ix,np.arange(len(ix))])
        for (w,s),(bw,bs) in zip(ls,saved):np.testing.assert_array_equal(w,bw);np.testing.assert_array_equal(s,bs)
        np.testing.assert_array_equal(self.x,original);self.assertEqual(sha256(path),before)

    def test_hand_cost_includes_both_heads_and_step_threshold_counts(self):
        ix=np.array([0,1,2,2],np.uint8);s=np.array([10,20,40,50],np.int64)
        c=ledger(ix,s,Policy(.01,None,None,.99))
        self.assertEqual(c['readout_mac'],9*128);self.assertEqual(c['synaptic_ac'],120*32)
        self.assertEqual(c['if_adds'],44*64);self.assertEqual(c['gate_threshold_comparisons'],7)

    def test_empty_and_invalid_contract(self):
        y,s,ix,c=incremental(self.x[:0],self.layers,CONTROL)
        self.assertEqual(y.shape,(0,4));self.assertEqual(sum(c.values()),0)
        for policy in (Policy(.35,None),Policy(None,.35),Policy(float('nan'),.99),Policy(.01,float('inf'))):
            with self.assertRaises(ValueError):policy.validate()
        with self.assertRaises(ValueError):incremental(self.x,self.layers,CONTROL,0)
        with self.assertRaises(ValueError):choose(np.full((3,2),np.nan),CONTROL)


class CalibrationTests(unittest.TestCase):
    def test_exact_histogram_exhaustive_grid_and_threshold_equalities(self):
        b=synthetic();neg=[None,.01,.3];pos=[None,.4,.99]
        table=histogram(b,neg,pos);stats=all_statistics(table,neg,pos);codes=grid_codes(neg,pos)
        self.assertEqual(len(codes),81)
        for i,code in enumerate(codes):check_direct(stats[i],table['meta'],score(b,policy_from_code(code,neg,pos)))

    def test_full_event_denominator_and_proposal_misses(self):
        b=synthetic();r=score(b,CONTROL,details=True)
        self.assertEqual(r['overall']['events'],240);self.assertEqual(r['overall']['target_events'],96)
        self.assertEqual(r['overall']['proposal_fn'],16)
        self.assertEqual(r['overall']['fn'],96-r['overall']['tp'])
        self.assertEqual(r['changes_vs_shared']['changed_predictions'],0)

    def test_same_grid_includes_shared_control_and_disabled_exits(self):
        codes=grid_codes();self.assertEqual(len(codes),10000)
        pp=[policy_from_code(c) for c in codes]
        self.assertIn(CONTROL,pp);self.assertIn(Policy(),pp)
        self.assertEqual(len({policy_id(p) for p in pp}),10000)

    def test_dev_cannot_enter_calibration(self):
        b=synthetic();b['role']='train_dev'
        with self.assertRaises(ValueError):histogram(b)
        with self.assertRaises(ValueError):select_main(b)

    def test_selection_preserves_control_and_uses_same_fit_validation_objectives(self):
        b=synthetic();result,arrays=select_main(b)
        self.assertEqual(arrays['statistics'].shape,(10000,2,6))
        self.assertIn(result['control_grid_index'],result['shortlist'])
        self.assertFalse(set(result['split']['fit'])&set(result['split']['validation']))
        baseline=score(subset(b,result['split']['validation']),fixed_step=16)
        for name,coefficient in DESIGN['objectives'].items():
            expected=max(result['shortlist'],key=lambda i:(rank(result['validation_candidates'][str(i)],coefficient,baseline),-i))
            self.assertEqual(result['selected'][name]['grid_index'],expected)
        self.assertFalse(result['dev_used_for_selection']);self.assertFalse(result['weights_updated'])

    def test_extra_teacher_fields_are_unused(self):
        b=synthetic();before=histogram(b);b['teacher']=np.full((200,4),np.nan)
        after=histogram(b);np.testing.assert_array_equal(before['hist'],after['hist'])
        self.assertEqual(score(b,CONTROL),score({k:v for k,v in b.items() if k!='teacher'},CONTROL))

    def test_histogram_coverage_mismatch_is_rejected(self):
        b=synthetic();b['sequences'][0]['row']['candidate_targets']+=1
        with self.assertRaises(ValueError):histogram(b)


class RecoveryTests(unittest.TestCase):
    def test_main_seal_binds_selection_and_full_search(self):
        with workspace_tmp() as run:
            sealed_json(run/'identity.json',{'test':1});selection=dict(role='train_main',dev_used_for_selection=False)
            arrays=dict(statistics=np.zeros((2,2,6),np.int64))
            with self.assertRaises(ValueError):experiment.require_seal(run)
            experiment.seal_main(run,selection,arrays,[])
            self.assertEqual(experiment.require_seal(run),selection)
            before=sha256(run/'fit_search.npz');experiment.seal_main(run,selection,arrays,[])
            self.assertEqual(sha256(run/'fit_search.npz'),before)
            (run/'fit_search.npz').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):experiment.require_seal(run)

    def test_dev_action_rejects_unsealed_main_before_any_prefix_load(self):
        with workspace_tmp() as base:
            with patch.object(experiment,'identity',return_value=({'test':1},None)), \
                 patch.object(experiment,'manifest',return_value={'roles':{}}), \
                 patch.object(experiment,'bundle',side_effect=AssertionError('Must not load Dev')):
                with self.assertRaisesRegex(ValueError,'Seal Main'):
                    experiment.run('dev',base/'run',DEFAULT_CACHE)

    def test_original_output_directories_are_protected(self):
        for path in (SOURCE,REFERENCE,DEFAULT_CACHE,REFERENCE/'child',REFERENCE.parent):
            with self.assertRaises(ValueError):
                experiment.run(directory=path)

    def test_executed_sequence_resume_and_corruption(self):
        ls=layers();x=np.random.default_rng(3).integers(0,256,(37,36),dtype=np.uint8)
        raw,spikes=trace_reference(x,ls);row=dict(sequence='train/fixture.npz')
        with workspace_tmp() as run:
            with patch.object(experiment,'read_prefix',return_value=({'features':x},raw,spikes,{})):
                first=experiment.verify_policy(run,Path('.'),[row],ls,CONTROL,{},11)
                with patch.object(experiment,'incremental',side_effect=AssertionError('Must reuse')):
                    second=experiment.verify_policy(run,Path('.'),[row],ls,CONTROL,{},11)
                self.assertEqual(first,second)
                (run/'executed_dev'/policy_id(CONTROL)/'fixture.npz').write_bytes(b'corrupt')
                with self.assertRaises(ValueError):experiment.verify_policy(run,Path('.'),[row],ls,CONTROL,{},11)

    def test_prefix_reader_never_regenerates_missing_old_evidence(self):
        with workspace_tmp() as ref:
            with patch.object(experiment,'REFERENCE',ref):
                with self.assertRaisesRegex(ValueError,'complete original'):
                    experiment.read_prefix(Path('.'),dict(role='train_main',sequence='train/a.npz'))
            self.assertEqual(list(ref.iterdir()),[])

    def test_completed_run_reuse_never_loads_prefixes(self):
        with workspace_tmp() as base:
            run=base/'run';run.mkdir();ident={'test':1};sealed_json(run/'identity.json',ident)
            experiment.seal_main(run,dict(role='train_main',dev_used_for_selection=False),dict(a=np.ones(2)),[])
            sealed_json(run/'comparison.json',dict(artifact_sha256={}))
            with patch.object(experiment,'identity',return_value=(ident,None)), \
                 patch.object(experiment,'manifest',side_effect=AssertionError('No new data loading')), \
                 patch.object(experiment,'bundle',side_effect=AssertionError('No new inference')):
                result=experiment.run(directory=run)
            self.assertTrue(result['reused'])


if __name__=='__main__':unittest.main()
