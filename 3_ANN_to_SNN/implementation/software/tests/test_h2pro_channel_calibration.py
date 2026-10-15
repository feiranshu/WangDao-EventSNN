"""Numerical equivalence, preserved synapses, role isolation, costs and restart."""
import contextlib
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'software'))
from wangdao.h2pro_channel_calibration_v1.core import Channels,defaults,counts,accumulate,forward,cost,load_layers
from wangdao.h2pro_channel_calibration_v1.common import PROTOCOL,parameter_id,parameter_key
from wangdao.h2pro_channel_calibration_v1.search import Runner,TaskRunner,sample,setup,proposals,fit_branch
from wangdao.h2pro_channel_calibration_v1.workflow import select,evaluate_config,calibrate,evaluate,read_json,atomic_json
from wangdao.h2pro_calibration_v1.core import Encoder,forward as previous_forward


class ChannelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers=load_layers();cls.x=np.random.default_rng(728).integers(0,256,(37,36),dtype=np.uint8)
        cls.device,_=setup('cpu')

    def heterogeneous(self):
        return tuple(Channels(tuple(int([128,255,384][j%3]) for j in range(n)),
                              tuple(int([128,255,384][j%3]*(j%4)//4) for j in range(n))) for n in (64,32))

    def sequences(self):
        teacher,_=previous_forward(self.x,self.layers,'ann')
        seqs=[]
        for hd in (False,True):
            y=(np.arange(len(self.x)+2)%3==0).astype(np.uint8);y[-1]=1
            c=np.r_[np.ones(len(self.x),bool),False,False]
            seqs.append({'row':{'sequence':str(hd),'role':'train_main','high_density':hd,
                'target_events':int(y.sum()),'candidate_targets':int(y[c].sum()),'candidate_events':int(c.sum()),
                'events':len(y),'proposal_sparse_mac':7,'proposal_dense_mac':10},
                'features':self.x,'labels':y,'candidate':c,'teacher':teacher,'lo':0,'hi':len(self.x)})
        return seqs

    def test_shared_exact_replay_all_stages_and_phases(self):
        for stage in ('ann','r1','r2'):
            for t in (4,8,16):
                for e in ((Encoder(),Encoder()),(Encoder(253,191),Encoder(386,24)),(Encoder(128,0),Encoder(510,509))):
                    enc=tuple(Channels.shared(w,x.threshold,x.initial_membrane) for w,x in zip((64,32),e))
                    old,oc=previous_forward(self.x,self.layers,stage,t,e,'spike_add')
                    new,nc=forward(self.x,self.layers,stage,t,enc,'spike_add')
                    np.testing.assert_array_equal(new,old)
                    self.assertEqual(nc['equivalent_ac'],oc.rate_network_ac)

    def test_mixed_threshold_original_synapse_add_exact(self):
        before=[a.copy() for pair in self.layers for a in pair]
        for stage in ('r1','r2'):
            for t in (1,4,8,16):
                a,ac=forward(self.x,self.layers,stage,t,self.heterogeneous(),'count_reference')
                b,bc=forward(self.x,self.layers,stage,t,self.heterogeneous(),'spike_add')
                np.testing.assert_array_equal(a,b)
                self.assertEqual(ac['equivalent_ac'],bc['executed_explicit_synaptic_ac'])
        for a,b in zip(before,[a for pair in self.layers for a in pair]):np.testing.assert_array_equal(a,b)

    def test_heterogeneous_decoder_requires_input_channel_theta(self):
        x=np.array([[20,40],[255,255]],np.uint8);w=np.array([[3,-2],[-5,7]],np.int8)
        e=Channels((128,384),(64,192));t=8
        raw,c=accumulate(x,w,t,e,'spike_add')
        expected=np.zeros((2,2),np.int64)
        for row in range(2):
            for i in range(2):
                for j in range(2):expected[row,j]+=int(c[row,i])*e.theta[i]*int(w[i,j])
        np.testing.assert_array_equal(raw,expected)

    def test_zero_and_saturated_currents_binary_one_spike_per_step(self):
        x=np.array([[0,1,127,255],[255,255,255,255]],np.uint8)
        e=Channels((128,128,255,510),(127,0,254,509));w=np.ones((4,3),np.int8)
        for t in (1,8,16):
            _,a=accumulate(x,w,t,e,'spike_add');b=counts(x,t,e)
            np.testing.assert_array_equal(a,b);self.assertTrue((a<=t).all());self.assertEqual(a[0,0],0)

    def test_numpy_torch_exact(self):
        runner=Runner(self.x,self.layers,self.device)
        for e in (defaults(),self.heterogeneous(),(Channels.shared(64,253,191),Channels.shared(32,386,24))):
            for stage in ('ann','r1','r2'):
                a,ac=forward(self.x,self.layers,stage,8,e,'count_reference')
                raw,n1,n2=runner.forward(stage,8,e)
                np.testing.assert_array_equal(a,raw.numpy());self.assertEqual(ac['equivalent_ac'],int(n1.sum()*32+n2.sum()*4))
        self.assertTrue(all(not w.requires_grad for w in runner.weights))

    @unittest.skipUnless(torch.cuda.is_available(),'GPU unavailable')
    def test_cuda_exact_with_explicit_original_weight_spikes(self):
        d,_=setup('cuda');runner=Runner(self.x,self.layers,d)
        for e in (defaults(),self.heterogeneous(),(Channels.shared(64,253,191),Channels.shared(32,386,24))):
            for stage,t in (('ann',8),('r1',16),('r2',8)):
                actual,n1,n2=runner.forward(stage,t,e)
                expected,c=forward(self.x,self.layers,stage,t,e,'spike_add')
                np.testing.assert_array_equal(actual.cpu().numpy(),expected)
                self.assertEqual(c['equivalent_ac'],int(n1.sum()*32+n2.sum()*4))

    def test_decoder_multiplications_not_hidden_in_ac(self):
        e=self.heterogeneous();c=cost(10,'r2',8,e,[100,20])
        self.assertEqual(c['equivalent_ac'],3280)
        self.assertEqual(c['refiner_equivalent_mac'],23040)
        self.assertEqual(c['extra_threshold_decode_mul'],10*3*(32+4))
        self.assertEqual(c['extra_threshold_bucket_adds'],10*2*(32+4))
        self.assertEqual(c['extra_threshold_decode_divisions'],10*(32+4))
        self.assertEqual(c['encoder_parameter_bytes_expanded_int32'],768)
        self.assertEqual(c['executed_reference_refiner_mac'],44800)

    def test_membrane_only_adds_no_threshold_decode_operations(self):
        e=tuple(Channels((255,)*w,tuple(i%255 for i in range(w))) for w in (64,32))
        c=cost(10,'r2',8,e,[100,20])
        self.assertEqual(c['extra_threshold_decode_mul'],0)
        self.assertEqual(c['extra_threshold_bucket_adds'],0)

    def test_empty_and_chunk_order_independence(self):
        e=self.heterogeneous()
        a,c=forward(self.x[:0],self.layers,'r2',8,e,'spike_add')
        self.assertEqual(a.shape,(0,4));self.assertEqual(c['equivalent_ac'],0)
        full,c=forward(self.x,self.layers,'r2',8,e,'count_reference')
        chunks=[forward(x,self.layers,'r2',8,e,'spike_add') for x in (self.x[:11],self.x[11:])]
        np.testing.assert_array_equal(full,np.concatenate([x[0] for x in chunks]))
        self.assertEqual(c['equivalent_ac'],sum(x[1]['equivalent_ac'] for x in chunks))

    def test_invalid_parameters_and_inputs(self):
        for e in (Channels((0,),(0,)),Channels((255,),(-1,)),Channels((255,),(255,)),Channels((True,),(0,))):
            with self.assertRaises(ValueError):e.validate(1)
        with self.assertRaises(ValueError):counts(np.zeros(4,np.uint8),8,Channels.shared(4))
        with self.assertRaises(ValueError):forward(self.x.astype(float),self.layers)

    def test_sample_is_main_only_stratified_and_reproducible(self):
        seqs=self.sequences()
        with patch.dict(PROTOCOL['sample'],{'max_per_class_per_main_sequence':3}):
            a=sample(seqs);b=sample(seqs)
        for k in ('features','labels','hd','weights','teacher'):np.testing.assert_array_equal(a[k],b[k])
        for h in (False,True):
            for y in (0,1):
                actual=a['weights'][(a['hd']==h)&(a['labels']==y)].sum()
                expected=sum(int((s['labels'][s['candidate']]==y).sum()) for s in seqs if s['row']['high_density']==h)
                self.assertAlmostEqual(actual,expected)
        seqs[0]['row']['role']='train_dev'
        with self.assertRaises(ValueError):sample(seqs)

    def test_task_proposal_misses_remain_and_groups_cover_channels(self):
        data=sample(self.sequences());r=TaskRunner(data,self.layers,self.device)
        score=r.assess('r2',8,defaults());self.assertTrue(np.isfinite(list(v for k,v in score.items() if k!='parameter_id')).all())
        groups,_=r.grouping()
        for w,gs in zip((64,32),groups):self.assertEqual(sorted(sum(gs,[])),list(range(w)))
        self.assertGreater(data['proposal_fn']['overall'],0)

    def test_m0_search_never_changes_threshold(self):
        e=Channels.shared(64,253,191)
        for candidate in proposals(e,[0,4,8],'membrane',0):self.assertEqual(candidate.theta,e.theta)

    def test_full_population_score_keeps_rejected_targets_and_cost(self):
        seqs=self.sequences();result,raw=evaluate_config(seqs,self.layers,self.device,'r2',8,self.heterogeneous(),11,True,True)
        self.assertEqual(result['overall']['events'],2*(len(self.x)+2))
        self.assertEqual(result['overall']['proposal_fn'],sum(int(s['labels'][~s['candidate']].sum()) for s in seqs))
        self.assertEqual(result['cost']['mac_sparse_proposal'],2*len(self.x)*2304+14)
        self.assertEqual(len(raw),len(self.x)*2);self.assertEqual(result['verified_explicit_rows'],len(self.x)*2)

    def test_main_selection_does_not_relax_recall(self):
        def c(pid,iou,tp,hd):
            return {'parameter_id':pid,'overall':{'iou':iou,'tp':tp,'recall':tp/10},
                    'high_density':{'tp':hd,'recall':hd/10},'ordinary':{},'cost':{'equivalent_ac':100},'encoders':[]}
        ann=c('ann',.9,9,8);a=c('high_iou',.89,8,7);b=c('recall',.8,9,8)
        self.assertEqual(select([a,b],ann,'iou','train_main')['parameter_id'],'high_iou')
        self.assertEqual(select([a,b],ann,'recall_preserving','train_main')['parameter_id'],'recall')
        self.assertEqual(select([a],ann,'recall_preserving','train_main')['status'],'infeasible')
        with self.assertRaises(ValueError):select([a],ann,'iou','train_dev')

    def test_completed_fit_reuses_without_forward(self):
        data=sample(self.sequences());r=TaskRunner(data,self.layers,self.device);grouping,_=r.grouping()
        settings={'coordinate_passes':1,'threshold_grid':[255],'phase_quarters':[2],
                  'local_threshold_offsets':[0],'local_membrane_offsets':[0]}
        with tempfile.TemporaryDirectory(dir=ROOT/'results/runs') as name,patch.dict(PROTOCOL,settings),contextlib.redirect_stdout(io.StringIO()):
            from pathlib import Path
            a=fit_branch(r,'r1',8,'shared',defaults(),grouping,Path(name))
            with patch.object(r,'assess',side_effect=AssertionError('completed calibration must be reused')):
                b=fit_branch(r,'r1',8,'shared',defaults(),grouping,Path(name))
            self.assertEqual(a,b)

    def test_sealed_main_reuse_does_not_load_dev_or_main(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results/runs') as name:
            d=Path(name);sealed={'selection_role':'train_main','main_evidence_sha256':{},'status':'complete'}
            atomic_json(d/'selection.json',sealed)
            with patch('wangdao.h2pro_channel_calibration_v1.workflow.load_role',side_effect=AssertionError('no loading on resume')):
                self.assertEqual(calibrate('.',d,self.device),sealed)

    def test_dev_cannot_load_without_main_seal(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results/runs') as name:
            with patch('wangdao.h2pro_channel_calibration_v1.workflow.load_role',side_effect=AssertionError('Dev must not load')):
                with self.assertRaises(FileNotFoundError):evaluate('.',Path(name),self.device)

    def test_interrupted_coordinate_search_resumes_same_result(self):
        data=sample(self.sequences());r=TaskRunner(data,self.layers,self.device);grouping,_=r.grouping()
        settings={'coordinate_passes':1,'threshold_grid':[255],'phase_quarters':[],
                  'local_threshold_offsets':[0],'local_membrane_offsets':[0]}
        with tempfile.TemporaryDirectory(dir=ROOT/'results/runs') as name,patch.dict(PROTOCOL,settings),contextlib.redirect_stdout(io.StringIO()):
            a=Path(name)/'uninterrupted';b=Path(name)/'resumed';a.mkdir();b.mkdir()
            expected=fit_branch(r,'r1',8,'group_m0',defaults(),grouping,a)
            real=r.assess;calls=[0]
            def interrupted(*args):
                calls[0]+=1
                if calls[0]==6:raise KeyboardInterrupt('synthetic interruption')
                return real(*args)
            with patch.object(r,'assess',side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):fit_branch(r,'r1',8,'group_m0',defaults(),grouping,b)
            self.assertEqual(read_json(b/'group_m0_cursor.json')['next_step'],2)
            actual=fit_branch(r,'r1',8,'group_m0',defaults(),grouping,b)
            self.assertEqual(actual,expected)


if __name__=='__main__':unittest.main()
