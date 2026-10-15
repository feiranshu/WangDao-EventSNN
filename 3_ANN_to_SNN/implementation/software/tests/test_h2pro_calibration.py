"""Calibrated IF arithmetic, split isolation, selection, and resumable evidence."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import cast
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'software'))
from wangdao.h2pro_direct_v1.core import forward as original_forward
from wangdao.h2pro_calibration_v1.core import (Encoder,Cost,forward,count_spikes,accumulate,
    probabilities,load_layers)
from wangdao.h2pro_calibration_v1.common import select_records,calibration_manifest,protocol
from wangdao.h2pro_calibration_v1.calibration import sample_features,select_candidates,search_point,refinement_grid
from wangdao.h2pro_calibration_v1.workflow import evaluate_point


class CalibrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers=load_layers()
        cls.x=np.random.default_rng(23).integers(0,256,(97,36),dtype=np.uint8)

    def test_default_reproduces_original_all_stages(self):
        for stage in ('ann','r1','r2'):
            for t in (4,8,16,255):
                old,_=original_forward(self.x,self.layers,stage,t)
                for backend in ('count_reference','spike_add'):
                    actual,_=forward(self.x,self.layers,stage,t,backend=backend)
                    np.testing.assert_array_equal(actual,old)

    def test_full_uint8_domain_and_saturated_if(self):
        a=np.tile(np.arange(256,dtype=np.uint8)[:,None],(1,3))
        w=np.array([[127,-128,0],[3,5,0],[-3,-6,0]],np.int8)
        for theta in (1,128,192,255,320,510,1000000):
            for initial in (0,theta//2,theta-1):
                e=Encoder(theta,initial)
                for t in (4,8,16):
                    reference,rc=accumulate(a,w,t,e,'count_reference')
                    actual,ac=accumulate(a,w,t,e,'spike_add')
                    np.testing.assert_array_equal(actual,reference)
                    self.assertEqual(ac.unit_spikes,rc.unit_spikes)
                    self.assertEqual(ac.executed_synaptic_mac,0)
                    self.assertEqual(ac.executed_synaptic_ac,ac.unit_spikes*3)
                    self.assertLessEqual(int(count_spikes(a,t,e).max()),t)
                    self.assertEqual(ac.encoder_saturated_cells,int((a>theta).sum()))

    def test_zero_input_never_emits_a_bias_spike(self):
        x=np.zeros((7,36),np.uint8)
        for e in (Encoder(128,127),Encoder(510,509)):
            raw,c=forward(x,self.layers,'r2',4,(e,e))
            self.assertFalse(raw.any()); self.assertEqual(c.unit_spikes,0)

    def test_decode_preserves_original_weight_and_scale_arrays(self):
        before=[(w.copy(),s.copy()) for w,s in self.layers]
        e=(Encoder(192,48),Encoder(320,240))
        raw,event=forward(self.x,self.layers,'r2',8,e)
        reference,dense=forward(self.x,self.layers,'r2',8,e,'count_reference')
        np.testing.assert_array_equal(raw,reference)
        self.assertEqual(event.executed_synaptic_mac,len(raw)*2304)
        self.assertEqual(dense.executed_synaptic_mac,len(raw)*4480)
        self.assertEqual(event.executed_synaptic_ac,dense.rate_network_ac)
        # The two changed decoders add an explicit scalar multiply/divide each.
        self.assertEqual(event.scalar_scale_mul,len(raw)*(64+32+4+32+4))
        for saved,actual in zip(before,self.layers):
            for x,y in zip(saved,actual):
                np.testing.assert_array_equal(x,y); self.assertFalse(y.flags.writeable)

    def test_chunk_and_order_independence(self):
        e=(Encoder(160,80),Encoder(448,336))
        raw,c=forward(self.x,self.layers,'r2',16,e)
        total=Cost(); parts=[]
        for block in np.array_split(self.x,9):
            output,cost=forward(block,self.layers,'r2',16,e)
            parts.append(output); total.merge(cost)
        np.testing.assert_array_equal(raw,np.concatenate(parts))
        self.assertEqual(asdict(c),asdict(total))
        reversed_raw,_=forward(self.x[::-1],self.layers,'r2',16,e)
        np.testing.assert_array_equal(raw,reversed_raw[::-1])

    def test_empty_candidate_matrix(self):
        for stage in ('ann','r1','r2'):
            raw,c=forward(np.empty((0,36),np.uint8),self.layers,stage,4)
            self.assertEqual(raw.shape,(0,4)); self.assertFalse(any(asdict(c).values()))

    def test_invalid_numeric_contract(self):
        # cast is deliberately a no-op: exercise rejection of a runtime float.
        for e in (Encoder(0,0),Encoder(255,-1),Encoder(255,255),Encoder(cast(int,255.),127),Encoder(True,0)):
            with self.assertRaises(ValueError):e.validate()
        with self.assertRaises(ValueError):forward(self.x,self.layers,steps=0)
        with self.assertRaises(ValueError):forward(self.x.astype(np.float32),self.layers)

    def test_selection_requires_both_group_errors_not_to_worsen(self):
        def item(name,ordinary,dense,ac):
            return {'key':name,'margin_mse':{'ordinary':ordinary,'high_density':dense},
                    'balanced_margin_mse':(ordinary+dense)/2,'equivalent_ac':ac}
        baseline=item('default',2,2,100)
        good=item('both_better',1.8,1.9,80)
        deceptive=item('dense_worse',.1,2.1,10)
        accurate=item('accurate',.5,.5,130)
        accuracy,low=select_candidates([baseline,good,deceptive,accurate],baseline)
        self.assertEqual(accuracy,accurate); self.assertEqual(low,good)
        _,fallback=select_candidates([baseline,deceptive],baseline)
        self.assertEqual(fallback,baseline)

    def test_sampling_does_not_load_labels_or_teacher_arrays(self):
        class FeaturesOnly:
            def __enter__(self):return self
            def __exit__(self,*args):return False
            def __getitem__(self,key):
                if key!='features':raise AssertionError('Calibration read labels or teacher output')
                return CalibrationTests.x
        manifest={'sequences':[{'sequence':'train/train_a.npz','cache_file':'a.npz','high_density':False},
                               {'sequence':'train/train_b.npz','cache_file':'b.npz','high_density':True}]}
        with patch('wangdao.h2pro_calibration_v1.calibration.np.load',return_value=FeaturesOnly()):
            first=sample_features(Path('.'),manifest)
            second=sample_features(Path('.'),manifest)
        for a,b in zip(first[:3],second[:3]):np.testing.assert_array_equal(a,b)
        self.assertEqual(first[3],second[3])

    def test_sequence_selection_is_deterministic_and_train_main_only(self):
        records=[{'path':f'train/train_{i:03d}.npz'} for i in range(24)]
        rows=[{'sequence':r['path'],'high_density':i>=16,'candidate_events':i+1}
              for i,r in enumerate(records)]
        teacher={'roles':{'train_main':rows,'train_dev':[{'sequence':'NEVER_READ'}]}}
        with patch('wangdao.h2pro_calibration_v1.common.role_records',return_value=records) as roles:
            a=select_records(teacher); b=select_records(teacher)
            self.assertEqual(a,b); self.assertEqual(len(a),12)
            self.assertEqual(len({r['path'] for r in a}),12)
            self.assertTrue(all(c.args==('train_main',) for c in roles.call_args_list))
            with self.assertRaises(ValueError):select_records({'roles':{'train_main':rows[:-1]}})

    def test_dev_cache_cannot_be_used_as_calibration(self):
        with patch('wangdao.h2pro_calibration_v1.common.read_json',return_value={'role':'train_dev'}):
            with self.assertRaises(ValueError):calibration_manifest(Path('.'))

    def test_small_search_contains_default_and_reproducible_choices(self):
        settings={**protocol(),'threshold_grid':[192,255,320],
                  'initial_phase_quarters':[1,2],'coordinate_passes':1,'refinement_passes':1,
                  'threshold_refinement_offsets':[-1,0,1],'membrane_refinement_offsets':[-1,0,1]}
        groups=np.arange(len(self.x))>=48
        sequences=groups.astype(np.int32)
        with patch('wangdao.h2pro_calibration_v1.calibration.protocol',return_value=settings):
            p=search_point(self.x,groups,sequences,self.layers,'r2',4)
            q=search_point(self.x,groups,sequences,self.layers,'r2',4)
        self.assertEqual(p,q); self.assertFalse(p['labels_used']); self.assertFalse(p['weights_updated'])
        for group in ('ordinary','high_density'):
            self.assertLessEqual(p['low_spike']['margin_mse'][group],p['uncalibrated']['margin_mse'][group])

    def test_refinement_preserves_integer_encoder_contract(self):
        for center in (Encoder(1,0),Encoder(),Encoder(510,509)):
            grid=refinement_grid(center,protocol())
            self.assertIn(center,grid); self.assertEqual(len(set(grid)),len(grid))
            for encoder in grid:encoder.validate()

    def test_full_event_metrics_and_sequence_resume(self):
        teacher,_=original_forward(self.x,self.layers)
        n=len(self.x)+2
        arrays={'features':self.x,'candidate':np.r_[np.ones(len(self.x),bool),[False,False]],
            'label':np.ones(n,np.uint8),'teacher':np.r_[probabilities(teacher),np.zeros(2,np.float32)],'logits':teacher}
        row={'sequence':'train/train_002.npz','cache_sha256':'unit-test','high_density':False,
             'proposal_synaptic_mac':100}
        e=(Encoder(192,48),Encoder(320,240))
        with tempfile.TemporaryDirectory(dir=ROOT/'results/runs') as name:
            with patch('wangdao.h2pro_calibration_v1.workflow.load_sequence',return_value=arrays):
                first=evaluate_point(name,{'sequences':[row]},self.layers,'r2',8,'accuracy',e,name,11)
            with patch('wangdao.h2pro_calibration_v1.workflow.load_sequence',side_effect=AssertionError('resume must reuse')):
                second=evaluate_point(name,{'sequences':[row]},self.layers,'r2',8,'accuracy',e,name,11)
            self.assertEqual(first,second); self.assertEqual(first['overall']['proposal_fn'],2)
            self.assertEqual(first['overall']['events'],n); self.assertTrue(first['reference_logits_exact'])
            with self.assertRaisesRegex(ValueError,'receipt mismatch'):
                evaluate_point(name,{'sequences':[row]},self.layers,'r2',8,'accuracy',(Encoder(),Encoder()),name,11)


if __name__=='__main__':unittest.main()
