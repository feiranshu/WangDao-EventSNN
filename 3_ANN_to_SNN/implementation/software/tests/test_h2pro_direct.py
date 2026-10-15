"""Frozen arithmetic, actual spike accumulation, preservation and resume contracts."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'software'))
from wangdao.h2.numpy_model import quantized_mlp, _refiner_probability
from wangdao.h2pro_direct_v1.core import (Cost, forward, load_layers, probabilities,
    spike_accumulate, spike_counts, validate_steps)
from wangdao.h2pro_direct_v1.common import base_identity, lock_directory, output_path
from wangdao.h2pro_direct_v1.workflow import point


class ConversionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers=load_layers()
        cls.features=np.random.default_rng(123).integers(0,256,(73,36),dtype=np.uint8)

    def test_original_mlp_and_readout_exact(self):
        raw,c=forward(self.features,self.layers)
        np.testing.assert_array_equal(raw,quantized_mlp(self.features,self.layers))
        np.testing.assert_array_equal(probabilities(raw),_refiner_probability(raw,2))
        self.assertEqual(c.executed_synaptic_mac,len(raw)*4480)
        self.assertEqual(c.executed_synaptic_ac,0)

    def test_if_exhaustive_domain_signed_additions(self):
        a=np.tile(np.arange(256,dtype=np.uint8)[:,None],(1,3))
        w=np.array([[127,-128,0],[-2,3,4],[1,-4,7]],np.int8)
        for t in (1,4,8,16,32,64,255):
            ref,c=spike_accumulate(a,w,t,'count_reference')
            actual,event=spike_accumulate(a,w,t,'spike_add')
            np.testing.assert_array_equal(actual,ref)
            self.assertEqual(event.unit_spikes,int(spike_counts(a,t).sum()))
            self.assertEqual(event.executed_synaptic_ac,3*event.unit_spikes)
            self.assertEqual(event.executed_synaptic_mac,0)
            self.assertEqual(c.executed_synaptic_mac,len(a)*w.size)
            self.assertEqual(c.executed_synaptic_ac,0)
        np.testing.assert_array_equal(spike_counts(a,255),a)

    def test_stages_have_exact_endpoint_and_no_weight_update(self):
        original=[(w.copy(),s.copy()) for w,s in self.layers]
        ann,_=forward(self.features,self.layers)
        for stage in ('r1','r2'):
            for backend in ('count_reference','spike_add'):
                actual,_=forward(self.features,self.layers,stage,255,backend)
                np.testing.assert_array_equal(actual,ann)
        for before,after in zip(original,self.layers):
            for x,y in zip(before,after):
                np.testing.assert_array_equal(x,y)
                self.assertFalse(y.flags.writeable)

    def test_chunk_independence_and_no_cross_event_state(self):
        for stage in ('r1','r2'):
            all_raw,total=forward(self.features,self.layers,stage,8,'spike_add')
            parts=[]; merged=Cost()
            for block in np.array_split(self.features,7):
                raw,c=forward(block,self.layers,stage,8,'spike_add')
                parts.append(raw); merged.merge(c)
            np.testing.assert_array_equal(all_raw,np.concatenate(parts))
            self.assertEqual(asdict(total),asdict(merged))

    def test_actual_mac_vs_collapsed_reference(self):
        for stage,expected in (('ann',4480),('r1',2432),('r2',2304)):
            raw,event=forward(self.features,self.layers,stage,4,'spike_add')
            reference,dense=forward(self.features,self.layers,stage,4,'count_reference')
            np.testing.assert_array_equal(raw,reference)
            self.assertEqual(event.executed_synaptic_mac,73*expected)
            self.assertEqual(dense.executed_synaptic_mac,73*4480)
            self.assertEqual(event.executed_synaptic_ac,dense.rate_network_ac)

    def test_zero_spikes_still_count_first_layer(self):
        features=np.zeros((5,36),np.uint8)
        raw,c=forward(features,self.layers,'r2',8,'spike_add')
        self.assertFalse(raw.any())
        self.assertEqual(c.executed_synaptic_mac,5*2304)
        self.assertEqual(c.executed_synaptic_ac,0)
        self.assertEqual(c.if_comparisons,5*(64+32)*8)

    def test_empty_candidate_set(self):
        for stage in ('ann','r1','r2'):
            raw,c=forward(np.empty((0,36),np.uint8),self.layers,stage,4,'spike_add')
            self.assertEqual(raw.shape,(0,4))
            self.assertEqual(probabilities(raw).shape,(0,))
            self.assertTrue(all(v==0 for v in asdict(c).values()))

    def test_invalid_inputs_and_numeric_range(self):
        for t in (0,-1,True,2.5,2**31):
            with self.assertRaises(ValueError): validate_steps(t)
        for a in (np.zeros((2,32),np.uint8),np.zeros((2,36),np.float32)):
            with self.assertRaises(ValueError): forward(a,self.layers)
        with self.assertRaises(ValueError): forward(self.features,self.layers,'full')
        with self.assertRaises(ValueError): forward(self.features,self.layers,'r2',4,'pretend_sparse')

    def test_output_directory_does_not_adopt_old_experiment(self):
        base=ROOT/'results/runs'
        with tempfile.TemporaryDirectory(dir=base) as name:
            directory=Path(name)
            (directory/'old_result.json').write_text('{"keep":true}')
            with self.assertRaisesRegex(ValueError,'pre-existing'):
                with lock_directory(directory,base_identity('unit_test')): pass
            self.assertEqual((directory/'old_result.json').read_text(),'{"keep":true}')
        with self.assertRaises(ValueError): output_path(ROOT/'software')

    def test_run_identity_requires_new_directory_for_new_protocol(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results/runs') as name:
            with lock_directory(name,{'version':'one'}): pass
            with lock_directory(name,{'version':'one'}): pass
            with self.assertRaisesRegex(ValueError,'Different conversion identity'):
                with lock_directory(name,{'version':'two'}): pass

    def test_sequence_resume_and_rejected_targets_not_dropped(self):
        raw,_=forward(self.features,self.layers)
        n=len(raw)+2
        a={'features':self.features,'candidate':np.r_[np.ones(len(raw),bool),[False,False]],
           'label':np.ones(n,np.uint8),'teacher':np.r_[probabilities(raw),np.zeros(2,np.float32)],'logits':raw}
        row={'sequence':'train/train_002.npz','cache_sha256':'synthetic-test',
             'high_density':False,'proposal_synaptic_mac':100}
        manifest={'sequences':[row]}
        with tempfile.TemporaryDirectory(dir=ROOT/'results/runs') as name:
            with patch('wangdao.h2pro_direct_v1.workflow.load_sequence',return_value=a):
                first=point(Path(name),manifest,self.layers,'ann',255,'count_reference',Path(name),16)
            with patch('wangdao.h2pro_direct_v1.workflow.load_sequence',side_effect=AssertionError('must reuse receipt')):
                second=point(Path(name),manifest,self.layers,'ann',255,'count_reference',Path(name),16)
            self.assertEqual(first,second)
            self.assertEqual(first['overall']['events'],n)
            self.assertEqual(first['overall']['proposal_fn'],2)
            self.assertFalse(first['end_to_end_complete'])
            self.assertTrue(first['teacher_exact'])


if __name__=='__main__':
    unittest.main()
