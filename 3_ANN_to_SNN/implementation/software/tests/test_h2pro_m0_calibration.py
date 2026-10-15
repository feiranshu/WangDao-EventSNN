"""Frozen identity, split isolation, population estimates, selection and recovery."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'software'))
from wangdao.h2pro_m0_calibration_v2 import common,search,workflow
from wangdao.h2pro_channel_calibration_v1.core import forward,load_layers
from wangdao.h2pro_channel_calibration_v1.workflow import evaluate_config

GROUPS=[list(range(i,i+8)) for i in range(0,64,8)]

def temporary_directory():
    return tempfile.TemporaryDirectory(prefix='m0-tests-',dir=ROOT/'results/runs')


def score(iou=.8,recall=.9,hd=.7,ac=100,pid='base'):
    return {'overall':{'iou':iou,'recall':recall},'high_density':{'recall':hd},
            'cost':{'equivalent_ac':ac},'parameter_id':pid}


class M0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.layers=load_layers();cls.device,_=search.setup('cpu')
        cls.x=np.random.default_rng(2026).integers(0,256,(41,36),dtype=np.uint8)
        cls.teacher,_=forward(cls.x,cls.layers,'ann')

    def sequences(self):
        out=[]
        for i in range(8):
            x=self.x;candidate=np.r_[np.ones(len(x),bool),False,False]
            labels=(np.arange(len(candidate))%3==0).astype(np.uint8);labels[-1]=1
            out.append({'features':x,'labels':labels,'candidate':candidate,'teacher':self.teacher,
                        'row':{'sequence':f'seq{i}','role':'train_main','high_density':i>=4,
                               'events':len(candidate),'target_events':int(labels.sum()),
                               'candidate_events':len(x),'candidate_targets':int(labels[candidate].sum()),
                               'proposal_sparse_mac':7,'proposal_dense_mac':11}})
        return out

    def test_split_stable_disjoint_density_stratified(self):
        rows=[s['row'] for s in self.sequences()]
        a=common.split_sequences(rows);b=common.split_sequences(rows[::-1])
        self.assertEqual(set(a['fit']),set(b['fit']));self.assertEqual(set(a['validation']),set(b['validation']))
        self.assertFalse(set(a['fit']) & set(a['validation']))
        self.assertEqual(set(a['fit'])|set(a['validation']),{r['sequence'] for r in rows})
        for names in [a['fit'],a['validation']]:
            self.assertEqual({r['high_density'] for r in rows if r['sequence'] in names},{False,True})
        rows[0]['role']='train_dev'
        with self.assertRaises(ValueError):common.split_sequences(rows)

    def test_sparse_m0_contract_and_historical_anchor(self):
        a=common.anchor();e=common.encoders({'0':175,'6':207},GROUPS)
        self.assertEqual(e[0].theta,a[0].theta);self.assertEqual(e[1],a[1])
        self.assertEqual(sum(x!=y for x,y in zip(e[0].membrane,a[0].membrane)),16)
        for invalid in ({'0':191},{'0':174},{'8':192},{'0':192,'1':192,'2':192}):
            with self.assertRaises(ValueError):common.encoders(invalid,GROUPS)
        broken=copy.deepcopy(GROUPS);broken[1][0]=0
        with self.assertRaises(ValueError):common.encoders({},broken)

    def test_all_positive_sampling_and_population_mass(self):
        seqs=self.sequences()
        with patch.dict(common.PROTOCOL['sample'],{'background_per_fit_sequence':5}):
            a=common.sample_fit(seqs);b=common.sample_fit(seqs)
        np.testing.assert_array_equal(a['features'],b['features'])
        self.assertEqual(int(a['labels'].sum()),sum(s['row']['candidate_targets'] for s in seqs))
        self.assertAlmostEqual(float(a['weights'].sum()),sum(len(s['features']) for s in seqs))
        self.assertEqual(sum(r['background_sampled'] for r in a['receipts']),40)
        seqs[0]['row']['role']='train_dev'
        with self.assertRaises(ValueError):common.sample_fit(seqs)

    def test_fit_metric_matches_complete_event_scoring_and_rejected_targets(self):
        seqs=self.sequences();data=common.sample_fit(seqs)
        runner=search.FitRunner(data,self.layers,self.device)
        e=common.encoders({'2':187},GROUPS);actual=runner.assess(e)
        expected,_=evaluate_config(seqs,self.layers,self.device,'r1',16,e,chunk_rows=13)
        for g in ('overall','ordinary','high_density'):
            for k in ('tp','fp','fn','iou','precision','recall'):
                self.assertAlmostEqual(actual[g][k],expected[g][k],places=11)
        self.assertEqual(actual['cost']['equivalent_ac'],expected['cost']['equivalent_ac'])
        groups,means=runner.grouping();common.validate_groups(groups)
        self.assertTrue(all(not w.requires_grad for w in runner.weights))

    def test_cpu_and_cuda_m0_exact_with_explicit_add(self):
        e=common.encoders({'1':176,'7':206},GROUPS)
        expected,c=forward(self.x,self.layers,'r1',16,e,'spike_add')
        for device in ('cpu','cuda') if torch.cuda.is_available() else ('cpu',):
            d,_=search.setup(device);runner=search.Runner(self.x,self.layers,d)
            raw,n1,n2=runner.forward('r1',16,e)
            np.testing.assert_array_equal(raw.cpu().numpy(),expected)
            self.assertEqual(int(n1.sum()*32+n2.sum()*4),c['equivalent_ac'])
            self.assertEqual(c['extra_threshold_decode_mul'],0)

    def test_one_objective_preserves_recall_and_ac(self):
        reference=score()
        bad=score(iou=.85,hd=.69,pid='bad');good=score(iou=.81,ac=99,pid='good')
        self.assertLess(common.rank(good,reference,{'0':190}),common.rank(bad,reference,{'0':190}))
        self.assertEqual(common.constraints(bad,reference),['hd_recall'])
        self.assertIn('AC',common.constraints(score(ac=101),reference))

    def test_main_gate_requires_fit_and_validation_and_keeps_anchor(self):
        def record(name,changes,views):
            return {'parameter_id':name,'changes':changes,'encoders':[],'partitions':views}
        base=record('base',{}, {g:score() for g in ('fit','validation','all')})
        changed=record('bad',{'0':190},{g:score(.81,pid='bad') for g in ('fit','validation','all')})
        changed['partitions']['validation']=score(.79,pid='bad')
        out=workflow.choose([base,changed])
        self.assertEqual(out['primary']['parameter_id'],'base')
        self.assertFalse(out['changed_1_group']['eligible'])
        self.assertEqual(out['changed_1_group']['eligibility_failures']['validation'],['overall_iou'])
        changed['partitions']['validation']=score(.82,pid='bad')
        out=workflow.choose([base,changed]);self.assertEqual(out['primary']['parameter_id'],'bad')

    def test_partition_counts_and_cost_recombine(self):
        seqs=self.sequences();split=common.split_sequences([s['row'] for s in seqs])
        r,_=evaluate_config(seqs,self.layers,self.device,'r1',16,common.anchor(),chunk_rows=11)
        a=workflow.partition(r,seqs,split['fit']);b=workflow.partition(r,seqs,split['validation'])
        for g in ('overall','ordinary','high_density'):
            for k in ('tp','fp','fn','events','target_events'):
                self.assertEqual(a[g][k]+b[g][k],r[g][k])
        for k in ('equivalent_ac','mac_sparse_proposal','mac_dense_proposal'):
            self.assertEqual(a['cost'][k]+b['cost'][k],r['cost'][k])

    def test_interrupted_fit_and_completed_reuse(self):
        class Fake:
            def __init__(self,fail=None):self.calls=0;self.fail=fail
            def assess(self,e):
                self.calls+=1
                if self.calls==self.fail:raise InterruptedError('simulated power loss')
                d=sum(191-v for v in e[0].membrane)
                return score(.8+d*1e-5,ac=100-d*.01,pid=common.parameter_id(e))
        with temporary_directory() as tmp, patch.dict(common.PROTOCOL,{'m0_offsets':[-1,1],'shortlist_per_group_count':2}),contextlib.redirect_stdout(io.StringIO()):
            p=Path(tmp)/'resume';q=Path(tmp)/'fresh';p.mkdir();q.mkdir()
            with self.assertRaises(InterruptedError):search.fit(Fake(4),GROUPS,p)
            resumed=search.fit(Fake(),GROUPS,p);fresh=search.fit(Fake(),GROUPS,q)
            self.assertEqual(resumed['candidates'],fresh['candidates'])
            self.assertEqual(resumed['evaluated_unique_parameters'],129)
            before={f.name:common.sha256(f) for f in p.rglob('*.json')}
            repeated=search.fit(Fake(1),GROUPS,p);self.assertEqual(resumed,repeated)
            self.assertEqual(before,{f.name:common.sha256(f) for f in p.rglob('*.json')})
            target=next((p/'fit_candidates').glob('*.json'));target.write_text('{}')
            with self.assertRaises(ValueError):search.fit(Fake(),GROUPS,p)

    def test_dev_requires_main_seal_without_loading_data(self):
        with temporary_directory() as tmp,patch.object(workflow,'load_role',side_effect=AssertionError('data loaded')):
            with self.assertRaises(ValueError):workflow.evaluate(Path(tmp),Path(tmp),self.device)

    def test_complete_main_reuse_does_not_read_sequences(self):
        with temporary_directory() as tmp,patch.object(workflow,'load_role',side_effect=AssertionError('data loaded')):
            p=Path(tmp);saved={'main_evidence_sha256':{},'status':'complete'}
            common.atomic_json(p/'selection.json',saved)
            self.assertEqual(workflow.calibrate(p,p,self.device),saved)


if __name__=='__main__':unittest.main()
