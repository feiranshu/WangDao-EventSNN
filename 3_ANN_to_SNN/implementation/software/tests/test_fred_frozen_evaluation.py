"""Regression checks for the independent FRED scorer and frozen model selection."""
from pathlib import Path
import hashlib
import json
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "scripts"))
import evaluate_fred_frozen as ev
from wangdao.h2pro_direct_v1.workflow import cache_manifest, load_sequence


class FredFrozenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.models = ev.FrozenModels()

    def test_mask_and_proposal_false_negatives(self):
        y = np.array([1,1,0,0,1],bool)
        c = np.array([0,1,1,0,1],bool)
        p = np.array([0,1,1,0,0],bool)
        v = np.array([1,1,1,1,0],bool)
        counts = ev.counts(y,p,c,v)
        self.assertEqual([counts[k] for k in ("tp","fp","fn","tn","proposal_fn","refiner_fn")], [1,1,1,1,1,0])
        self.assertEqual(ev.summarize(counts)["iou"],1/3)
        self.assertEqual(ev.summarize(counts)["proposal_recall"],.5)

    def test_bit_encoding_and_tail_exclusion(self):
        # First event: valid positive proposed/predicted by all. Second: invalid
        # positive. Third: valid positive in unfinished tail. Fourth: valid FP.
        bits=np.array([1+2+4+8+16+32+64+128, 1+32+64, 16+32, 1+2+4+8+16+64],np.uint8)
        totals=ev.new_totals()
        ev.score_bits(bits,totals)
        for model in ev.MODELS:
            c=totals[model]["all"]
            self.assertEqual([c[k] for k in ("events","tp","fp","fn")],[2,1,1,0])
            self.assertEqual(totals[model]["opening_5s"]["tp"],1)
            self.assertEqual(totals[model]["after_5s"]["fp"],1)

    def test_cross_shard_state_equals_unsplit(self):
        rng=np.random.default_rng(1234)
        events=np.empty(1200,dtype=ev.EVENT_DTYPE)
        events['t']=np.linspace(4.56,327.21,len(events))
        events['x']=rng.integers(0,346,len(events))
        events['y']=rng.integers(0,260,len(events))
        events['p']=rng.integers(0,2,len(events))
        end=np.nextafter(events['t'][-1],np.inf)
        original=self.models.pipeline();original.reset(events['t'][0])
        original.push(events,end);expected,tail=original.finalize(end)
        split=self.models.pipeline();split.reset(events['t'][0])
        actual=[];lo=0
        for right in [17.0,100.0,104.56,168.0,204.56,251.0,end]:
            hi=np.searchsorted(events['t'],right,side='left')
            split.push(events[lo:hi],right)
            actual.extend(split.flush_completed_windows());lo=hi
        final,tail_split=split.finalize(end);actual.extend(final)
        self.assertEqual(tail,tail_split);self.assertGreater(tail,0)
        self.assertEqual(len(actual),3)
        for a,b in zip(actual,expected):
            for field in ('event_indices','candidate_mask','features','probabilities','predictions','heatmap'):
                np.testing.assert_array_equal(getattr(a,field),getattr(b,field))

    def test_exact_encoder_identity_and_spike_reference(self):
        self.assertEqual([ev.asdict(e) for e in self.models.encoders],
            [{'threshold':255,'initial_membrane':191},{'threshold':255,'initial_membrane':127}])
        x=np.random.default_rng(9).integers(0,256,(513,36),dtype=np.uint8)
        for model in ev.MODELS[1:]:
            reference,rc=self.models.snn(x,model)
            explicit,ec=self.models.snn(x,model,'spike_add')
            np.testing.assert_array_equal(reference,explicit)
            self.assertEqual(rc.rate_network_ac,ec.executed_synaptic_ac)
            self.assertEqual(rc.unit_spikes,ec.unit_spikes)

    def test_full_original_dev_replay(self):
        directory=ev.ROOT/'results/runs/h2pro_direct_v1_features'
        manifest=cache_manifest(directory)
        prior_r1=ev.read_json(ev.ROOT/'results/runs/h2pro_task_calibration_v1_dev/r1_t16_task_rescore.json')
        prior_r2=ev.read_json(ev.STUDENT.parent/'epochs/epoch_088.json')['evaluation']
        r1_by_seq={r['sequence']:r for r in prior_r1['sequences']}
        totals={m:ev.zero_counts() for m in ev.MODELS}
        ac={m:0 for m in ev.MODELS}
        rows=[]
        for row in manifest['sequences']:
            a=load_sequence(directory,row)
            n=len(a['label']);candidate=a['candidate'];features=a['features']
            for model in ev.MODELS:
                parts=[]
                for lo in range(0,len(features),32768):
                    x=features[lo:lo+32768]
                    if model==ev.MODELS[0]:raw,cost=ev.forward(x,self.models.ann,'ann')
                    else:raw,cost=self.models.snn(x,model)
                    parts.append(raw);ac[model]+=cost.rate_network_ac
                raw=np.concatenate(parts) if parts else np.empty((0,4),np.float32)
                p=np.zeros(n,np.float32);p[candidate]=ev.probabilities(raw)
                if model==ev.MODELS[0]:
                    np.testing.assert_array_equal(raw,a['logits'])
                    np.testing.assert_array_equal(p,a['teacher'])
                if model==ev.MODELS[1]:
                    digest=hashlib.sha256(raw.astype('<f4').tobytes()).hexdigest()
                    self.assertEqual(digest,r1_by_seq[row['sequence']]['logits_sha256'])
                c=ev.counts(a['label'].astype(bool),p>=.35,candidate,np.ones(n,bool))
                ev.add_counts(totals[model],c)
            rows.append(row['sequence'])
        for model,prior in [(ev.MODELS[1],prior_r1),(ev.MODELS[2],prior_r2)]:
            for key in ev.COUNT_KEYS:self.assertEqual(totals[model][key],prior['overall'][key])
        self.assertEqual(ac[ev.MODELS[1]],prior_r1['refiner_cost']['executed_synaptic_ac'])
        self.assertEqual(ac[ev.MODELS[2]],prior_r2['cost']['inference_equivalent_ac'])
        ev.atomic_json(ev.OUTPUT/'original_dev_replay.json',{
            'status':'passed','sequences':rows,'models':{m:ev.summarize(totals[m]) for m in ev.MODELS},
            'ac':ac,'r1_full_logits_exact':True,'student_epoch':88,'events':totals[ev.MODELS[0]]['events'],
            'training_started':False})


if __name__=='__main__':
    unittest.main(verbosity=2)
