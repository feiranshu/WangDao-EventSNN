"""Frozen Test adapter regression: synthetic arrays and existing Dev only."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import evaluate_h2pro_frozen_test as ev


class FrozenTestEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.layers = ev.load_export(ev.SOURCE/'epochs/epoch_159_int8.npz')

    def test_index_is_full_test_only(self):
        index = ev.development_index()
        records = ev.test_records(index)
        self.assertEqual(len(records), 24)
        self.assertTrue(all(r['path'].startswith('test/test_') for r in records))
        bad = copy.deepcopy(index)
        bad['roles']['test'][0] = '../train_000.npz'
        with self.assertRaises(ValueError): ev.test_records(bad)
        bad = copy.deepcopy(index)
        bad['roles']['test'].pop()
        with self.assertRaises(ValueError): ev.test_records(bad)

    def test_no_candidates(self):
        arrays, checks = ev.infer(np.empty((0,36), np.uint8), self.layers, 17)
        self.assertEqual(checks['all_candidates_actual_spike_verified'], 0)
        self.assertEqual(arrays['adaptive_raw'].shape, (0,4))

    def test_full_event_rejects_are_counted(self):
        a = dict(labels=np.array([1,1,0,0],np.uint8), candidate=np.array([False,True,True,False]),
                 teacher=np.array([[0,0,4,4],[0,0,4,4]],np.float32))
        a.update(fixed_raw=a['teacher'], adaptive_raw=a['teacher'])
        c = ev.count_models(a)[ev.MODELS[0]]
        self.assertEqual((c['tp'],c['fp'],c['fn'],c['tn']), (1,1,1,1))
        self.assertEqual(c['proposal_fn'], 1)

    def test_batching_does_not_change_predictions(self):
        x = np.random.default_rng(42).integers(0,256,(67,36),np.uint8)
        a, va = ev.infer(x,self.layers,13)
        b, vb = ev.infer(x,self.layers,67)
        for k in a: np.testing.assert_array_equal(a[k],b[k])
        self.assertEqual(va,vb)

    def test_seal_binds_identity_and_preservation(self):
        with tempfile.TemporaryDirectory(dir=ev.ROOT/'results/runs') as tmp:
            path = Path(tmp)
            ident = dict(test=1)
            ev.atomic_json(path/'identity.json',ident)
            ev.atomic_json(path/'preservation_before.json',{})
            with self.assertRaises(FileNotFoundError): ev.require_seal(path,ident)
            ev.atomic_json(path/'pretest_seal.json',dict(
                identity_sha256=ev.sha256(path/'identity.json'),
                preservation_sha256=ev.sha256(path/'preservation_before.json')))
            ev.require_seal(path,ident)
            ev.atomic_json(path/'preservation_before.json',dict(changed=True))
            with self.assertRaises(ValueError): ev.require_seal(path,ident)

    def test_completed_reuse_never_reads_test_events(self):
        with tempfile.TemporaryDirectory(dir=ev.ROOT/'results/runs') as tmp:
            path = Path(tmp)
            ident = dict(test=1)
            ev.atomic_json(path/'identity.json',ident)
            ev.atomic_json(path/'preservation_before.json',{})
            ev.atomic_json(path/'pretest_seal.json',dict(
                identity_sha256=ev.sha256(path/'identity.json'),
                preservation_sha256=ev.sha256(path/'preservation_before.json')))
            ev.atomic_json(path/'comparison.json',dict(artifact_sha256={}))
            with patch.object(ev,'identity',return_value=(ident,None)), \
                 patch.object(ev,'load_events',side_effect=AssertionError('No Test reread')):
                self.assertTrue(ev.run('run',path)['reused'])

    def test_raw_dev_pipeline_matches_original_cache(self):
        cache = ev.ROOT/'results/runs/h2pro_distill_v1_features'
        row = ev.read_json(cache/'manifest.json')['roles']['train_dev'][0]
        events = ev.load_events(ev.DEFAULT_DATA/row['sequence'],require_labels=True)
        a, m = ev.features_from_events(events)
        with np.load(cache/row['cache_file'],allow_pickle=False) as z:
            for key in ('features','labels','candidate','teacher'):
                np.testing.assert_array_equal(a[key],z[key])
        for key in ('events','target_events','candidate_events','candidate_targets','high_density',
                    'proposal_sparse_mac','proposal_dense_mac'):
            self.assertEqual(m[key],row[key],key)

    def test_all_dev_models_reproduce_frozen_evidence(self):
        cache = ev.ROOT/'results/runs/h2pro_distill_v1_features'
        manifest = ev.read_json(cache/'manifest.json')
        records, outputs = [], []
        for row in manifest['roles']['train_dev']:
            with np.load(cache/row['cache_file'],allow_pickle=False) as z:
                a = {k:z[k] for k in z.files}
            inferred, checks = ev.infer(a['features'],self.layers)
            a.update(inferred)
            meta = dict(row,raw_events=row['events']+row['trailing_events'],
                        raw_target_events=row['target_events'],trailing_target_events=0)
            records.append(dict(metadata=meta,counts=ev.count_models(a)))
            outputs.append(a)
            self.assertEqual(checks['all_candidates_actual_spike_verified'],row['candidate_events'])
        got = ev.summarize(records,outputs)['models']
        ref = ev.read_json(ev.DEV_REFERENCE)
        prior = ev.read_json(ev.PRIOR/'comparison.json')
        expected = (ref['frozen_ann'],ref['fixed']['16'],prior['adaptive']['accuracy'])
        for name, old in zip(ev.MODELS,expected):
            for group in ('overall','ordinary','high_density'):
                self.assertEqual(got[name][group],old[group])
            for key in ('inference_equivalent_mac_sparse_proposal','inference_equivalent_mac_dense_proposal','inference_equivalent_ac'):
                oldkey = {'inference_equivalent_mac_sparse_proposal':'mac_sparse_proposal',
                          'inference_equivalent_mac_dense_proposal':'mac_dense_proposal',
                          'inference_equivalent_ac':'ac'}[key] if name==ev.MODELS[0] else key
                self.assertEqual(got[name]['cost'][key],old['cost'][oldkey])


if __name__ == '__main__':
    unittest.main()
