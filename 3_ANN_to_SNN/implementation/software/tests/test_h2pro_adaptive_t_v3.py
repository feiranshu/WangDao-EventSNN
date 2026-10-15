import copy
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np
from wangdao.h2pro_adaptive_t_v3.common import (DESIGN,ANCHORS,OBJECTIVES,SOURCE,REFERENCE,PRIMARY_V2,
    EXTRA_V2,DEFAULT_CACHE,sealed_json,sha256)
from wangdao.h2pro_adaptive_t_v3.selection import grids,rank,select_main
from wangdao.h2pro_adaptive_t_v3.analysis import paired_analysis
from wangdao.h2pro_adaptive_t_v3 import experiment
from wangdao.h2pro_adaptive_t_v2.core import Policy,incremental,choose,ledger
from wangdao.h2pro_adaptive_t_v2.scoring import score,subset,policy_id
from wangdao.h2pro_adaptive_t_v1.core import trace_reference
from wangdao.h2pro_calibration_v1.core import probabilities
from software.tests.test_h2pro_adaptive_t_v2 import synthetic,layers


@contextmanager
def workspace_tmp():
    root=(SOURCE.parent/'h2pro_adaptive_t_v3_checks_20261015/tmp').resolve()
    assert SOURCE.parent.resolve() in root.parents
    root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as name:
        path=Path(name).resolve();assert path.parent==root
        yield path


class SingleFactorTests(unittest.TestCase):
    def test_only_one_t8_parameter_varies_and_t4_is_frozen(self):
        gs=grids();self.assertEqual(sum(map(len,gs.values())),42)
        for arm,ps in gs.items():
            for p in ps:
                self.assertEqual(p.at(4),(.01,.99))
                if arm=='negative_only':self.assertEqual(p.positive8,.99)
                else:self.assertEqual(p.negative8,.01)
            self.assertIn(ANCHORS['shared'],ps);self.assertIn(ANCHORS[arm],ps)
        self.assertEqual(len({policy_id(p) for ps in gs.values() for p in ps}),41)

    def test_fixed_factorial_anchor_execution_matches_frozen_prefixes(self):
        x=np.random.default_rng(321).integers(0,256,(67,36),dtype=np.uint8);ls=layers()
        raw,spikes=trace_reference(x,ls);p=np.stack([probabilities(r) for r in raw])
        for policy in ANCHORS.values():
            y,n,ix,c=incremental(x,ls,policy,19)
            np.testing.assert_array_equal(ix,choose(p,policy))
            np.testing.assert_array_equal(y,raw[ix,np.arange(len(ix))])
            np.testing.assert_array_equal(n,spikes[ix,np.arange(len(ix))])
            expected=ledger(ix,n,policy)
            self.assertEqual(c,{k:expected[k] for k in c})

    def test_missing_target_cohort_and_factorial_attribution(self):
        data=dict(role='train_dev',p=np.array([[.5]*5,[.05,.97,.05,.97,.5],[.8,.1,.1,.8,.7]],np.float32),
                  labels=np.array([1,0,0,1,1],np.uint8),sequences=[
                      dict(row={'high_density':False},lo=0,hi=3),dict(row={'high_density':True},lo=3,hi=5)])
        out=paired_analysis(data,{k:p.as_dict() for k,p in ANCHORS.items()})
        r=out['policies'];self.assertEqual(out['factorial_nonadditive_decisions'],0)
        self.assertEqual(r['combined']['overall']['old_combined_extra_misses'],1)
        self.assertEqual(r['positive_only']['overall']['recovered_from_old_combined_misses'],1)
        self.assertEqual(r['negative_only']['overall']['recovered_from_old_combined_misses'],0)
        self.assertEqual(r['positive_only']['overall']['vs_shared']['extra_background_false_positives'],1)
        self.assertEqual(r['negative_only']['high_density']['vs_shared']['additional_target_misses'],0)

    def test_main_selection_and_all_preregistered_diagnostics(self):
        data=synthetic();selection=select_main(data)
        self.assertEqual(selection['design'],DESIGN)
        self.assertFalse(set(selection['split']['fit'])&set(selection['split']['validation']))
        gs=grids()
        for arm,record in selection['arms'].items():
            for objective in OBJECTIVES:
                ix=max(record['shortlist'],key=lambda i:(rank(record['validation_candidates'][str(i)],objective,
                                                             selection['baselines']['validation']),-i))
                self.assertEqual(record['selected'][objective]['grid_index'],ix)
                self.assertEqual(selection['evaluation_policies'][arm+'_'+objective],gs[arm][ix].as_dict())
            self.assertIn(gs[arm].index(ANCHORS[arm]),record['shortlist'])
        for name,policy in ANCHORS.items():self.assertEqual(selection['evaluation_policies'][name],policy.as_dict())
        self.assertFalse(selection['dev_used_for_selection'])

    def test_recovery_objective_has_explicit_equal_metric_weights(self):
        r=score(synthetic(),ANCHORS['shared']);base=score(synthetic(),fixed_step=16)
        expected=(r['overall']['iou']+r['overall']['recall']+r['high_density']['recall'])/3
        expected-=.0025*r['cost']['inference_equivalent_ac']/base['cost']['inference_equivalent_ac']
        self.assertAlmostEqual(rank(r,'recovery',base)[0],expected)

    def test_dev_and_mixed_roles_rejected(self):
        data=synthetic();data['role']='train_dev'
        with self.assertRaises(ValueError):select_main(data)
        data['role']='train_main';data['sequences'][0]['row']['role']='train_dev'
        with self.assertRaises(ValueError):select_main(data)

    def test_extra_teacher_fields_do_not_influence_selection(self):
        data=synthetic();a=select_main(data);data['teacher']=np.full((200,4),np.nan)
        self.assertEqual(a,select_main(data))


class RecoveryTests(unittest.TestCase):
    def test_main_seal_binds_selection_and_identity(self):
        with workspace_tmp() as run:
            sealed_json(run/'identity.json',{'test':1})
            s=dict(role='train_main',dev_used_for_selection=False)
            with self.assertRaisesRegex(ValueError,'Seal Main'):experiment.require_seal(run)
            experiment.seal_main(run,s,[]);before=sha256(run/'main_seal.json')
            self.assertEqual(experiment.require_seal(run),s)
            experiment.seal_main(run,s,[]);self.assertEqual(sha256(run/'main_seal.json'),before)
            (run/'selection.json').write_text('{}',encoding='utf-8')
            with self.assertRaises((ValueError,KeyError)):experiment.require_seal(run)

    def test_dev_cannot_load_before_main_seal(self):
        with workspace_tmp() as run:
            with patch.object(experiment,'identity',return_value=({'test':1},None)), \
                 patch.object(experiment,'manifest',return_value={'roles':{}}), \
                 patch.object(experiment,'bundle',side_effect=AssertionError('No Dev data allowed')):
                with self.assertRaisesRegex(ValueError,'Seal Main'):experiment.run('dev',run/'run')

    def test_all_existing_run_and_cache_paths_protected(self):
        for path in (SOURCE,REFERENCE,PRIMARY_V2,EXTRA_V2,DEFAULT_CACHE,EXTRA_V2/'child',EXTRA_V2.parent):
            with self.assertRaises(ValueError):experiment.run(directory=path)

    def test_completed_resume_never_reselects_or_infers_and_detects_corruption(self):
        with workspace_tmp() as run:
            ident={'test':1};sealed_json(run/'identity.json',ident)
            experiment.seal_main(run,dict(role='train_main',dev_used_for_selection=False),[])
            payload=run/'evidence.json';sealed_json(payload,{'value':1})
            sealed_json(run/'comparison.json',dict(artifact_sha256={'evidence.json':sha256(payload)}))
            with patch.object(experiment,'identity',return_value=(ident,None)), \
                 patch.object(experiment,'manifest',side_effect=AssertionError('No manifest reload')), \
                 patch.object(experiment,'bundle',side_effect=AssertionError('No data reload')), \
                 patch.object(experiment,'select_main',side_effect=AssertionError('No reselection')), \
                 patch.object(experiment,'verify_policy',side_effect=AssertionError('No inference')):
                self.assertTrue(experiment.run(directory=run)['reused'])
                payload.write_text('{}',encoding='utf-8')
                with self.assertRaisesRegex(ValueError,'artifact'):experiment.run(directory=run)


if __name__=='__main__':unittest.main()
