"""Input-control, denominator and source-label regressions; no parameter updates."""
import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).absolute().parents[1]/'scripts'))
import diagnose_fred_inputs as d
from wangdao.fred_adapter_v1.core import label_events


class FredInputDiagnosisTests(unittest.TestCase):
    def test_label_interpolation_boundaries_and_missing_annotations(self):
        times=np.array([0,33333,66666,99999],np.int64)
        ann={'times':times,'boxes':[{}, {1:np.array([1.,1.,5.,5.])},
             {1:np.array([2.,1.,6.,5.])},{}], 'classes':{1:'test'},
             'id_map':{1:1},'frame_step_us':33333.}
        raw=np.zeros(24,dtype=d.source.DTYPE)
        raw['x']=np.tile([0,2,4,5],6);raw['y']=2
        raw['t']=np.repeat([0,33333,40000,66666,80000,99999],4)
        expected,_,valid=label_events(raw,ann)
        actual,v,_,_=d.source.independent_labels(raw,times,d.source.dense_annotations(ann))
        np.testing.assert_array_equal(expected,actual)
        np.testing.assert_array_equal(valid,v)
        self.assertFalse(valid[16:20].any())

    def test_identity_geometry_keeps_original_pixel_center_contract(self):
        raw=np.zeros(1280*720,dtype=d.source.DTYPE)
        raw['x']=np.tile(np.arange(1280),720)
        raw['y']=np.repeat(np.arange(720),1280)
        packet,view=d.transform(raw,1)
        np.testing.assert_array_equal(packet['x'],np.floor((raw['x']+.5)*.2703125))
        np.testing.assert_array_equal(packet['y'],np.floor((raw['y']+.5)*.2703125+32.6875))
        self.assertFalse(view.any())

    def test_tiles_cover_every_source_pixel_once_without_label_dependence(self):
        raw=np.zeros(1280*720,dtype=d.source.DTYPE)
        raw['x']=np.tile(np.arange(1280),720);raw['y']=np.repeat(np.arange(720),1280)
        a,views=d.transform(raw,2)
        self.assertEqual(np.bincount(views).tolist(),[640*360]*4)
        self.assertTrue(np.all((a['x']>=0)&(a['x']<346)&(a['y']>=0)&(a['y']<260)))
        raw['label']=1;raw['valid']=False
        b,v2=d.transform(raw,2)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(views,v2)

    def test_density_selection_is_nested_and_independent_of_chunking(self):
        ids=np.arange(1_000_000,dtype=np.int64)
        a,b=d.keep_events(ids,4),d.keep_events(ids,16)
        self.assertFalse(np.any(b&~a))
        self.assertLess(abs(a.mean()-.25),.002)
        self.assertLess(abs(b.mean()-.0625),.002)
        chunks=np.concatenate([d.keep_events(x,4) for x in np.array_split(ids,37)])
        np.testing.assert_array_equal(chunks,a)
        self.assertTrue(d.keep_events(ids,1).all())

    def test_retained_and_full_source_denominators_are_distinct(self):
        # Two targets: retained TP and dropped target; two background events.
        old=np.array([64+16+32+1+2+4+8,64+16+32+1,64+16+1+2,64+16],np.uint8)
        new=old.astype(np.uint16)&(64+16+32+128)
        new[[0,3]]|=256
        new[0]|=1+2+4+8
        totals={};d.update_metrics(totals,new,old)
        m=d.metric_report(totals)['h2pro_ann']
        self.assertEqual(m['retained']['all']['events'],2)
        self.assertEqual(m['retained']['all']['recall'],1.)
        self.assertEqual(m['full_source']['all']['events'],4)
        self.assertEqual(m['full_source']['all']['recall'],.5)
        self.assertEqual(m['paired_baseline']['all']['events'],2)
        self.assertEqual(m['paired_baseline']['all']['fp'],0)

    def test_original_pipeline_does_not_receive_labels_or_retention_flags(self):
        raw=np.zeros(1000,dtype=d.source.DTYPE)
        raw['x']=np.arange(1000)%1280;raw['y']=360;raw['t']=np.arange(1000)*200
        raw['label']=1;raw['valid']=True
        packet,_=d.transform(raw)
        self.assertEqual(packet.dtype.names,('x','y','t','p'))
        model=d.frozen.FrozenModels()
        p1,p2=model.pipeline(),model.pipeline()
        for p in (p1,p2):p.reset(0.)
        p1.push(packet,200.)
        p2.push(packet[:500],100.);p2.push(packet[500:],200.)
        a,b=p1.flush_completed_windows(),p2.flush_completed_windows()
        for u,v in zip(a,b):
            np.testing.assert_array_equal(u.features,v.features)
            np.testing.assert_array_equal(u.predictions,v.predictions)


if __name__=='__main__':
    unittest.main(verbosity=2)
