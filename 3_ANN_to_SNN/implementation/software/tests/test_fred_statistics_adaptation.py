"""Semantic regression of neighborhood-only full-coverage adaptation; no training."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).absolute().parents[1]/'scripts'))
import adapt_fred_statistics as run
from wangdao.h2.constants import EVENT_FEATURE_SPECS
from wangdao.h2.features import NEIGHBOR_OFFSETS, compress_counts

a=run.adapter


def example():
    rng=np.random.default_rng(123)
    e=np.zeros(1800,dtype=run.frozen.EVENT_DTYPE)
    e['t']=np.sort(rng.choice(np.arange(3.25,203.25,.25),len(e)))
    e['x']=rng.choice([0,1,2,7,8,9,10,344,345],len(e))
    e['y']=rng.choice([0,1,4,8,9,258,259],len(e))
    e['p']=rng.integers(0,2,len(e))
    return e,int(np.searchsorted(e['t'],103.25)),3.25


def oracle(context,queries,updates,t0):
    rows=[]
    for event,is_update in zip(queries,updates):
        row=[]
        for cell,ms in EVENT_FEATURE_SPECS:
            x,y,p,t=int(event['x'])//cell,int(event['y'])//cell,int(event['p']),int(np.floor((event['t']-t0)/ms))
            cx,cy=context['x']//cell,context['y']//cell
            ct=np.floor((context['t']-t0)/ms).astype(np.int64)
            same=(cx==x)&(cy==y)&(ct==t)
            row.extend([int(np.count_nonzero(same&(context['p']==p)))+int(not is_update),
                        int(np.count_nonzero(same&(context['p']!=p)))])
            for lag in (1,2):
                for polarity in (p,1-p):
                    total=0
                    for dx,dy in NEIGHBOR_OFFSETS:
                        total+=int(np.count_nonzero((cx==x+dx)&(cy==y+dy)&(ct==t-lag)&(context['p']==polarity)))
                    row.append(total)
        rows.append(row)
    return compress_counts(np.asarray(rows,np.int64).reshape(-1,24))


class FredStatisticsTests(unittest.TestCase):
    def test_denominator_one_exactly_reproduces_original_count_columns(self):
        e,start,t0=example()
        original=run.base_features(e,start,t0)
        got=a.count_columns(e,e[start:],np.ones(len(e)-start,bool),t0)
        np.testing.assert_array_equal(got,original[:,4:28])

    def test_sparse_context_matches_independent_loop_oracle_at_sensor_edges(self):
        e,start,t0=example();keep=a.update_mask(np.arange(len(e)),16)
        ids=np.linspace(start,len(e)-1,100,dtype=np.int64)
        np.testing.assert_array_equal(a.count_columns(e[keep],e[ids],keep[ids],t0),oracle(e[keep],e[ids],keep[ids],t0))

    def test_updating_queries_match_original_sparse_stream_and_self_is_not_doubled(self):
        e,start,t0=example();keep=a.update_mask(np.arange(len(e)),16)
        sparse=e[keep];query=e[start:][keep[start:]]
        expected=run.base_features(sparse,int(keep[:start].sum()),t0)[:,4:28]
        actual=a.count_columns(sparse,query,np.ones(len(query),bool),t0)
        np.testing.assert_array_equal(actual,expected)

    def test_no_state_updates_still_produce_every_query_and_self_count(self):
        e,start,t0=example();q=e[start:]
        result=a.count_columns(e[:0],q,np.zeros(len(q),bool),t0)
        self.assertEqual(result.shape,(len(q),24))
        expected=np.zeros_like(result);expected[:,[0,6,12,18]]=compress_counts(np.array([1]))[0]
        np.testing.assert_array_equal(result,expected)

    def test_only_count_columns_change_and_input_is_not_modified(self):
        rng=np.random.default_rng(3);features=rng.integers(0,256,(80,36),dtype=np.uint8)
        saved=features.copy();counts=np.zeros((80,24),np.uint8)
        adapted=a.replace_counts(features,counts)
        np.testing.assert_array_equal(adapted[:,a.UNCHANGED_COLUMNS],saved[:,a.UNCHANGED_COLUMNS])
        np.testing.assert_array_equal(features,saved)
        self.assertEqual(adapted.shape,features.shape)

    def test_query_chunking_and_source_selection_are_invariant(self):
        e,start,t0=example();ids=np.arange(len(e));keep=a.update_mask(ids,16,1)
        np.testing.assert_array_equal(keep,run.prior.keep_events(ids,16,1))
        np.testing.assert_array_equal(keep,np.concatenate([a.update_mask(x,16,1) for x in np.array_split(ids,9)]))
        q=e[start:];flags=keep[start:];context=e[keep]
        expected=a.count_columns(context,q,flags,t0)
        actual=np.concatenate([a.count_columns(context,q[x],flags[x],t0) for x in np.array_split(np.arange(len(q)),7)])
        np.testing.assert_array_equal(actual,expected)

    def test_extra_labels_cannot_change_counts_or_update_selection(self):
        e,start,t0=example();keep=a.update_mask(np.arange(len(e)),16)
        labelled=np.zeros(len(e),dtype=e.dtype.descr+[('label','u1'),('valid','?')])
        for field in e.dtype.names:labelled[field]=e[field]
        labelled['label']=1;labelled['valid']=False
        np.testing.assert_array_equal(a.count_columns(e[keep],e[start:],keep[start:],t0),
                                      a.count_columns(labelled[keep],labelled[start:],keep[start:],t0))

    def test_query_only_metrics_keep_targets_in_the_full_denominator(self):
        # Both weak targets are detected: one state updater and one query-only event.
        bits=np.array([64+16+32+1+2+4+8+256,64+16+32+1+2+4+8,64+16],np.uint16)
        totals={};run.score(bits,totals);m=run.report_metrics(totals)['h2pro_ann']
        self.assertEqual(m['all']['target_events'],2)
        self.assertEqual(m['all']['recall'],1.)
        self.assertEqual(m['query_only']['target_events'],1)
        self.assertEqual(m['query_only']['recall'],1.)


if __name__=='__main__':
    unittest.main(verbosity=2)
