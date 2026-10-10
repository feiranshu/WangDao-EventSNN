import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).absolute().parent.parent
SOFTWARE = ROOT.parents[1] / 'implementation' / 'software'
sys.path[:0] = [str(ROOT), str(SOFTWARE)]
from src.stream_state import CausalPipeline, EVENT_DTYPE
from src.causal_feature_builder import base_features
from src.motion_coherence import lmec, proposal_motion
from wangdao.h2.features import extract_event_features
from wangdao.h2.representation import build_patch_features, _fine_counts_to_patch_features

PROPOSAL = SOFTWARE / 'weights' / 'H2_D1_seed42.npz'
REFINER = ROOT.parent / 'step7' / 'intermediate_H2-plus' / 'weights' / 'A4-2_seed42_epoch008_int8.npz'
SAMPLE = ROOT.parents[1] / 'implementation' / 'data' / 'EV-UAV-dataset' / 'train' / 'train_000.npz'


class CausalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(SAMPLE, allow_pickle=False) as archive:
            cls.events = archive['ev'][:].copy()
        cls.start = float(cls.events['t'][0])
        cls.two = cls.events[cls.events['t'] < cls.start+200]

    def pipeline(self, variant='C2-LMEC'):
        p = CausalPipeline(PROPOSAL, REFINER, variant)
        p.reset(self.start)
        return p

    def test_old_35_columns_and_proposal_patch(self):
        current = self.two[self.two['t'] < self.start+100]
        old = extract_event_features(self.events)[0][:len(current)]
        new = base_features(current, 0, self.start)
        columns = [0, 1, 3, *range(4, 28)]
        np.testing.assert_array_equal(new[:, columns], old[:, columns])
        fine = np.zeros((1, 2, 68, 88), dtype=np.int32)
        np.add.at(fine[0], (current['p'], current['y']//4, current['x']//4), 1)
        np.testing.assert_array_equal(_fine_counts_to_patch_features(fine),
                                      build_patch_features(self.events, [0]))

    def test_variants_only_change_column_two_and_initial_outputs_identical(self):
        outputs = []
        for variant in ('C0-zero', 'C1-proposal-motion', 'C2-LMEC'):
            p = self.pipeline(variant)
            p.push(self.two, self.start+200)
            outputs.append(p.flush_completed_windows())
        for window in (0, 1):
            a, b, c = [item[window] for item in outputs]
            self.assertEqual(a.features.shape[1], 36)
            self.assertEqual(a.features.dtype, np.uint8)
            self.assertEqual(len(a.candidate_mask), len(a.event_indices))
            self.assertEqual(len(a.probabilities), len(a.event_indices))
            for other in (b, c):
                np.testing.assert_array_equal(a.features[:, [*range(2), *range(3, 36)]],
                                              other.features[:, [*range(2), *range(3, 36)]])
                np.testing.assert_array_equal(a.probabilities, other.probabilities)

    def test_cross_window_old_temporal_bins(self):
        p = self.pipeline('C0-zero')
        p.push(self.two,self.start+200)
        second = p.flush_completed_windows()[1]
        old = extract_event_features(self.events)[0]
        selected = second.event_indices[second.candidate_mask]
        columns = [0,1,3,*range(4,28)]
        self.assertGreater(len(selected),0)
        np.testing.assert_array_equal(second.features[:,columns],old[selected][:,columns])

    def test_motion_boundaries_and_missing_bins(self):
        image1 = np.zeros((34,44),dtype=np.float32)
        image2 = np.zeros_like(image1)
        image1[4,5] = 0.8
        image2[4,4] = 0.6
        self.assertEqual(proposal_motion(np.array([6*8]),np.array([4*8]),[image2,image1])[0],
                         np.rint(255*0.6))
        self.assertEqual(proposal_motion(np.array([0]),np.array([0]),[])[0],0)
        events = np.array([(16,16,0,1),(16,16,25,1),(16,16,50,1)],dtype=EVENT_DTYPE)
        self.assertEqual(lmec(events,np.array([0,1]),0).tolist(),[0,0])
        self.assertGreater(int(lmec(events,np.array([2]),0)[0]),0)

    def test_chunking_future_and_late_packet(self):
        one = self.pipeline()
        one.push(self.two, self.start+200)
        whole = one.flush_completed_windows()
        two = self.pipeline()
        pieces = np.array_split(self.two, 9)
        for piece in pieces[:-1]:
            two.push(piece, float(piece['t'][-1])+0.00001)
        two.push(pieces[-1], self.start+200)
        chunked = two.flush_completed_windows()
        for a, b in zip(whole, chunked):
            np.testing.assert_array_equal(a.event_indices, b.event_indices)
            np.testing.assert_array_equal(a.features, b.features)
            np.testing.assert_array_equal(a.probabilities, b.probabilities)
        more = self.events[(self.events['t'] >= self.start+200) & (self.events['t'] < self.start+300)]
        one.push(more, self.start+300)
        np.testing.assert_array_equal(whole[0].probabilities, chunked[0].probabilities)
        frozen = whole[0].probabilities.copy()
        late_future = np.array([(12,12,self.start+8050,1)],dtype=EVENT_DTYPE)
        one.push(late_future,self.start+8100)
        np.testing.assert_array_equal(whole[0].probabilities,frozen)
        with self.assertRaises(ValueError):
            one.push(self.two[:1], self.start+8200)

    def test_boundary_empty_windows_and_after_eight_seconds(self):
        p = self.pipeline('C0-zero')
        sparse = np.array([(4, 4, self.start+99.999, 1),
                           (4, 4, self.start+100.0, 1),
                           (4, 4, self.start+8050.0, 1)], dtype=EVENT_DTYPE)
        p.push(sparse, self.start+8100)
        rows = p.flush_completed_windows()
        self.assertEqual(len(rows), 81)
        self.assertEqual([len(rows[i].event_indices) for i in (0, 1, 2, 80)], [1, 1, 0, 1])
        self.assertEqual(rows[80].event_indices.tolist(), [2])
        self.assertEqual(len(p.previous_heatmaps), 2)
        self.assertEqual(len(p.previous_events), 1)

    def test_partial_is_not_published_and_labels_are_ignored(self):
        p = self.pipeline()
        subset = self.two[self.two['t'] < self.start+75].copy()
        p.push(subset, self.start+75)
        self.assertEqual(p.flush_completed_windows(), [])
        partial = self.pipeline()
        partial.push(subset,self.start+75)
        rows, leftover = partial.finalize()
        self.assertEqual(rows,[])
        self.assertEqual(leftover, len(subset))
        with self.assertRaises(ValueError):
            partial.push(np.empty(0,dtype=EVENT_DTYPE),self.start+100)
        p.push(np.empty(0, dtype=EVENT_DTYPE), self.start+100)
        first = p.flush_completed_windows()[0]
        subset['label'] = 1-subset['label']
        subset['name'] = 7
        q = self.pipeline()
        q.push(subset, self.start+100)
        second = q.flush_completed_windows()[0]
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.probabilities, second.probabilities)


if __name__ == '__main__':
    unittest.main()
