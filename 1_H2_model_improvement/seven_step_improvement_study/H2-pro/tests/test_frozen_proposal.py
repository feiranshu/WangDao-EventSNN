import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).absolute().parent.parent
SOFTWARE = ROOT.parents[1]/'implementation'/'software'
sys.path[:0] = [str(ROOT),str(SOFTWARE)]
from src.stream_state import CausalPipeline


class FrozenProposalTests(unittest.TestCase):
    def test_closed_windows_match_legacy_heatmaps_and_routing(self):
        try:
            import torch  # noqa: F401 - h2_research needs torch
        except ImportError:
            self.skipTest('PyTorch is not installed in this interpreter')
        from wangdao.h2_research import FrozenH2Pipeline
        with np.load(ROOT.parents[1]/'implementation'/'data'/'EV-UAV-dataset'/'train'/'train_000.npz',
                     allow_pickle=False) as archive:
            events = archive['ev'].copy()
        proposal = SOFTWARE/'weights'/'H2_D1_seed42.npz'
        refiner = ROOT.parent/'step7'/'intermediate_H2-plus'/'weights'/'A4-2_seed42_epoch008_int8.npz'
        old = FrozenH2Pipeline(proposal).prepare(events,windows=[0,1])
        t0 = float(events['t'][0])
        new = CausalPipeline(proposal,refiner)
        new.reset(t0)
        new.push(events[events['t']<t0+200],t0+200)
        rows = new.flush_completed_windows()
        for i in (0,1):
            np.testing.assert_array_equal(old.heatmaps[i],rows[i].heatmap)
        indices = np.concatenate([row.event_indices[row.candidate_mask] for row in rows])
        np.testing.assert_array_equal(old.indices,indices)


if __name__ == '__main__':
    unittest.main()
