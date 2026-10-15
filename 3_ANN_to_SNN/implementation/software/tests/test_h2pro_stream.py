"""Run frozen ANN stream contract checks in an isolated interpreter."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).absolute().parents[2]
ANN = ROOT.parents[1] / '1_H2_model_improvement/implementation'

CHECK = r'''
import json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1])
sys.path.insert(0,str(root/'software'))
from wangdao.h2_pro.stream_state import CausalPipeline,EVENT_DTYPE

def pipeline():
    return CausalPipeline(root/'software/weights/H2_D1_seed42.npz',
        root/'software/weights/H2-pro-Causal-ANN-v1_int8.npz',
        variant='C1-proposal-motion',threshold=.35,initial_refiner=False)

events=np.zeros(1600,dtype=EVENT_DTYPE)
events['t']=np.linspace(7.,356.,len(events))
events['x']=70+np.arange(len(events))%6
events['y']=90+np.arange(len(events))%4
events['p']=np.arange(len(events))%2
one=pipeline(); one.reset(7.); one.push(events,357.)
expected,tail=one.finalize()
chunk=pipeline(); chunk.reset(7.)
start=0; observed=[]
for end in [37.,107.,137.,207.,307.,357.]:
    stop=int(np.searchsorted(events['t'],end))
    chunk.push(events[start:stop],end); observed.extend(chunk.flush_completed_windows()); start=stop
last,tail2=chunk.finalize(); observed.extend(last)
assert len(expected)==len(observed)==3 and tail==tail2
for a,b in zip(expected,observed):
    for field in ['event_indices','probabilities','predictions','candidate_mask','features','heatmap']:
        np.testing.assert_array_equal(getattr(a,field),getattr(b,field))
altered=events.copy(); altered['x'][altered['t']>=107.]=300
future=pipeline(); future.reset(7.); future.push(altered,357.)
future_windows,_=future.finalize()
np.testing.assert_array_equal(expected[0].probabilities,future_windows[0].probabilities)
np.testing.assert_array_equal(expected[0].features,future_windows[0].features)
all_ids=np.concatenate([w.event_indices for w in observed])
np.testing.assert_array_equal(all_ids,np.arange(len(all_ids)))
assert len(all_ids)+tail==len(events)
chunk.reset(7.); chunk.push(events,357.); reset_windows,_=chunk.finalize()
for a,b in zip(expected,reset_windows):
    np.testing.assert_array_equal(a.probabilities,b.probabilities)
print(json.dumps({'chunk_invariance':True,'prefix_causality':True,'reset':True,
                  'event_partition':True,'closed_windows':len(expected),'tail_events':tail}))
'''


class FrozenStreamTests(unittest.TestCase):
    def test_isolated_actual_ann_chunk_prefix_reset_and_tail(self):
        completed = subprocess.run([sys.executable, '-B', '-c', CHECK, str(ANN)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   encoding='utf-8', timeout=60)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        result = json.loads(completed.stdout)
        for key in ('chunk_invariance', 'prefix_causality', 'reset', 'event_partition'):
            self.assertTrue(result[key])
        self.assertGreater(result['tail_events'], 0)


if __name__ == '__main__':
    unittest.main()
