"""Train-main-only software timing and state measurement for all three variants."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import numpy as np

from run_stage1 import ROOT, _role_files, PROPOSAL, SOURCE_INT8, sha256
from src.stream_state import CausalPipeline, VARIANTS


def run(output: Path, limit: int, names: tuple[str,...] = ()):
    if output.exists():
        raise FileExistsError(output)
    report = {'role':'train_main','sequences':list(names) if names else f'first {limit}',
              'variants':{},'status':'Python CPU timing, not hardware PPA'}
    for variant in VARIANTS:
        times, candidates, state, events_count = [], [], [], 0
        for name,path in _role_files('train_main',0 if names else limit):
            if names and name not in names:
                continue
            with np.load(path,allow_pickle=False) as archive:
                events = archive['ev'].copy()
            t0 = float(events['t'][0])
            windows = int((float(events['t'][-1])-t0)//100)
            pipe = CausalPipeline(PROPOSAL,SOURCE_INT8,variant)
            pipe.reset(t0)
            for w in range(windows):
                left,right = t0+w*100,t0+(w+1)*100
                first = int(np.searchsorted(events['t'],left,side='left'))
                last = int(np.searchsorted(events['t'],right,side='left'))
                pipe.push(events[first:last],right)
                result = pipe.flush_completed_windows()[0]
                times.append(result.computation_s)
                candidates.append(int(result.candidate_mask.sum()))
                events_count += len(result.event_indices)
                state.append(pipe.previous_events.nbytes+
                             sum(x.nbytes for x in pipe.previous_heatmaps)+pipe.pending.nbytes)
        report['variants'][variant] = {
            'complete_windows':len(times),'events':events_count,
            'mean_compute_ms':float(1000*np.mean(times)) if times else None,
            'p95_compute_ms':float(1000*np.quantile(times,0.95)) if times else None,
            'max_compute_ms':float(1000*np.max(times)) if times else None,
            'mean_candidates_per_window':float(np.mean(candidates)) if candidates else None,
            'max_candidates_per_window':int(max(candidates)) if candidates else None,
            'peak_retained_numpy_state_bytes':int(max(state)) if state else 0,
        }
    report['architectural_cost'] = {
        'refiner_weights':4480,'dense_mac_per_candidate':4480,
        'two_float32_proposal_heatmaps_bytes':2*34*44*4,
        'C2_max_needed_bin_maps_per_window':5,
        'C2_one_int32_count_map_bytes':65*87*4,
        'C2_one_float32_quality_map_bytes':65*87*4,
        'C2_velocity_hypotheses':9,
        'note':'C2 current NumPy implementation materializes up to five count and quality maps. '
               'The previous raw 100ms event buffer has a bounded time horizon but no fixed '
               'event-count cap. Hardware SRAM, overflow and PPA remain unmeasured.'}
    output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=ROOT/'reports'/'cpu_benchmark.json')
    p.add_argument('--limit',type=int,default=2)
    p.add_argument('--names',nargs='*',default=[])
    a=p.parse_args()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    print(json.dumps(run(a.output,a.limit,tuple(a.names))))
