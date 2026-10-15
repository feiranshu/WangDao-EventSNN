"""Audit frozen FRED predictions against source shards and summarize paired errors."""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np

import evaluate_fred_frozen as ev


def review(dataset, output):
    report = ev.read_json(output / 'comparison.json')
    index = ev.read_json(dataset / 'index.json')
    if ev.sha256(dataset / 'index.json') != report['identity']['dataset_index_sha256']:
        raise ValueError('Source dataset identity changed')
    if len(report['sequences']) != len(index['sequences']) or report['input_events'] != index['events']:
        raise AssertionError('Incomplete source-recording coverage')
    if ev.aggregate(report['sequences'], report['identity']) != report:
        raise AssertionError('Published aggregate differs from the sequence reports')
    pair_keys = ('ann_tp_lost', 'ann_fn_recovered', 'ann_fp_removed', 'new_fp_vs_ann',
                 'all_valid_decision_disagreements')
    paired = {m: dict.fromkeys(pair_keys, 0) for m in ev.MODELS[1:]}
    present = {m: ev.zero_counts() for m in ev.MODELS}
    absent = {m: ev.zero_counts() for m in ev.MODELS}
    audits = []
    for seq, result in zip(index['sequences'], report['sequences']):
        if seq['source_sequence'] != result['sequence']:
            raise AssertionError('Sequence ordering changed')
        directory = output / result['sequence']
        if ev.sha256(directory / 'predictions.npy') != result['predictions_sha256']:
            raise AssertionError('Saved predictions changed')
        if ev.sha256(directory / 'windows.jsonl') != result['windows_sha256']:
            raise AssertionError('Saved window ledger changed')
        bits = np.load(directory / 'predictions.npy', mmap_mode='r', allow_pickle=False)
        closed_boundary = result['t0_ms'] + result['closed_windows']*100.
        offset = 0
        for shard in seq['shards']:
            source = dataset / shard['path']
            if ev.sha256(source) != shard['sha256']:
                raise AssertionError('Source shard changed')
            with np.load(source, allow_pickle=False) as z:
                events, valid = z['ev'], z['eval_mask']
            for lo in range(0, len(events), 1_000_000):
                block = events[lo:lo+1_000_000]
                b = bits[offset+lo:offset+lo+len(block)]
                truth = (block['label'].astype(np.uint8)*32 + valid[lo:lo+len(block)].astype(np.uint8)*16 +
                         (block['t'] < 5000.).astype(np.uint8)*128)
                np.testing.assert_array_equal(b & 176, truth)
                np.testing.assert_array_equal((b & 64) != 0, block['t'] < closed_boundary)
                mask = ((b & 16) != 0) & ((b & 64) != 0)
                y, ann = (b & 32) != 0, (b & 2) != 0
                for mi, model in enumerate(ev.MODELS[1:], start=1):
                    p = (b & (2 << mi)) != 0
                    values = (y & ann & ~p, y & ~ann & p, ~y & ann & ~p, ~y & ~ann & p, ann != p)
                    for key, value in zip(pair_keys, values):
                        paired[model][key] += int(np.count_nonzero(mask & value))
            offset += len(events)
            del events, valid
        if offset != len(bits):
            raise AssertionError('Source coverage differs from saved predictions')
        event_offset = window_count = candidate_count = 0
        cost_sums = {m: {} for m in ev.MODELS}
        prop_sums = dict(dense_mac=0, zero_skip_equivalent_mac=0, output_scalars=0)
        counts_sums = ev.new_totals()
        with (directory / 'windows.jsonl').open('r', encoding='utf-8') as stream:
            for line in stream:
                row = json.loads(line)
                if row['window'] != window_count or row['first_event_index'] != event_offset:
                    raise AssertionError('Window/event sequence discontinuity')
                if row['right_boundary_ms'] != result['t0_ms']+(window_count+1)*100.:
                    raise AssertionError('Window clock changed')
                n = row['candidate_events']
                for model in ev.MODELS:
                    target = present if row['counts'][model]['all']['target_events'] else absent
                    ev.add_counts(target[model], row['counts'][model]['all'])
                    for segment in counts_sums[model]:
                        ev.add_counts(counts_sums[model][segment], row['counts'][model][segment])
                    cost = row['refiner_cost'][model]
                    for key, value in cost.items():
                        cost_sums[model][key] = cost_sums[model].get(key, 0) + value
                    mac_per_candidate = {ev.MODELS[0]:4480, ev.MODELS[1]:2432, ev.MODELS[2]:2304}[model]
                    if cost['rate_network_mac'] != n*mac_per_candidate:
                        raise AssertionError('Window Refiner MAC formula mismatch')
                    ac, spikes = cost['rate_network_ac'], cost['unit_spikes']
                    if model == ev.MODELS[1] and ac != 32*spikes:
                        raise AssertionError('r1 AC fanout mismatch')
                    if model == ev.MODELS[2]:
                        first = (ac-4*spikes)//28
                        second = spikes-first
                        if 32*first+4*second != ac or not (0<=first<=n*64*8 and 0<=second<=n*32*8):
                            raise AssertionError('r2 layer spike/fanout identity mismatch')
                for key, value in row['proposal'].items():
                    prop_sums[key] += value
                candidate_count += n
                event_offset += row['events']
                window_count += 1
        if event_offset != result['closed_events'] or window_count != result['closed_windows'] or candidate_count != result['candidate_events']:
            raise AssertionError('Window manifest totals mismatch')
        for model in ev.MODELS:
            for segment in counts_sums[model]:
                if ev.summarize(counts_sums[model][segment]) != result['models'][model]['metrics'][segment]:
                    raise AssertionError('Window metric totals differ')
            if cost_sums[model] != result['models'][model]['cost']['refiner_detail']:
                raise AssertionError('Window cost totals differ')
            if ev.cost_report(cost_sums[model],prop_sums,model,candidate_count) != result['models'][model]['cost']:
                raise AssertionError('Proposal/Refiner ledger differs')
        audits.append({'sequence': result['sequence'], 'source_events_checked': offset,
            'labels_masks_times_exact': True, 'continuous_windows_checked': window_count,
            'cost_ledgers_exact': True})
    baseline = report['models'][ev.MODELS[0]]
    summary = {}
    for model in ev.MODELS:
        m = report['models'][model]
        a = m['metrics']['all']
        c, bc = m['cost'], baseline['cost']
        summary[model] = {
            'delta_to_ann_pp': {metric: 100*(a[metric]-baseline['metrics']['all'][metric])
                for metric in ('iou','precision','recall','proposal_recall')},
            'paired_errors_vs_ann': paired.get(model),
            'opening_share_of_fp': m['metrics']['opening_5s']['fp']/a['fp'] if a['fp'] else None,
            'proposal_share_of_fn': a['proposal_fn']/a['fn'] if a['fn'] else None,
            'target_present_windows': ev.summarize(present[model]),
            'target_absent_windows': ev.summarize(absent[model]),
            'mac_reduction_dense_proposal': 1-c['neural_mac_dense_proposal']/bc['neural_mac_dense_proposal'],
            'mac_reduction_zero_skip_proposal': 1-c['neural_mac_zero_skip_proposal']/bc['neural_mac_zero_skip_proposal'],
            'unweighted_mac_ac_ratio_dense_proposal': (c['neural_mac_dense_proposal']+c['neural_ac'])/bc['neural_mac_dense_proposal'],
            'unweighted_mac_ac_ratio_zero_skip_proposal': (c['neural_mac_zero_skip_proposal']+c['neural_ac'])/bc['neural_mac_zero_skip_proposal'],
        }
        if model in paired:
            pair = paired[model]
            if a['tp'] != baseline['metrics']['all']['tp']-pair['ann_tp_lost']+pair['ann_fn_recovered']:
                raise AssertionError('Paired TP changes do not reconcile')
            if a['fp'] != baseline['metrics']['all']['fp']-pair['ann_fp_removed']+pair['new_fp_vs_ann']:
                raise AssertionError('Paired FP changes do not reconcile')
    result = {'status':'passed', 'source_audit':audits, 'model_analysis':summary,
        'comparison_sha256': ev.sha256(output/'comparison.json'),
        'reviewer_sha256':ev.sha256(Path(__file__)), 'training_started':False,
        'analysis_changes_no_predictions_or_parameters':True}
    ev.atomic_json(output/'analysis.json',result)
    ev.preservation(output)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ev.DATASET)
    parser.add_argument('--output',type=Path,default=ev.OUTPUT)
    args=parser.parse_args()
    review(args.dataset,args.output)
