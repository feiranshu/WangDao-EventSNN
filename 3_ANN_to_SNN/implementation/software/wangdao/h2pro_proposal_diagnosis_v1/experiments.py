"""Single-factor interventions and explicitly priced deployable candidates."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..h2pro_proposal_transfer_v1.pipeline import TransferPipeline
from ..h2pro_proposal_transfer_v1.workflow import metrics, transition
from ..h2pro_snn.diagnostics import error_counts
from .common import load_bundle, save_bundle
from .cache import cache_sequence
from .core import (GROUPS, context_features, component_column, event_packet, event_scores,
                   map_heatmaps, refiner, swapped)


def spec(name, route='s1', gate=.51, context='s1', context_gate=.51, swap=None):
    return {'name': name, 'route': route, 'gate': float(gate), 'context': context,
            'context_gate': float(context_gate), 'swap': swap,
            'diagnostic_only': swap is not None or (route == 'ann') != (context == 'ann')}


def infer_spec(data, arrays, meta, configuration, calibration, layers, shared=None):
    c = configuration
    shared = {} if shared is None else shared
    packet = event_packet(arrays)
    windows, base = data['windows'], data['base']

    def heatmaps(mode):
        if mode != 'calibrated':
            return data[mode+'_heatmaps']
        if 'calibrated_heatmaps' not in shared:
            shared['calibrated_heatmaps'] = map_heatmaps(data['s1_heatmaps'], calibration)
        return shared['calibrated_heatmaps']

    def features(mode):
        key = 'features_'+mode
        if key not in shared:
            shared[key] = context_features(packet, windows, base, heatmaps(mode), .51, meta['t0_ms'])
        return shared[key]

    ctx = c['context']
    if c['swap'] is not None:
        f = swapped(features('ann'), features('s1'), c['swap'])
        prob = refiner(f, layers)
        extra_rows = len(f)
    else:
        f = features(ctx)
        if ctx == 'calibrated':
            if 'calibrated_prob' not in shared:
                shared['calibrated_prob'] = refiner(f, layers)
            prob = shared['calibrated_prob'].copy()
            extra_rows = len(f)  # Conservative, includes reusable calibrated inference.
        else:
            prob = data[ctx+'_prob_all'].copy()
            extra_rows = 0
        if c['context_gate'] != .51:
            column = component_column(packet, windows, heatmaps(ctx), c['context_gate'])
            changed = column != f[:, 30]
            subset = f[changed].copy()
            subset[:, 30] = column[changed]
            if len(subset):
                prob[changed] = refiner(subset, layers)
            extra_rows += len(subset)
    route = c['route']
    scores = (event_scores(packet, windows, heatmaps(route)) if route == 'calibrated'
              else data[route+'_scores'])
    candidate = scores >= c['gate']
    return {'candidate': candidate, 'probabilities': np.where(candidate, prob, 0).astype(np.float32)}, extra_rows


def run_specs(directory, store, role, specifications, calibration):
    rows = {s['name']: [] for s in specifications}
    layers = TransferPipeline('ann_dense').refiner_layers
    for name in store.names(role):
        data, cache, arrays, meta = cache_sequence(directory, store, role, name)
        shared = {}
        baseline = {'candidate': arrays['candidate'], 'probabilities': arrays['teacher']}
        for s in specifications:
            stem = Path(directory)/'points'/role/s['name']/Path(name).stem
            if stem.with_suffix('.json').exists():
                output, row = load_bundle(stem)
                if row['specification'] != s or row['calibration'] != calibration:
                    raise ValueError('Experiment definition changed')
                if row['counts'] != error_counts(arrays['label'], output['probabilities'], output['candidate']):
                    raise ValueError('Saved prediction/metric mismatch')
            else:
                output, extra = infer_spec(data, arrays, meta, s, calibration, layers, shared)
                counts = error_counts(arrays['label'], output['probabilities'], output['candidate'])
                needed = {'ann' if v == 'ann' else 's1' for v in (s['route'], s['context'])}
                if s['swap'] is not None:
                    needed = {'ann', 's1'}
                prop_mac = sum(cache['costs'][m]['proposal_mac'] for m in needed)
                ac = sum(cache['costs'][m]['proposal_ac'] for m in needed)
                row = {'sequence': name, 'role': role, 'high_density': meta['high_density'],
                       'specification': s, 'calibration': calibration, 'counts': counts,
                       'proposal_mac': prop_mac, 'refiner_mac': counts['candidate_events']*4480,
                       'ac': ac, 'required_proposals': sorted(needed),
                       'extra_diagnostic_refiner_mac': extra*4480,
                       'transition': transition(arrays['label'], output, baseline)}
                save_bundle(stem, output, row)
            rows[s['name']].append(row)
        print({'stage': 'experiments', 'role': role, 'sequence': name, 'configurations': len(specifications)}, flush=True)
    results = []
    for s in specifications:
        r = rows[s['name']]
        total = metrics([v['counts'] for v in r])
        results.append({'name': s['name'], 'specification': s, 'diagnostic_only': s['diagnostic_only'],
                        'overall': total, 'high_density': metrics([v['counts'] for v in r if v['high_density']]),
                        'ordinary': metrics([v['counts'] for v in r if not v['high_density']]),
                        'mac': sum(v['proposal_mac']+v['refiner_mac'] for v in r),
                        'ac': sum(v['ac'] for v in r), 'sequences': r,
                        'transition': {k: sum(v['transition'][k] for v in r) for k in r[0]['transition']}})
    return results


def acceptance(result, baseline):
    accuracy = all(result['overall'][k] >= baseline['overall'][k]
                   for k in ('iou', 'precision', 'recall'))
    accuracy = accuracy and result['high_density']['recall'] >= baseline['high_density']['recall']
    return {'accuracy_preserved': bool(accuracy), 'mac_below_sparse_ann': result['mac'] < baseline['mac'],
            'mac_plus_ac_below_sparse_ann': result['mac']+result['ac'] < baseline['mac'],
            'deployable': not result['diagnostic_only'],
            'passes': bool(accuracy and result['mac'] < baseline['mac'] and not result['diagnostic_only'])}
