"""One frozen Test evaluation: Dev-selected adaptive SNN, fixed T16, and ANN.

No optimizer, fitting, threshold search, or model-selection entrypoint. Test
access is separate from (and does not relax) the development-only entrypoints.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path, PurePosixPath
import json
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT.parents[1] / '1_H2_model_improvement/implementation'
sys.path.insert(0, str(ANN / 'software'))
import wangdao
wangdao.__path__.append(str(ROOT / 'software/wangdao'))
from wangdao.data import load_events
from wangdao.h2pro_snn.foundation import verify_baseline, development_index, sha256, read_json
from wangdao.h2pro_snn.diagnostics import error_counts
from wangdao.h2pro_direct_v1.common import lock_directory, output_path
from wangdao.h2pro_direct_v1.core import load_layers, forward as ann_forward
from wangdao.h2pro_proposal_transfer_v1.pipeline import TransferPipeline, stream_state
from wangdao.h2pro_r1_train_v1.evaluation import load_export
from wangdao.h2pro_calibration_v1.core import Encoder, forward, probabilities
from wangdao.h2pro_adaptive_t_v1.common import atomic_json, atomic_npz, sealed_json, SOURCE, SOURCE_SHA
from wangdao.h2pro_adaptive_t_v1.core import trace_reference
from wangdao.h2pro_adaptive_t_v1.selection import summarize_counts
from wangdao.h2pro_adaptive_t_v2.core import Policy, incremental, choose, ledger
from wangdao.h2pro_adaptive_t_v2.scoring import error_changes

DEFAULT_RUN = ROOT / 'results/runs/h2pro_frozen_test_v1_ep159_20261016'
DEFAULT_DATA = ROOT / 'data/EV-UAV-dataset'
PRIOR = ROOT / 'results/runs/h2pro_adaptive_t_v2_alternatives_s42_ep159'
DEV_REFERENCE = ROOT / 'results/runs/h2pro_adaptive_t_v1_s42_ep159/comparison.json'
POLICY = Policy(.01, .99, .10, .95)
MODELS = ('h2pro_ann', 'r1_t16_ep159', 'adaptive_ep159')
PROTOCOL = dict(
    version='h2pro_frozen_test_v1', data_role='test', expected_sequences=24,
    models=list(MODELS), primary_model='adaptive_ep159', selection_metric='existing Train-dev overall Event IoU',
    selection_reason='Dev IoU 0.7972939729397294; equal-IoU negative-only policy has higher AC',
    source_epoch=159, exit_policy=POLICY.as_dict(), final_threshold=.35, steps=[4, 8, 16], theta=255, m0=191,
    input_window_ms=100, time_origin='first event', tail='discard final incomplete 100 ms window, report coverage',
    high_density='raw sequence events * 1000 / duration_ms >= 20000; same definition as development',
    proposal='frozen H2-pro ANN, exact zero-skip replay checked against dense original',
    refiner='unchanged 36-64-32-4, epoch159 int8 export; only 64->32 spike transport',
    backend='CPU: full actual binary spike-add for fixed/adaptive SNN; dense prefix reference for verification only',
    historical_teacher_test_exposure=True, test_used_for_this_selection=False,
    threshold_search=False, training_started=False, weight_updates=0,
    deployment_cost_scope='neural synapses plus separately reported IF/readout/control scalars; audit computations excluded',
    not_measured=['FPGA latency', 'energy', 'memory traffic', 'feature construction operations'],
)


def test_records(index):
    """Only the 24 explicitly indexed Test records; no role override or limit."""
    names = index['roles']['test']
    if len(names) != 24 or len(set(names)) != 24:
        raise ValueError('Require all 24 Test sequences')
    by_path = {r['path']: r for r in index['files']}
    result = []
    for name in names:
        if PurePosixPath(name).name != name or '\\' in name or not name.startswith('test_') or not name.endswith('.npz'):
            raise ValueError('Unsafe Test filename')
        result.append(by_path['test/' + name])
    return result


def identity(dataset):
    baseline = verify_baseline()
    index = development_index()  # index metadata only, not event/label loading
    if sha256(SOURCE / 'epochs/epoch_159.pt') != SOURCE_SHA:
        raise ValueError('The frozen epoch159 checkpoint changed')
    layers = load_export(SOURCE / 'epochs/epoch_159_int8.npz')
    selection = read_json(PRIOR / 'selection.json')
    if selection['selected']['accuracy']['policy'] != POLICY.as_dict():
        raise ValueError('Chosen policy differs from the existing Main-sealed candidate')
    prior = read_json(PRIOR / 'comparison.json')
    if prior['adaptive']['accuracy']['overall']['iou'] != 0.7972939729397294:
        raise ValueError('Prior Dev selection evidence changed')
    paths = [Path(__file__), ROOT / 'software/tests/test_h2pro_frozen_test.py',
             ROOT / 'evaluate_h2pro_frozen_test.ps1', DEV_REFERENCE]
    paths += [SOURCE / 'epochs' / ('epoch_159' + s) for s in ('.pt', '.json', '_int8.npz')]
    paths += [PRIOR / s for s in ('identity.json', 'selection.json', 'main_seal.json', 'comparison.json')]
    paths += sorted((ROOT / 'software/wangdao').rglob('*.py'))
    sources = {str(p.resolve()): sha256(p) for p in paths}
    return dict(protocol=PROTOCOL, baseline=baseline, dataset_root=str(Path(dataset).resolve()),
                test_records=test_records(index), dataset_fingerprint=index['dataset_fingerprint'],
                source_sha256=sources, numpy=np.__version__, torch=torch.__version__), layers


def protected_files():
    paths = set((ROOT / 'results').glob('*.json'))
    for name in ('h2pro_r1t16_train_v1_s42', 'h2pro_adaptive_t_v1_s42_ep159',
                 'h2pro_adaptive_t_v2_s42_ep159', 'h2pro_adaptive_t_v2_alternatives_s42_ep159',
                 'h2pro_adaptive_t_v3_s42_ep159'):
        paths.update(p for p in (ROOT / 'results/runs' / name).rglob('*') if p.is_file())
    return {str(p.resolve()): sha256(p) for p in sorted(paths)}


def verify_hashes(mapping):
    for path, expected in mapping.items():
        if not Path(path).is_file() or sha256(path) != expected:
            raise ValueError('Protected file changed: ' + path)


def require_seal(directory, ident):
    seal = read_json(directory / 'pretest_seal.json')
    if (seal['identity_sha256'] != sha256(directory / 'identity.json')
            or seal['preservation_sha256'] != sha256(directory / 'preservation_before.json')
            or read_json(directory / 'identity.json') != ident):
        raise ValueError('Frozen Test seal changed')
    return seal


def features_from_events(events):
    """Keep labels outside the unmodified ANN pipeline; replay its exact features."""
    pipe = TransferPipeline('ann_sparse_proposal')
    packet = np.empty(len(events), dtype=stream_state.EVENT_DTYPE)
    for key in ('x', 'y', 't', 'p'):
        packet[key] = events[key]
    t0 = float(events['t'][0])
    pipe.reset(t0)
    pipe.push(packet, float(events['t'][-1]) + 1e-6)
    windows, trailing = pipe.finalize()
    ids = np.concatenate([w.event_indices for w in windows]) if windows else np.empty(0, np.int64)
    np.testing.assert_array_equal(ids, np.arange(len(events) - trailing))
    candidate = np.concatenate([w.candidate_mask for w in windows]) if windows else np.empty(0, bool)
    features = np.concatenate([w.features for w in windows]) if windows else np.empty((0, 36), np.uint8)
    teacher = np.concatenate([w.probabilities for w in windows]) if windows else np.empty(0, np.float32)
    raw, _ = ann_forward(features, load_layers())
    np.testing.assert_array_equal(probabilities(raw), teacher[candidate])
    elapsed = float(events['t'][-1]) - t0
    arrays = dict(features=features, candidate=candidate, labels=events['label'][ids].astype(np.uint8), teacher=raw)
    meta = dict(raw_events=len(events), events=len(ids), target_events=int(arrays['labels'].sum()),
                candidate_events=int(candidate.sum()), candidate_targets=int(arrays['labels'][candidate].sum()),
                raw_target_events=int(events['label'].sum()), trailing_events=trailing,
                trailing_target_events=int(events['label'][len(ids):].sum()), closed_windows=len(windows),
                high_density=bool(len(events)*1000/elapsed >= 20000) if elapsed > 0 else False,
                duration_ms=elapsed, proposal_sparse_mac=pipe.executor.cost.proposal_mac,
                proposal_dense_mac=pipe.executor.cost.proposal_dense_equivalent_mac,
                ann_execution=asdict(pipe.executor.cost))
    return arrays, meta


def infer(features, layers, batch_size=8192):
    prefix, spike_prefix = trace_reference(features, layers, 'cpu', batch_size)
    raw, spikes, exits, measured = incremental(features, layers, POLICY, batch_size)
    expected = choose(np.stack([probabilities(y) for y in prefix]), POLICY)
    ids = np.arange(len(features))
    np.testing.assert_array_equal(exits, expected)
    np.testing.assert_array_equal(raw, prefix[exits, ids])
    np.testing.assert_array_equal(spikes, spike_prefix[exits, ids])
    cost = ledger(exits, spikes, POLICY)
    if any(cost[k] != v for k, v in measured.items()):
        raise AssertionError('Actual adaptive arithmetic differs from cost ledger')
    fixed_ac = fixed_mac = 0
    for lo in range(0, len(features), batch_size):
        y, c = forward(features[lo:lo+batch_size], layers, 'r1', 16,
                       (Encoder(255, 191), Encoder()), 'spike_add')
        np.testing.assert_array_equal(y, prefix[2, lo:lo+len(y)])
        fixed_ac += c.executed_synaptic_ac
        fixed_mac += c.executed_synaptic_mac
    if fixed_ac != int(spike_prefix[2].sum())*32 or fixed_mac != len(features)*2432:
        raise AssertionError('Full fixed T16 arithmetic differs from cost ledger')
    return dict(fixed_raw=prefix[2], fixed_spikes=spike_prefix[2], adaptive_raw=raw,
                adaptive_spikes=spikes, exit_index=exits), dict(
                    adaptive_measured=measured, fixed_measured_mac=fixed_mac, fixed_measured_ac=fixed_ac,
                    all_candidates_actual_spike_verified=len(features), prefix_reference_only=True)


def count_models(arrays):
    records = {}
    for model, key in zip(MODELS, ('teacher', 'fixed_raw', 'adaptive_raw')):
        p = np.zeros(len(arrays['labels']), np.float32)
        p[arrays['candidate']] = probabilities(arrays[key])
        records[model] = error_counts(arrays['labels'], p, arrays['candidate'], .35)
    return records


def summarize(records, arrays):
    rows = [r['metadata'] for r in records]
    n = sum(r['candidate_events'] for r in rows)
    prop = sum(r['proposal_sparse_mac'] for r in rows)
    dense = sum(r['proposal_dense_mac'] for r in rows)
    ix = np.concatenate([a['exit_index'] for a in arrays])
    sn = np.concatenate([a['adaptive_spikes'] for a in arrays])
    fn = np.concatenate([a['fixed_spikes'] for a in arrays])
    costs = {MODELS[1]: ledger(np.full(n, 2, np.uint8), fn, fixed=True),
             MODELS[2]: ledger(ix, sn, POLICY)}
    for c in costs.values():
        ref = c['input_projection_mac'] + c['readout_mac']
        c.update(proposal_sparse_mac=prop, proposal_dense_mac=dense, refiner_mac=ref,
                 inference_equivalent_mac_sparse_proposal=prop+ref,
                 inference_equivalent_mac_dense_proposal=dense+ref, inference_equivalent_ac=c['synaptic_ac'])
    costs[MODELS[0]] = dict(proposal_sparse_mac=prop, proposal_dense_mac=dense, refiner_mac=n*4480,
        inference_equivalent_mac_sparse_proposal=prop+n*4480,
        inference_equivalent_mac_dense_proposal=dense+n*4480, inference_equivalent_ac=0)
    result = {}
    for model in MODELS:
        result[model] = {group: summarize_counts([r['counts'][model] for r in records
                         if group == 'overall' or r['metadata']['high_density'] == (group == 'high_density')])
                         for group in ('overall', 'ordinary', 'high_density')}
        result[model].update(cost=costs[model], threshold=.35)
    yy = np.concatenate([a['labels'][a['candidate']] for a in arrays])
    pp = np.concatenate([probabilities(a['adaptive_raw']) >= .35 for a in arrays])
    old = np.concatenate([probabilities(a['fixed_raw']) >= .35 for a in arrays])
    return dict(models=result, changes_adaptive_vs_fixed=error_changes(yy, pp, old),
        coverage={k: sum(r[k] for r in rows) for k in ('raw_events', 'events', 'trailing_events',
                  'raw_target_events', 'target_events', 'trailing_target_events', 'candidate_events', 'closed_windows')},
        high_density_sequences=[r['sequence'] for r in rows if r['high_density']])


def run(action, directory=DEFAULT_RUN, dataset=DEFAULT_DATA, batch_size=8192):
    directory, dataset = output_path(directory), Path(dataset).resolve()
    if action == 'status':
        return read_json(directory/'progress.json') if (directory/'progress.json').exists() else dict(stage='not_started')
    if action not in ('seal', 'run') or type(batch_size) is not int or batch_size < 1:
        raise ValueError('Known action and positive batch size required')
    if directory.exists() and not (directory/'identity.json').exists() and any(directory.iterdir()):
        raise ValueError('Require a new independent result directory')
    ident, layers = identity(dataset)
    with lock_directory(directory, ident):
        if not (directory/'pretest_seal.json').exists():
            sealed_json(directory/'preservation_before.json', protected_files())
            sealed_json(directory/'pretest_seal.json', dict(unix_s=time.time(),
                identity_sha256=sha256(directory/'identity.json'),
                preservation_sha256=sha256(directory/'preservation_before.json'),
                user_authorized_test=True, model_selection_complete_before_test_read=True))
        seal = require_seal(directory, ident)
        if action == 'seal':
            return dict(stage='sealed', test_sequences_read=0, directory=str(directory))
        if (directory/'comparison.json').exists():
            done = read_json(directory/'comparison.json')
            verify_hashes({str(directory/k): v for k, v in done['artifact_sha256'].items()})
            verify_hashes(read_json(directory/'preservation_before.json'))
            return dict(stage='complete', reused=True, comparison=str(directory/'comparison.json'))
        records, outputs = [], []
        for j, record in enumerate(ident['test_records']):
            name = Path(record['path']).stem
            target = directory/'sequences'/name
            receipt, payload = target.with_suffix('.json'), target.with_suffix('.npz')
            started = time.perf_counter()
            if receipt.exists():
                rec = read_json(receipt)
                if (rec['source_sha256'] != record['sha256'] or rec['payload_sha256'] != sha256(payload)
                        or rec['seal_sha256'] != sha256(directory/'pretest_seal.json')):
                    raise ValueError('Completed sequence provenance changed')
                with np.load(payload, allow_pickle=False) as z:
                    arrays = {k: z[k] for k in z.files}
            else:
                source = dataset/record['path']
                if sha256(source) != record['sha256']:
                    raise ValueError('Test source hash mismatch: ' + str(source))
                # The identity and seal were durably written before the first Test read.
                read_time = time.time()
                events = load_events(source, require_labels=True)
                if len(events) != record['events'] or int(events['label'].sum()) != record['target_events']:
                    raise ValueError('Indexed raw event coverage mismatch')
                arrays, meta = features_from_events(events)
                inference, verification = infer(arrays['features'], layers, batch_size)
                arrays.update(inference)
                meta.update(sequence=record['path'], role='test')
                rec = dict(metadata=meta, source_sha256=record['sha256'], verification=verification,
                           counts=count_models(arrays), first_test_read_unix_s=read_time,
                           seal_sha256=sha256(directory/'pretest_seal.json'), seconds=time.perf_counter()-started)
                atomic_npz(payload, **arrays)
                rec['payload_sha256'] = sha256(payload)
                atomic_json(receipt, rec)
                del events
            if rec['first_test_read_unix_s'] < seal['unix_s']:
                raise AssertionError('Test read preceded frozen seal')
            if count_models(arrays) != rec['counts']:
                raise AssertionError('Saved predictions do not reproduce counts')
            records.append(rec)
            outputs.append({k: v for k, v in arrays.items() if k != 'features'})
            progress = dict(stage='test', completed=j+1, total=24, sequence=record['path'],
                            events=rec['metadata']['events'], seconds=rec['seconds'], unix_s=time.time())
            atomic_json(directory/'progress.json', progress)
            with (directory/'progress.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(progress)+'\n')
            print(json.dumps(progress), flush=True)
        result = summarize(records, outputs)
        verify_hashes(ident['source_sha256'])
        protection = read_json(directory/'preservation_before.json')
        verify_hashes(protection)
        result.update(status='complete', role='test', protocol=PROTOCOL, sequences=records,
                      source_files_unchanged=True, protected_old_files_unchanged=len(protection),
                      training_started=False, weights_updated=False, test_parameter_search=False,
                      artifact_sha256={p.relative_to(directory).as_posix(): sha256(p) for p in directory.rglob('*')
                                       if p.is_file() and p.suffix in ('.json', '.npz') and p.name != 'progress.json'})
        atomic_json(directory/'comparison.json', result)
        atomic_json(directory/'progress.json', dict(stage='complete', sequences=24, comparison=str(directory/'comparison.json')))
        return dict(stage='complete', reused=False, comparison=str(directory/'comparison.json'), models=result['models'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--action', choices=('seal', 'run', 'status'), default='run')
    parser.add_argument('--run-directory', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--dataset-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(run(args.action, args.run_directory, args.dataset_root, args.batch_size), indent=2))


if __name__ == '__main__':
    main()
