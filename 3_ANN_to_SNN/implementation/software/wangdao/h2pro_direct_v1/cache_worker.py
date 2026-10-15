"""Read-only frozen ANN execution; cache its original 36-dimensional features."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import time

import numpy as np

from ..h2pro_snn.foundation import ANN_ROOT, read_json, role_records, sha256, verified_sequence
from .common import atomic_json, base_identity, lock_directory, sources
from .core import forward, load_layers, probabilities


def prepare(dataset_root, output, teacher_cache):
    stream_state = import_module('wangdao.h2_pro.stream_state')
    from wangdao.h2 import representation
    from wangdao.data import load_events
    expected = ANN_ROOT/'software/wangdao'
    if any(Path(str(m.__file__)).absolute().parent.parent != expected for m in (stream_state,representation)):
        raise RuntimeError('Run the isolated frozen ANN worker, not the historical H2 package')
    identity = base_identity('original_feature_cache')
    identity['role'] = 'train_dev'
    original_teacher = read_json(Path(teacher_cache)/'manifest.json')
    if (original_teacher['baseline_lock_sha256'] != identity['baseline_lock_sha256']
            or not original_teacher['complete_development_roles'] or original_teacher['max_windows'] is not None):
        raise ValueError('Require the complete existing frozen teacher cache for parity')
    identity['teacher_manifest_sha256'] = sha256(Path(teacher_cache)/'manifest.json')
    teacher_rows = {r['sequence']:r for r in original_teacher['roles']['train_dev']}
    layers = load_layers()
    original = stream_state.quantized_mlp
    with lock_directory(output, identity) as directory:
        rows = []
        for record in role_records('train_dev'):
            target = directory/Path(record['path']).name
            receipt = target.with_suffix('.json')
            if receipt.exists():
                row = read_json(receipt)
                if (row['source_sha256'] != record['sha256'] or row['sequence'] != record['path']
                        or row['cache_sha256'] != sha256(target)):
                    raise ValueError('Existing feature cache mismatch')
                rows.append(row)
                continue
            started = time.perf_counter()
            events = load_events(verified_sequence(Path(dataset_root),record), require_labels=True)
            captured = []
            proposal_mac = 0
            def measured(features, weights):
                nonlocal proposal_mac
                raw = original(features, weights)
                if features.shape[1] == 128:
                    proposal_mac += len(features)*sum(w.size for w,_ in weights)
                elif features.shape[1] == 36:
                    captured.append(raw.copy())
                else:
                    raise ValueError('Unexpected frozen neural layer')
                return raw
            setattr(stream_state, 'quantized_mlp', measured)
            try:
                pipeline = stream_state.CausalPipeline(ANN_ROOT/'software/weights/H2_D1_seed42.npz',
                    ANN_ROOT/'software/weights/H2-pro-Causal-ANN-v1_int8.npz',
                    variant='C1-proposal-motion',threshold=.35,initial_refiner=False)
                t0 = float(events['t'][0])
                pipeline.reset(t0)
                packet = np.empty(len(events),dtype=stream_state.EVENT_DTYPE)
                for name in ('x','y','t','p'):
                    packet[name] = events[name]
                pipeline.push(packet,float(events['t'][-1])+1e-6)
                windows,trailing = pipeline.finalize()
            finally:
                setattr(stream_state, 'quantized_mlp', original)
            ids = np.concatenate([w.event_indices for w in windows])
            if not np.array_equal(ids,np.arange(len(events)-trailing)):
                raise AssertionError('Frozen closed-prefix coverage mismatch')
            features = np.concatenate([w.features for w in windows])
            candidate = np.concatenate([w.candidate_mask for w in windows])
            teacher = np.concatenate([w.probabilities for w in windows])
            logits = np.concatenate(captured) if captured else np.empty((0,4),np.float32)
            rebuilt,_ = forward(features,layers)
            if not np.array_equal(rebuilt,logits) or not np.array_equal(probabilities(rebuilt),teacher[candidate]):
                raise AssertionError('Independent original MLP replay is not bit-exact')
            prior = teacher_rows[record['path']]
            prior_path = Path(teacher_cache)/prior['cache_file']
            if sha256(prior_path) != prior['cache_sha256']:
                raise ValueError('Existing teacher cache integrity mismatch')
            labels = events['label'][ids].astype(np.uint8)
            with np.load(prior_path,allow_pickle=False) as z:
                if any(not np.array_equal(a,z[key]) for a,key in
                       ((candidate,'candidate'),(teacher,'teacher'),(labels,'label'))):
                    raise AssertionError('Fresh original pipeline differs from previous frozen teacher')
            elapsed = float(events['t'][-1])-t0
            meta = {'sequence':record['path'],'role':'train_dev','source_sha256':record['sha256'],
                'events':len(ids),'target_events':int(labels.sum()),'candidate_events':int(candidate.sum()),
                'closed_windows':len(windows),'trailing_events':trailing,
                'high_density':len(events)*1000/elapsed >= 20000 if elapsed > 0 else False,
                'proposal_synaptic_mac':int(proposal_mac),'original_features':True,'teacher_exact_replay':True}
            temporary = target.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.savez_compressed(stream,features=features,label=labels,candidate=candidate,
                    teacher=teacher,logits=logits,metadata=np.array(json.dumps(meta)))
            os.replace(temporary,target)
            row = {**meta,'cache_file':target.name,'cache_sha256':sha256(target),
                   'prepare_seconds':time.perf_counter()-started}
            atomic_json(receipt,row)
            rows.append(row)
            print(json.dumps({'stage':'original_features','sequence':record['path'],
                              'events':len(ids),'candidate_events':int(candidate.sum()),'teacher_replay':'exact'}),flush=True)
        manifest = {**identity,'status':'complete','sequences':rows,'sources_utf8_lf':sources(),
                    'feature_shape':[36], 'reserved_sequences_read':0,
                    'complete_train_dev':len(rows)==len(role_records('train_dev'))}
        manifest_path = directory/'manifest.json'
        if not manifest_path.exists():
            atomic_json(manifest_path,manifest)
        else:
            saved = read_json(manifest_path)
            if saved['sequences'] != rows:
                raise ValueError('Completed cache changed')
            manifest = saved
        print(json.dumps({'status':'feature_cache_ready','cache':str(directory),'sequences':len(rows)}),flush=True)
        return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True,type=Path)
    p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--teacher-cache',required=True,type=Path)
    a=p.parse_args()
    prepare(a.dataset_root,a.cache,a.teacher_cache)


if __name__ == '__main__':
    main()
