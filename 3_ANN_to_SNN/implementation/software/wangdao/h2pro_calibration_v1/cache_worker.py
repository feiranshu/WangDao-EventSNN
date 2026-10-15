"""Cache a fixed Train-main subset using the isolated, unchanged ANN package."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
import os
from pathlib import Path
import time
import numpy as np

from ..h2pro_snn.foundation import ANN_ROOT, read_json, sha256, verified_sequence
from ..h2pro_direct_v1.core import forward, load_layers, probabilities
from .common import identity, lock_directory, atomic_json, select_records


def prepare(dataset_root,output,teacher_cache):
    stream=import_module('wangdao.h2_pro.stream_state')
    from wangdao.h2 import representation
    from wangdao.data import load_events
    expected=ANN_ROOT/'software/wangdao'
    if any(Path(str(m.__file__)).absolute().parent.parent!=expected for m in (stream,representation)):
        raise RuntimeError('Use the isolated original ANN worker')
    teacher_path=Path(teacher_cache)/'manifest.json'
    teacher=read_json(teacher_path)
    spec=identity('train_main_original_features',cache_only=True)
    if (teacher['baseline_lock_sha256']!=spec['baseline_lock_sha256'] or
            not teacher['complete_development_roles'] or teacher['max_windows'] is not None):
        raise ValueError('Require complete frozen teacher cache')
    records=select_records(teacher)
    spec.update(role='train_main',teacher_manifest_sha256=sha256(teacher_path),
                selected_sequences=[r['path'] for r in records])
    prior={r['sequence']:r for r in teacher['roles']['train_main']}
    layers=load_layers(); original=stream.quantized_mlp
    with lock_directory(output,spec) as directory:
        rows=[]
        for record in records:
            target=directory/Path(record['path']).name
            receipt=target.with_suffix('.json')
            if receipt.exists():
                row=read_json(receipt)
                if (row['sequence']!=record['path'] or row['source_sha256']!=record['sha256']
                        or row['cache_sha256']!=sha256(target)):
                    raise ValueError('Existing calibration feature receipt mismatch')
                rows.append(row); continue
            started=time.perf_counter()
            events=load_events(verified_sequence(Path(dataset_root),record),require_labels=True)
            captured=[]; proposal_mac=0
            def measured(features,weights):
                nonlocal proposal_mac
                raw=original(features,weights)
                if features.shape[1]==128:
                    proposal_mac+=len(features)*sum(w.size for w,_ in weights)
                elif features.shape[1]==36:
                    captured.append(raw.copy())
                else:
                    raise ValueError('Unexpected original feature dimension')
                return raw
            setattr(stream,'quantized_mlp',measured)
            try:
                pipeline=stream.CausalPipeline(ANN_ROOT/'software/weights/H2_D1_seed42.npz',
                    ANN_ROOT/'software/weights/H2-pro-Causal-ANN-v1_int8.npz',
                    variant='C1-proposal-motion',threshold=.35,initial_refiner=False)
                t0=float(events['t'][0]); pipeline.reset(t0)
                packet=np.empty(len(events),dtype=stream.EVENT_DTYPE)
                for name in ('x','y','t','p'):packet[name]=events[name]
                pipeline.push(packet,float(events['t'][-1])+1e-6)
                windows,trailing=pipeline.finalize()
            finally:
                setattr(stream,'quantized_mlp',original)
            ids=np.concatenate([w.event_indices for w in windows])
            if not np.array_equal(ids,np.arange(len(events)-trailing)):
                raise AssertionError('Original closed-prefix coverage changed')
            features=np.concatenate([w.features for w in windows])
            candidate=np.concatenate([w.candidate_mask for w in windows])
            teacher_p=np.concatenate([w.probabilities for w in windows])
            logits=np.concatenate(captured) if captured else np.empty((0,4),np.float32)
            for lo in range(0,len(features),8192):
                rebuilt,_=forward(features[lo:lo+8192],layers)
                if not np.array_equal(rebuilt,logits[lo:lo+8192]):
                    raise AssertionError('Original frozen Refiner replay changed')
            if not np.array_equal(probabilities(logits),teacher_p[candidate]):
                raise AssertionError('Original probability readout changed')
            previous=prior[record['path']]
            previous_path=Path(teacher_cache)/previous['cache_file']
            if sha256(previous_path)!=previous['cache_sha256']:
                raise ValueError('Frozen teacher cache hash mismatch')
            labels=events['label'][ids].astype(np.uint8)
            with np.load(previous_path,allow_pickle=False) as z:
                for value,key in ((candidate,'candidate'),(teacher_p,'teacher'),(labels,'label')):
                    if not np.array_equal(value,z[key]):
                        raise AssertionError('Fresh Train-main replay differs from frozen teacher')
            duration=float(events['t'][-1])-t0
            dense=bool(len(events)*1000/duration>=20000) if duration>0 else False
            if dense!=previous['high_density']:
                raise AssertionError('Density grouping changed')
            meta={'sequence':record['path'],'role':'train_main','source_sha256':record['sha256'],
                  'events':len(ids),'target_events':int(labels.sum()),'candidate_events':int(candidate.sum()),
                  'high_density':dense,'proposal_synaptic_mac':int(proposal_mac),
                  'closed_windows':len(windows),'trailing_events':trailing,'original_features':True}
            tmp=target.with_suffix('.tmp')
            with tmp.open('wb') as f:
                np.savez_compressed(f,features=features,label=labels,candidate=candidate,
                    teacher=teacher_p,logits=logits,metadata=np.array(json.dumps(meta)))
            os.replace(tmp,target)
            row={**meta,'cache_file':target.name,'cache_sha256':sha256(target),
                 'prepare_seconds':time.perf_counter()-started}
            atomic_json(receipt,row); rows.append(row)
            print(json.dumps({'stage':'prepare_calibration','sequence':record['path'],
                'candidate_events':len(features),'seconds':row['prepare_seconds']}),flush=True)
        manifest={**spec,'status':'complete','sequences':rows,'reserved_sequences_read':0}
        path=directory/'manifest.json'
        if path.exists():
            if read_json(path)!=manifest:raise ValueError('Completed calibration cache changed')
        else:
            atomic_json(path,manifest)
        return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root','cache','teacher-cache'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    result=prepare(a.dataset_root,a.cache,a.teacher_cache)
    print(json.dumps({'status':result['status'],'calibration_sequences':len(result['sequences']),
                      'training_started':False}),flush=True)


if __name__=='__main__':main()
