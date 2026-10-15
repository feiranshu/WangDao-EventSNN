"""Read-only source/weak-label audit and raw-coordinate cache for FRED diagnosis.

Run with the existing base Anaconda environment (h5py and the existing ECF DLL).
The cache is a new artifact; no existing dataset, model or calibration is edited.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / 'software'))
from wangdao.fred_adapter_v1.core import parse_annotations
from wangdao.fred_adapter_v1.download import atomic_json, sha256

DATA = ROOT / 'results/runs/fred_external_v1'
BASELINE = ROOT / 'results/runs/fred_frozen_three_v1'
OUT = ROOT / 'results/runs/fred_input_diagnosis_v1'
DTYPE = np.dtype([('x','<u2'),('y','<u2'),('t','<i8'),('p','u1'),
                  ('label','u1'),('valid','?'),('region','u1'),('alternative','u1')])


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def dense_annotations(annotations):
    # Both selected recordings contain the one declared drone identity. Fail on
    # a different source schema rather than silently mishandling overlapping IDs.
    if len(annotations['classes']) != 1:
        raise ValueError('This independent audit expects the two single-track pilot recordings')
    boxes = np.full((len(annotations['times']),4), np.nan, np.float64)
    for i, row in enumerate(annotations['boxes']):
        if row:
            boxes[i] = next(iter(row.values()))
    return boxes


def independent_labels(raw, times, boxes):
    """Vectorized alternative implementation, independent of label_events()."""
    t = raw['t']
    index = np.searchsorted(times, t, side='right') - 1
    left = np.clip(index, 0, len(times)-1)
    right = np.minimum(left+1, len(times)-1)
    exact = (index >= 0) & (t == times[left])
    covered = (index >= 0) & (index < len(times)-1)
    gap = np.maximum(times[right]-times[left], 1)
    a, b = boxes[left], boxes[right]
    has_a, has_b = np.isfinite(a[:,0]), np.isfinite(b[:,0])
    stable = covered & (has_a == has_b) & (gap <= np.median(np.diff(times))*1.5)
    valid = stable | exact
    alpha = (t-times[left])/gap
    interpolated = a + alpha[:,None]*(b-a)
    x, y = raw['x'].astype(np.float64)+.5, raw['y'].astype(np.float64)+.5
    def hit(v):
        return (x >= v[:,0]) & (x < v[:,2]) & (y >= v[:,1]) & (y < v[:,3])
    left_hit, right_hit = hit(a), hit(b)
    shared = covered & has_a & has_b
    label = (shared & hit(interpolated)) | (exact & left_hit)
    nearest_hit = np.where(alpha <= .5, left_hit, right_hit)
    # Regions are diagnostics only, never fed into inference or input selection.
    box = interpolated.copy()
    box[exact] = a[exact]
    with np.errstate(invalid='ignore', divide='ignore'):
        rho = np.maximum(np.abs(2*x-box[:,0]-box[:,2])/(box[:,2]-box[:,0]),
                         np.abs(2*y-box[:,1]-box[:,3])/(box[:,3]-box[:,1]))
    region = np.zeros(len(raw), np.uint8)
    region[label] = 1
    region[label & (rho <= .8)] = 2
    region[label & (rho <= .5)] = 3
    # bit0 nearest frame, bit1 left frame, bit2 right frame; same valid mask.
    alternative = nearest_hit.astype(np.uint8) | (left_hit.astype(np.uint8)<<1) | (right_hit.astype(np.uint8)<<2)
    return label, valid, region, alternative


def snapshot(output):
    path = output/'preservation_before.json'
    if path.exists():
        return read_json(path)
    old = read_json(BASELINE/'preservation_before.json')
    paths = {Path(p) for p in old}
    ann = ROOT.parents[1]/'1_H2_model_improvement/implementation'
    for directory in [ROOT/'software', ann/'software', DATA, BASELINE]:
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                paths.add(p)
    # Our newly created standalone scripts are frozen separately in the run identity.
    paths = {p for p in paths if p.name not in ('prepare_fred_diagnosis.py','diagnose_fred_inputs.py',
                                              'review_fred_inputs.py','test_fred_input_diagnosis.py')}
    result = {str(p):{'sha256':sha256(p),'bytes':p.stat().st_size} for p in sorted(paths)}
    atomic_json(path,result)
    return result


def prepare(output=OUT):
    import h5py
    from wangdao.fred_adapter_v1.ecf import ECFCodec, read_chunk
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    snapshot(output)
    index = read_json(DATA/'dataset/index.json')
    codec_record = read_json(DATA/'runtime/codec_build.json')
    dll = DATA/'runtime/fred_ecf.dll'
    assert sha256(dll) == codec_record['binary_sha256']
    codec = ECFCodec(dll)
    source_identity = {'index_sha256':sha256(DATA/'dataset/index.json'),
        'baseline_comparison_sha256':sha256(BASELINE/'comparison.json'),
        'preparation_sha256':sha256(Path(__file__)), 'training_started':False}
    identity_path = output/'source_identity.json'
    if identity_path.exists() and read_json(identity_path) != source_identity:
        raise ValueError('Source/preparation identity changed; choose a new directory')
    atomic_json(identity_path,source_identity)
    reports=[]
    for seq in index['sequences']:
        name=seq['source_sequence']
        directory=output/'source'/name
        directory.mkdir(parents=True,exist_ok=True)
        if (directory/'audit.json').exists():
            report=read_json(directory/'audit.json')
            assert report['cache_sha256']==sha256(directory/'events.npy')
            reports.append(report)
            print(json.dumps({'stage':'reuse_source_cache','sequence':name}),flush=True)
            continue
        start=time.perf_counter()
        src=DATA/'dataset/source'/name
        assert sha256(src/'events.hdf5')==seq['source_hdf5_sha256']
        assert sha256(src/'coordinates.txt')==seq['annotation_sha256']
        times=np.asarray(read_json(src/'frame_timestamps.json'),np.int64)
        ann=parse_annotations((src/'coordinates.txt').read_text('utf-8'),times)
        boxes=dense_annotations(ann)
        temp=directory/'events.partial.npy'
        if temp.exists():temp.rename(temp.with_name(f'events.interrupted_{time.time_ns()}.npy'))
        cache=np.lib.format.open_memmap(str(temp),mode='w+',dtype=DTYPE,shape=(seq['raw_events'],))
        cursor=0
        with h5py.File(src/'events.hdf5','r') as handle:
            d=handle['CD/events']
            for begin in range(0,len(d),d.chunks[0]):
                block=read_chunk(d,begin,codec)
                for field in ('x','y','t','p'):
                    cache[field][cursor:cursor+len(block)]=block[field]
                cursor+=len(block)
        assert cursor==len(cache)
        old_bits=np.load(str(BASELINE/name/'predictions.npy'),mmap_mode='r',allow_pickle=False)
        mismatches=dict(coordinates=0,timestamps=0,polarity=0,weak_labels=0,valid_masks=0,saved_labels=0)
        alternatives={k:dict(events=0,flipped=0,positive_lost=0,positive_added=0) for k in ('nearest','left','right')}
        regions={str(i):dict(events=0,ann_tp=0,ann_fn=0) for i in range(1,4)}
        cursor=0
        for shard in seq['shards']:
            path=DATA/'dataset'/shard['path']
            assert sha256(path)==shard['sha256']
            with np.load(path,allow_pickle=False) as z:
                ev,valid=z['ev'],z['eval_mask']
            for offset in range(0,len(ev),500_000):
                n=min(500_000,len(ev)-offset)
                sl=slice(cursor+offset,cursor+offset+n)
                raw=cache[sl]
                original=ev[offset:offset+n]
                mask=valid[offset:offset+n]
                label,expected_valid,region,alt=independent_labels(raw,times,boxes)
                g=seq['geometry']
                for axis in ('x','y'):
                    mapped=np.floor((raw[axis].astype(np.float64)+.5)*g['scale']+g[f'offset_{axis}'])
                    mismatches['coordinates']+=int(np.count_nonzero(mapped!=original[axis]))
                mismatches['timestamps']+=int(np.count_nonzero(raw['t']/1000.!=original['t']))
                mismatches['polarity']+=int(np.count_nonzero(raw['p']!=original['p']))
                mismatches['weak_labels']+=int(np.count_nonzero(label!=original['label']))
                mismatches['valid_masks']+=int(np.count_nonzero(expected_valid!=mask))
                b=old_bits[sl]
                mismatches['saved_labels']+=int(np.count_nonzero(((b&32)!=0)!=label))
                raw['label'],raw['valid'],raw['region'],raw['alternative']=label,mask,region,alt
                take=mask & ((b&64)!=0)
                for j,k in enumerate(alternatives):
                    a=(alt&(1<<j))!=0
                    v=alternatives[k]
                    v['events']+=int(take.sum())
                    v['flipped']+=int(np.count_nonzero(take & (a!=label)))
                    v['positive_lost']+=int(np.count_nonzero(take & label & ~a))
                    v['positive_added']+=int(np.count_nonzero(take & ~label & a))
                for r,v in regions.items():
                    selected=take & (region==int(r))
                    v['events']+=int(selected.sum())
                    v['ann_tp']+=int(np.count_nonzero(selected & ((b&2)!=0)))
                    v['ann_fn']+=int(np.count_nonzero(selected & ((b&2)==0)))
            cursor+=len(ev)
            print(json.dumps({'stage':'source_label_audit','sequence':name,'events':cursor,
                             'seconds':time.perf_counter()-start}),flush=True)
        assert cursor==len(cache) and not any(mismatches.values()),mismatches
        cache.flush()
        del raw
        cache._mmap.close()
        del cache
        os.replace(temp,directory/'events.npy')
        present=np.isfinite(boxes[:,0])
        has_boxes=boxes[present]
        report={'status':'passed','sequence':name,'events':cursor,'mismatches':mismatches,
            'independent_label_formula':True,'region_definition':{'1':'outer 20% radial band',
                '2':'between central 50% and 80% width/height','3':'central 50% width/height'},
            'region_diagnostics':regions,'alternative_label_sensitivity':alternatives,
            'frames':len(times),'frames_with_box':int(present.sum()),
            'frame_gap_us_quantiles':np.quantile(np.diff(times),[0,.5,.99,1]).tolist(),
            'boxes_cross_sensor_edge':int(np.count_nonzero((has_boxes[:,0]<0)|(has_boxes[:,1]<0)|
                 (has_boxes[:,2]>1280)|(has_boxes[:,3]>720))),
            'cache_sha256':sha256(directory/'events.npy'),'source_identity':source_identity,
            'seconds':time.perf_counter()-start,'training_started':False}
        atomic_json(directory/'audit.json',report)
        reports.append(report)
    atomic_json(output/'source_audit.json',{'status':'passed','sequences':reports,
        'audited_events':sum(r['events'] for r in reports),
        'completed_at':datetime.now(timezone.utc).isoformat(),
        'limits':['independent formula agreement is not pixel-level ground truth',
                  'unannotated existing frames retain the original no-target assumption'],
        'training_started':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    prepare(p.parse_args().output)
