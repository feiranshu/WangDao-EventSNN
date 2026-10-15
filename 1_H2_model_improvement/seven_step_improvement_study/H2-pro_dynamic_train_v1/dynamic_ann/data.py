"""Immutable full-coverage feature cache and GPU-resident paired event batching."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import math
from pathlib import Path
import time
import numpy as np
import torch
from .common import (ARMS,ROLES,PRIOR,ROOT,SequenceContext,POLICY_MAP,sha,read,digest,manifests,
                     load_sequence,cache_identity,atomic_json,atomic_npz,local_output,preserve_before)
from .storage import process_lock


def prepared_sequence(cache,role,row,frow,identity):
    cache=Path(cache);record=cache/(role+'_'+Path(row['sequence']).stem+'.json')
    if record.exists():
        r=read(record)
        if r['identity_sha256']!=digest(identity):raise ValueError('Existing feature identity differs')
        for v in r['arms'].values():
            if sha(cache/v['file'])!=v['sha256']:raise ValueError('Saved feature payload changed')
        return r,True
    e,y,fc,fp,ff,meta=load_sequence(role,row,frow)
    prior=read(PRIOR/role/(Path(row['sequence']).stem+'.json'))
    pred=PRIOR/role/prior['payload']
    if sha(pred)!=prior['payload_sha256']:raise ValueError('Prior prediction payload changed')
    context=SequenceContext(e,meta['t0_ms'],meta['closed_windows']);arms={}
    with np.load(pred,allow_pickle=False) as z:
        for arm,policy in ARMS.items():
            a=context.run(POLICY_MAP[policy])
            np.testing.assert_array_equal(a['candidate'],z[policy+'__candidate'])
            np.testing.assert_array_equal(a['probabilities'],z[policy+'__probabilities'])
            if a['cost']!=prior['policies'][policy]['cost']:raise AssertionError('Frozen policy cost differs')
            if arm=='fixed100':np.testing.assert_array_equal(a['features'],ff)
            candidate_ids=np.flatnonzero(a['candidate']).astype(np.int32)
            filename=f'{role}_{Path(row["sequence"]).stem}_{arm}.npz'
            arrays=dict(features=a['features'],candidate_ids=candidate_ids,full_labels=y.astype(np.uint8),
                        initial_probability=a['probabilities'],windows=a['windows'])
            output=cache/filename
            if output.exists():
                with np.load(output,allow_pickle=False) as old:
                    if set(old.files)!=set(arrays):raise ValueError('Orphan feature file schema differs')
                    for key,value in arrays.items():np.testing.assert_array_equal(old[key],value)
            else:atomic_npz(output,**arrays)
            arms[arm]=dict(file=filename,sha256=sha(output),candidate_events=len(candidate_ids),
                candidate_targets=int(y[candidate_ids].sum()),cost=a['cost'],waiting_ms=a['wait_ms'],
                initial_metrics=prior['policies'][policy]['metrics'])
    result=dict(identity_sha256=digest(identity),role=role,sequence=row['sequence'],metadata=meta,
        source_sha256=row['source_sha256'],raw_cache_sha256=row['cache_sha256'],arms=arms)
    atomic_json(record,result,immutable=True)
    return result,False


def prepare(cache,workers=4):
    cache=local_output(cache);preserve_before();ident=cache_identity();raw,feat=manifests()
    with process_lock(cache):
        atomic_json(cache/'identity.json',ident,immutable=True)
        rows={role:[] for role in ROLES};tasks=[];started=time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for role in ROLES:
                frows={r['sequence']:r for r in feat['roles'][role]}
                for row in raw['roles'][role]:tasks.append(pool.submit(prepared_sequence,str(cache),role,row,frows[row['sequence']],ident))
            for i,future in enumerate(as_completed(tasks),1):
                row,resumed=future.result();rows[row['role']].append(row)
                print(f'cache {i}/89 {row["role"]} {row["sequence"]} resumed={resumed} elapsed={time.perf_counter()-started:.1f}s',flush=True)
        for values in rows.values():values.sort(key=lambda r:r['sequence'])
        stats={role:dict(sequences=len(values),source_events=sum(r['metadata']['published_events'] for r in values),
                        candidates={arm:sum(r['arms'][arm]['candidate_events'] for r in values) for arm in ARMS}) for role,values in rows.items()}
        manifest=dict(version=ident['version'],status='complete',identity=ident,roles=rows,stats=stats,
                      real_data_optimizer_steps=0,reserved_sequences_read=False)
        atomic_json(cache/'manifest.json',manifest,immutable=True)
    return manifest


def verify_cache(cache):
    cache=local_output(cache);m=read(cache/'manifest.json')
    if m['status']!='complete' or m['identity']!=cache_identity() or m['identity']!=read(cache/'identity.json'):
        raise ValueError('Incomplete or incompatible feature cache')
    source,_=manifests()
    if set(m['roles'])!=set(ROLES):raise ValueError('Unexpected data role')
    for role in ROLES:
        expected={r['sequence']:r for r in source['roles'][role]}
        if len(m['roles'][role])!=len(expected) or set(r['sequence'] for r in m['roles'][role])!=set(expected):
            raise ValueError('Incomplete source sequence coverage')
        for row in m['roles'][role]:
            if row['role']!=role or row['source_sha256']!=expected[row['sequence']]['source_sha256']:
                raise ValueError('Source role or identity changed')
            if set(row['arms'])!=set(ARMS):raise ValueError('Incomplete paired feature record')
            for arm,v in row['arms'].items():
                if Path(v['file']).name!=v['file'] or sha(cache/v['file'])!=v['sha256']:
                    raise ValueError('Feature payload hash mismatch')
    return m


class ResidentData:
    """Every source event has a mapping; Proposal rejects remain evaluation FNs."""
    def __init__(self,cache,arm,device,manifest=None):
        started=time.perf_counter();self.cache=Path(cache);self.arm=arm;self.device=device
        self.manifest=verify_cache(cache) if manifest is None else manifest
        self.roles={};self.sequences={};total_bytes=0
        for role,records in self.manifest['roles'].items():
            features=[];labels=[];maps=[];sequences=[];offset=0;source_offset=0
            for row in records:
                rec=row['arms'][arm]
                with np.load(self.cache/rec['file'],allow_pickle=False) as z:
                    x=z['features'];ids=z['candidate_ids'];y=z['full_labels'];p=z['initial_probability']
                    n=row['metadata']['published_events']
                    if (x.dtype!=np.uint8 or x.shape!=(len(ids),36) or ids.dtype!=np.int32 or y.dtype!=np.uint8
                            or y.shape!=(n,) or p.shape!=(n,) or not np.isin(y,[0,1]).all()
                            or (len(ids) and (ids[0]<0 or ids[-1]>=n or np.any(np.diff(ids)<=0)))
                            or len(ids)!=rec['candidate_events'] or int(y[ids].sum())!=rec['candidate_targets']):
                        raise ValueError('Feature arrays or candidate coverage invalid')
                    features.append(x.copy());labels.append(y[ids].copy())
                    mapping=np.full(n,-1,np.int32);mapping[ids]=np.arange(offset,offset+len(ids),dtype=np.int32);maps.append(mapping)
                    sequences.append(dict(row=row,lo=offset,hi=offset+len(ids),source_lo=source_offset,
                        source_hi=source_offset+n,full_labels=y.copy(),candidate_ids=ids.copy(),initial_probability=p.copy()))
                    offset+=len(ids);source_offset+=n
            host={'features':np.concatenate(features),'labels':np.concatenate(labels),'mapping':np.concatenate(maps)}
            self.roles[role]={}
            for key,value in host.items():
                tensor=torch.from_numpy(value);total_bytes+=tensor.numel()*tensor.element_size()
                if device.type=='cuda':tensor=tensor.pin_memory()
                self.roles[role][key]=tensor.to(device,non_blocking=device.type=='cuda')
            self.sequences[role]=sequences
        if device.type=='cuda':torch.cuda.synchronize(device)
        self.stats=dict(arm=arm,placement=str(device),resident_bytes=total_bytes,load_seconds=time.perf_counter()-started,
            source_events=self.source_count,candidate_events=self.candidate_count,dev_candidate_events=len(self.roles['train_dev']['features']),
            per_batch_disk_reads=0,per_batch_input_h2d_bytes=0,per_batch_cpu_features=False,
            shuffle='one deterministic GPU source-event permutation per epoch',source_permutation_bytes=self.source_count*8)

    @property
    def source_count(self):return len(self.roles['train_main']['mapping'])
    @property
    def candidate_count(self):return len(self.roles['train_main']['features'])
    def batches(self,config):return math.ceil(self.source_count/config.source_batch_size)
    def order(self,epoch,seed):
        generator=torch.Generator(device=self.device);generator.manual_seed(seed*1000003+epoch)
        return torch.randperm(self.source_count,generator=generator,device=self.device)
    def batch(self,source_ids):
        mapped=self.roles['train_main']['mapping'][source_ids]
        selected=mapped[mapped>=0].long()
        return self.roles['train_main']['features'][selected],self.roles['train_main']['labels'][selected]
