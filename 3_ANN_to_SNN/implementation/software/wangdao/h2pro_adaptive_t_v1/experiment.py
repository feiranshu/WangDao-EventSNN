"""Resumable full-data frozen evaluation; Main seal is mandatory before Dev."""
from pathlib import Path
import json
import time
import numpy as np
import torch

from ..h2pro_r1_train_v1.data import manifest
from ..h2pro_calibration_v1.core import forward, Encoder, probabilities
from .common import (DEFAULT_CACHE, DEFAULT_RUN, SOURCE, DESIGN, identity, output_path,
    lock_directory, read_json, sha256, atomic_json, atomic_npz, sealed_json)
from .core import Policy, STEPS, THRESHOLD, trace_reference, incremental, choose, ledger
from .selection import select_main, score, policy_id, counts, summarize_counts


def cache_arrays(cache,row):
    with np.load(cache/row['cache_file'],allow_pickle=False) as z:
        a={k:z[k].copy() for k in z.files}
    y,c=a['labels'],a['candidate']
    if (c.dtype!=np.bool_ or c.shape!=(row['events'],) or y.shape!=c.shape
            or not np.isin(y,[0,1]).all() or int(c.sum())!=row['candidate_events']
            or int(y.sum())!=row['target_events'] or int(y[c].sum())!=row['candidate_targets']
            or a['features'].shape!=(int(c.sum()),36) or a['features'].dtype!=np.uint8
            or a['teacher'].shape!=(int(c.sum()),4) or not np.isfinite(a['teacher']).all()):
        raise ValueError('Frozen cache contents/coverage mismatch')
    return a


def emit(run,value):
    value=dict(value,unix_s=time.time())
    atomic_json(run/'progress.json',value)
    with (run/'progress.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(value,allow_nan=False)+'\n')
    print(json.dumps(value,allow_nan=False),flush=True)


def get_trace(run,cache,row,layers,device,batch_size):
    path=run/'traces'/row['role']/(Path(row['sequence']).stem+'.npz')
    receipt=path.with_suffix('.json')
    a=cache_arrays(cache,row)
    if receipt.exists():
        saved=read_json(receipt)
        if saved['cache_sha256']!=row['cache_sha256'] or sha256(path)!=saved['trace_sha256']:
            raise ValueError('Completed trace was changed')
        with np.load(path,allow_pickle=False) as z:
            raw,spikes=z['raw'],z['spikes']
    else:
        started=time.perf_counter()
        raw,spikes=trace_reference(a['features'],layers,device,batch_size)
        ids=np.unique(np.linspace(0,len(a['features'])-1,min(64,len(a['features'])),dtype=np.int64))
        for i,t in enumerate(STEPS):
            exact,cost=forward(a['features'][ids],layers,'r1',t,(Encoder(255,191),Encoder()),'spike_add')
            np.testing.assert_array_equal(exact,raw[i,ids])
            if cost.executed_synaptic_ac!=int(spikes[i,ids].sum())*32:
                raise AssertionError('Explicit prefix AC mismatch')
        atomic_npz(path,raw=raw,spikes=spikes)
        saved=dict(sequence=row['sequence'],role=row['role'],cache_sha256=row['cache_sha256'],
            trace_sha256=sha256(path),candidates=len(a['features']),explicit_candidates_per_prefix=len(ids),
            seconds=time.perf_counter()-started,device=str(device),torch=torch.__version__,numpy=np.__version__,
            backend='dense FP64 count reference for all prefixes; no early exit during trace preparation')
        atomic_json(receipt,saved)
    if (raw.shape!=(3,row['candidate_events'],4) or raw.dtype!=np.float32
            or spikes.shape!=(3,row['candidate_events']) or not np.isfinite(raw).all()
            or np.any(spikes<0) or np.any(np.diff(spikes,axis=0)<0)):
        raise ValueError('Invalid saved prefix trace')
    return a,raw,spikes,saved


def bundle(run,cache,rows,layers,device,batch_size):
    roles={r['role'] for r in rows}
    if len(roles)!=1:raise ValueError('One role at a time')
    pp=[];ss=[];yy=[];seqs=[];receipts=[];offset=0
    for j,row in enumerate(rows):
        a,raw,spikes,receipt=get_trace(run,cache,row,layers,device,batch_size)
        n=row['candidate_events']
        pp.append(np.stack([probabilities(r) for r in raw]));ss.append(spikes)
        yy.append(a['labels'][a['candidate']]);seqs.append(dict(row=row,lo=offset,hi=offset+n))
        receipts.append(receipt);offset+=n
        emit(run,dict(stage='trace_'+row['role'],sequence=row['sequence'],completed=j+1,total=len(rows)))
    return dict(role=next(iter(roles)),sequences=seqs,p=np.concatenate(pp,axis=1),
                spikes=np.concatenate(ss,axis=1),labels=np.concatenate(yy)),receipts


def seal_main(run,selection,receipts):
    sealed_json(run/'selection.json',selection)
    seal=dict(role='train_main',selection_sha256=sha256(run/'selection.json'),
              identity_sha256=sha256(run/'identity.json'),dev_used_for_selection=False,
              traces={r['sequence']:r['trace_sha256'] for r in receipts})
    sealed_json(run/'main_seal.json',seal)
    return seal


def require_seal(run):
    if not (run/'main_seal.json').exists():
        raise ValueError('Finish and seal Main selection before any Dev evaluation')
    seal=read_json(run/'main_seal.json');selection=read_json(run/'selection.json')
    if (seal['selection_sha256']!=sha256(run/'selection.json')
            or seal['identity_sha256']!=sha256(run/'identity.json')
            or selection['role']!='train_main' or selection['dev_used_for_selection']
            or seal['dev_used_for_selection']):
        raise ValueError('Main selection seal mismatch')
    for sequence,expected in seal['traces'].items():
        if sha256(run/'traces/train_main'/(Path(sequence).stem+'.npz'))!=expected:
            raise ValueError('Main trace changed after selection')
    return selection


def verify_policy(run,cache,rows,layers,policy,device,batch_size):
    records=[];key=policy_id(policy)
    for j,row in enumerate(rows):
        a,raw,spikes,_=get_trace(run,cache,row,layers,device,batch_size)
        p=np.stack([probabilities(r) for r in raw]);ix=choose(p,policy);ids=np.arange(len(ix))
        directory=run/'executed_dev'/key
        path=directory/(Path(row['sequence']).stem+'.npz');receipt=path.with_suffix('.json')
        if receipt.exists():
            rec=read_json(receipt)
            if rec['policy']!=policy.as_dict() or sha256(path)!=rec['output_sha256']:
                raise ValueError('Executed Dev result changed')
            with np.load(path,allow_pickle=False) as z:
                y,n,actual_ix=z['raw'],z['spikes'],z['exit_index']
            measured=rec['measured_operations']
        else:
            started=time.perf_counter()
            y,n,actual_ix,measured=incremental(a['features'],layers,policy,batch_size)
            rec=dict(sequence=row['sequence'],policy=policy.as_dict(),policy_id=key,
                     candidates=len(ix),seconds=time.perf_counter()-started,
                     backend='actual CPU binary spike-add with incremental membrane/accumulator and active-row exits',
                     measured_operations=measured)
        np.testing.assert_array_equal(actual_ix,ix)
        np.testing.assert_array_equal(y,raw[ix,ids])
        np.testing.assert_array_equal(n,spikes[ix,ids])
        expected=ledger(ix,n,policy)
        if any(value!=expected[k] for k,value in measured.items()):
            raise AssertionError('Actual early-exit operation counters differ from ledger')
        if not receipt.exists():
            atomic_npz(path,raw=y,spikes=n,exit_index=ix)
            rec.update(output_sha256=sha256(path),full_prefix_selection_exact=True,full_cost_exact=True)
            atomic_json(receipt,rec)
        records.append(rec)
        emit(run,dict(stage='explicit_dev',policy_id=key,sequence=row['sequence'],completed=j+1,total=len(rows)))
    return dict(policy_id=key,policy=policy.as_dict(),records=records,
                full_candidates_verified=sum(r['candidates'] for r in records),
                measured_operations={k:sum(r['measured_operations'][k] for r in records) for k in records[0]['measured_operations']})


def ann_reference(cache,rows):
    rec=[]
    for row in rows:
        a=cache_arrays(cache,row)
        c=counts(row,a['labels'][a['candidate']],probabilities(a['teacher'])>=THRESHOLD)
        rec.append(dict(sequence=row['sequence'],high_density=row['high_density'],counts=c))
    result={g:summarize_counts([r['counts'] for r in rec if g=='overall' or r['high_density']==(g=='high_density')])
            for g in ('overall','ordinary','high_density')}
    n=sum(r['candidate_events'] for r in rows)
    result.update(sequences=rec,cost=dict(mac_sparse_proposal=sum(r['proposal_sparse_mac'] for r in rows)+n*4480,
        mac_dense_proposal=sum(r['proposal_dense_mac'] for r in rows)+n*4480,ac=0),threshold=THRESHOLD,role='train_dev')
    return result


def run(action='run',directory=DEFAULT_RUN,cache=DEFAULT_CACHE,device='cuda',batch_size=8192):
    directory,cache=output_path(directory),Path(cache).resolve()
    if action=='status':
        return read_json(directory/'progress.json') if (directory/'progress.json').exists() else {'stage':'not_started'}
    if action not in ('run','main','dev'):raise ValueError('Unknown action')
    for source in (SOURCE.resolve(),cache):
        if directory==source or directory in source.parents or source in directory.parents:
            raise ValueError('Require independent output outside source and cache')
    if type(batch_size) is not int or batch_size<1:raise ValueError('Positive batch size')
    ident,layers=identity(cache)
    with lock_directory(directory,ident):
        m=manifest(cache)
        if (directory/'comparison.json').exists():
            require_seal(directory)
            done=read_json(directory/'comparison.json')
            for name,expected in done['artifact_sha256'].items():
                if sha256(directory/name)!=expected:raise ValueError('Completed experiment artifact changed')
            return dict(stage='complete',reused=True,comparison=str(directory/'comparison.json'))
        started=time.perf_counter()
        if action in ('run','main'):
            if not (directory/'main_seal.json').exists():
                main,receipts=bundle(directory,cache,m['roles']['train_main'],layers,device,batch_size)
                selection=select_main(main,lambda v:emit(directory,v))
                seal_main(directory,selection,receipts)
                emit(directory,dict(stage='main_sealed',selected={k:v['policy'] for k,v in selection['selected'].items()}))
                del main
            else:
                selection=require_seal(directory)
            if action=='main':return dict(stage='main_sealed',selection=str(directory/'selection.json'))
        selection=require_seal(directory)  # must run BEFORE loading/inferencing Dev
        dev,receipts=bundle(directory,cache,m['roles']['train_dev'],layers,device,batch_size)
        fixed={str(t):score(dev,fixed_step=t,details=True) for t in STEPS}
        source_result=read_json(SOURCE/'epochs/epoch_159.json')['evaluation']
        for g in ('overall','ordinary','high_density'):
            if fixed['16'][g]!=source_result[g]:raise AssertionError('Full T16 metrics do not reproduce source159')
        for k in ('inference_equivalent_mac_sparse_proposal','inference_equivalent_mac_dense_proposal','inference_equivalent_ac'):
            if fixed['16']['cost'][k]!=source_result['cost'][k]:raise AssertionError('Full T16 cost differs')
        ann=ann_reference(cache,m['roles']['train_dev'])
        old_ann=read_json(SOURCE/'ready.json')['frozen_ann']
        if any(ann[k]!=old_ann[k] for k in ('overall','ordinary','high_density','cost')):
            raise AssertionError('Frozen ANN reference differs')
        results={};verification={}
        for name,item in selection['selected'].items():
            policy=Policy(**item['policy']);key=policy_id(policy)
            if key not in verification:
                verification[key]=verify_policy(directory,cache,m['roles']['train_dev'],layers,policy,device,batch_size)
            results[name]=score(dev,policy,details=True)
            results[name]['full_explicit_verification']=key
        artifacts={p.relative_to(directory).as_posix():sha256(p) for p in directory.rglob('*')
                   if p.is_file() and p.suffix in ('.json','.npz') and p.name not in ('progress.json','comparison.json')}
        result=dict(status='complete',identity=ident,selection_sha256=sha256(directory/'selection.json'),
            main_selected_policies={k:v['policy'] for k,v in selection['selected'].items()},
            fixed=fixed,adaptive=results,frozen_ann=ann,verification=verification,
            source159_reproduced_exactly=True,artifact_sha256=artifacts,
            data_role='train_dev',weights_updated=False,training_started=False,reserved_splits_read=False,
            run_seconds_this_invocation=time.perf_counter()-started,
            timing_scope='calibration, verification and file IO; not deployment speed or energy')
        atomic_json(directory/'comparison.json',result)
        emit(directory,dict(stage='complete',comparison=str(directory/'comparison.json'),weights_updated=False,training_started=False))
        return dict(stage='complete',comparison=str(directory/'comparison.json'),reused=False)
