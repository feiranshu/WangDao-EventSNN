"""Immutable prefix reuse, Main sealing, and full executed Dev verification."""
from pathlib import Path
import time
import numpy as np
from ..h2pro_adaptive_t_v1.experiment import cache_arrays,emit,ann_reference
from ..h2pro_calibration_v1.core import probabilities
from ..h2pro_r1_train_v1.data import manifest
from .common import (DEFAULT_RUN,DEFAULT_CACHE,REFERENCE,SOURCE,identity,output_path,lock_directory,
    read_json,sha256,atomic_json,atomic_npz,sealed_json)
from .core import Policy,CONTROL,choose,incremental,ledger
from .scoring import score,policy_id
from .search import select_main


def read_prefix(cache,row,expected=None):
    """Read existing evidence only. Never call v1's cache-generating helper."""
    relative='traces/'+row['role']+'/'+Path(row['sequence']).stem+'.npz'
    path=REFERENCE/relative;receipt=path.with_suffix('.json')
    if not path.exists() or not receipt.exists():raise ValueError('Require complete original v1 prefixes')
    saved=read_json(receipt);digest=sha256(path)
    if digest!=saved['trace_sha256'] or saved['cache_sha256']!=row['cache_sha256']:
        raise ValueError('Original prefix/cache hash changed')
    if expected is not None and (expected[relative]!=digest or expected[relative[:-4]+'.json']!=sha256(receipt)):
        raise ValueError('Original Dev prefix receipt does not match completed v1 evidence')
    with np.load(path,allow_pickle=False) as z:raw,spikes=z['raw'],z['spikes']
    if (raw.shape!=(3,row['candidate_events'],4) or raw.dtype!=np.float32
            or spikes.shape!=(3,row['candidate_events']) or spikes.dtype!=np.int64
            or not np.isfinite(raw).all() or np.any(spikes<0) or np.any(np.diff(spikes,axis=0)<0)):
        raise ValueError('Invalid original prefix arrays')
    a=cache_arrays(cache,row)
    return a,raw,spikes,dict(sequence=row['sequence'],relative=relative,sha256=digest,cache_sha256=row['cache_sha256'])


def bundle(run,cache,rows,expected=None):
    roles={r['role'] for r in rows}
    if len(roles)!=1:raise ValueError('Load one data role at a time')
    parts={k:[] for k in ('p','spikes','labels')};sequences=[];receipts=[];offset=0
    for row in rows:
        a,raw,spikes,receipt=read_prefix(cache,row,expected)
        parts['p'].append(np.stack([probabilities(r) for r in raw]));parts['spikes'].append(spikes)
        parts['labels'].append(a['labels'][a['candidate']]);n=row['candidate_events']
        sequences.append(dict(row=row,lo=offset,hi=offset+n));offset+=n;receipts.append(receipt)
    emit(run,dict(stage='reuse_prefix_'+next(iter(roles)),sequences=len(rows),candidates=offset))
    return dict(role=next(iter(roles)),sequences=sequences,p=np.concatenate(parts['p'],axis=1),
                spikes=np.concatenate(parts['spikes'],axis=1),labels=np.concatenate(parts['labels'])),receipts


def seal_main(run,selection,arrays,receipts):
    archive=run/'fit_search.npz'
    if archive.exists():
        with np.load(archive,allow_pickle=False) as z:
            if set(z.files)!=set(arrays) or any(not np.array_equal(z[k],v) for k,v in arrays.items()):
                raise ValueError('Existing Main search archive changed')
    else:atomic_npz(archive,**arrays)
    sealed_json(run/'selection.json',selection)
    seal=dict(role='train_main',selection_sha256=sha256(run/'selection.json'),identity_sha256=sha256(run/'identity.json'),
        fit_search_sha256=sha256(archive),traces=receipts,dev_used_for_selection=False)
    sealed_json(run/'main_seal.json',seal)


def require_seal(run):
    if not (run/'main_seal.json').exists():raise ValueError('Seal Main before loading any Dev prefixes')
    seal=read_json(run/'main_seal.json');selection=read_json(run/'selection.json')
    if (seal['role']!='train_main' or selection['role']!='train_main' or seal['dev_used_for_selection']
            or selection['dev_used_for_selection'] or sha256(run/'identity.json')!=seal['identity_sha256']
            or sha256(run/'selection.json')!=seal['selection_sha256']
            or sha256(run/'fit_search.npz')!=seal['fit_search_sha256']):
        raise ValueError('Main selection seal mismatch')
    for row in seal['traces']:
        if not row['relative'].startswith('traces/train_main/') or sha256(REFERENCE/row['relative'])!=row['sha256']:
            raise ValueError('Main prefix changed after selection')
    return selection


def verify_policy(run,cache,rows,layers,policy,expected,batch_size):
    records=[];key=policy_id(policy)
    for j,row in enumerate(rows):
        a,raw,spikes,_=read_prefix(cache,row,expected)
        p=np.stack([probabilities(r) for r in raw]);ix=choose(p,policy);ids=np.arange(len(ix))
        path=run/'executed_dev'/key/(Path(row['sequence']).stem+'.npz');receipt=path.with_suffix('.json')
        if receipt.exists():
            rec=read_json(receipt)
            if rec['policy']!=policy.as_dict() or sha256(path)!=rec['output_sha256']:
                raise ValueError('Completed executed output changed')
            with np.load(path,allow_pickle=False) as z:y,n,actual_ix=z['raw'],z['spikes'],z['exit_index']
            measured=rec['measured_operations']
        else:
            started=time.perf_counter();y,n,actual_ix,measured=incremental(a['features'],layers,policy,batch_size)
            rec=dict(sequence=row['sequence'],policy=policy.as_dict(),policy_id=key,candidates=len(ix),
                seconds=time.perf_counter()-started,measured_operations=measured,
                backend='actual CPU binary spike-add with incrementally reused state and stopped candidate rows')
        np.testing.assert_array_equal(actual_ix,ix);np.testing.assert_array_equal(y,raw[ix,ids])
        np.testing.assert_array_equal(n,spikes[ix,ids]);cost=ledger(ix,n,policy)
        if any(v!=cost[k] for k,v in measured.items()):raise AssertionError('Executed operation counters differ')
        if not receipt.exists():
            atomic_npz(path,raw=y,spikes=n,exit_index=ix)
            rec.update(output_sha256=sha256(path),full_prefix_selection_exact=True,full_cost_exact=True)
            atomic_json(receipt,rec)
        records.append(rec)
        emit(run,dict(stage='explicit_dev',policy_id=key,sequence=row['sequence'],completed=j+1,total=len(rows)))
    return dict(policy_id=key,policy=policy.as_dict(),records=records,
                full_candidates_verified=sum(r['candidates'] for r in records),
                measured_operations={k:sum(r['measured_operations'][k] for r in records) for k in records[0]['measured_operations']})


def run(action='run',directory=DEFAULT_RUN,cache=DEFAULT_CACHE,batch_size=8192):
    run_dir,cache=output_path(directory),Path(cache).resolve()
    if action=='status':return read_json(run_dir/'progress.json') if (run_dir/'progress.json').exists() else {'stage':'not_started'}
    if action not in ('run','main','dev'):raise ValueError('Unknown action')
    if type(batch_size) is not int or batch_size<1:raise ValueError('Positive batch size')
    for source in (SOURCE.resolve(),REFERENCE.resolve(),cache):
        if run_dir==source or run_dir in source.parents or source in run_dir.parents:
            raise ValueError('Independent output directory outside original runs/cache required')
    ident,layers=identity(cache)
    with lock_directory(run_dir,ident):
        if (run_dir/'comparison.json').exists():
            require_seal(run_dir);done=read_json(run_dir/'comparison.json')
            for name,digest in done['artifact_sha256'].items():
                if sha256(run_dir/name)!=digest:raise ValueError('Completed experiment artifact changed')
            return dict(stage='complete',reused=True,comparison=str(run_dir/'comparison.json'))
        m=manifest(cache);started=time.perf_counter()
        if action in ('main','run'):
            if not (run_dir/'main_seal.json').exists():
                main,receipts=bundle(run_dir,cache,m['roles']['train_main'])
                original_split=read_json(REFERENCE/'selection.json')['split']
                selection,arrays=select_main(main,lambda v:emit(run_dir,v))
                if selection['split']!=original_split:raise AssertionError('Main split must be the same as v1')
                original_seal=read_json(REFERENCE/'main_seal.json')
                if any(r['sha256']!=original_seal['traces'][r['sequence']] for r in receipts):
                    raise ValueError('Main prefixes differ from original v1 seal')
                seal_main(run_dir,selection,arrays,receipts)
                emit(run_dir,dict(stage='main_sealed',selected={k:v['policy'] for k,v in selection['selected'].items()}))
                del main,arrays
            else:selection=require_seal(run_dir)
            if action=='main':return dict(stage='main_sealed',selection=str(run_dir/'selection.json'))
        selection=require_seal(run_dir)  # no Dev loading/scoring precedes this check
        original=read_json(REFERENCE/'comparison.json');expected=original['artifact_sha256']
        dev,_=bundle(run_dir,cache,m['roles']['train_dev'],expected)
        fixed=score(dev,fixed_step=16,details=True);shared=score(dev,CONTROL,details=True)
        for result,old in ((fixed,original['fixed']['16']),(shared,original['adaptive']['balanced'])):
            if any(result[k]!=old[k] for k in ('overall','ordinary','high_density','cost','sequences')):
                raise AssertionError('Frozen T16/shared control must exactly reproduce v1')
        ann=ann_reference(cache,m['roles']['train_dev'])
        if ann!=original['frozen_ann']:raise AssertionError('Frozen ANN control changed')
        key=policy_id(CONTROL)
        verification={key:verify_policy(run_dir,cache,m['roles']['train_dev'],layers,CONTROL,expected,batch_size)}
        results={}
        for name,item in selection['selected'].items():
            policy=Policy(**item['policy']);key=policy_id(policy)
            if key not in verification:verification[key]=verify_policy(run_dir,cache,m['roles']['train_dev'],layers,policy,expected,batch_size)
            results[name]=score(dev,policy,details=True);results[name]['full_explicit_verification']=key
        artifact_sha256={p.relative_to(run_dir).as_posix():sha256(p) for p in run_dir.rglob('*')
                         if p.is_file() and p.suffix in ('.npz','.json') and p.name not in ('progress.json','comparison.json')}
        result=dict(status='complete',identity=ident,main_selected_policies={k:v['policy'] for k,v in selection['selected'].items()},
            selection_sha256=sha256(run_dir/'selection.json'),fixed_t16=fixed,shared_control=shared,adaptive=results,
            frozen_ann=ann,verification=verification,artifact_sha256=artifact_sha256,
            controls_reproduced_exactly=True,data_role='train_dev',weights_updated=False,training_started=False,
            reserved_splits_read=False,run_seconds_this_invocation=time.perf_counter()-started,
            timing_scope='Main search / Dev verification / IO; no GPU-sparse, FPGA-latency or energy claim')
        atomic_json(run_dir/'comparison.json',result)
        emit(run_dir,dict(stage='complete',comparison=str(run_dir/'comparison.json'),weights_updated=False,training_started=False))
        return dict(stage='complete',reused=False,comparison=str(run_dir/'comparison.json'))
