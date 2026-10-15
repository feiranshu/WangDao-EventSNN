"""Reproducible Main selection, sealed Dev verification, and prediction audit."""
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .core import (ANN_ROOT, SNN_ROOT, EXPERIMENT, EVENT_DTYPE, POLICIES,
                   POLICY_MAP, SequenceContext, counts, metrics, schedule)

RAW = SNN_ROOT / 'results/runs/h2pro_teacher_v1'
FEATURES = SNN_ROOT / 'results/runs/h2pro_distill_v1_features'
LOCK = SNN_ROOT / 'software/wangdao/h2pro_snn/baseline_lock.json'
ALLOWED = ('train_main', 'train_dev')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def save(path, value):
    """Immutable complete artifacts; recovery can replace only own partial file."""
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing artifact differs; use a new run directory: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    partial.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(partial, path)


def verify_baseline():
    lock = read(LOCK)
    for rel, expected in lock['files_sha256'].items():
        if sha(ANN_ROOT / rel) != expected:
            raise ValueError(f'Frozen ANN file changed: {rel}')
    for rel, expected in lock['source_sha256_utf8_lf'].items():
        data = (ANN_ROOT / rel).read_text(encoding='utf-8').encode('utf-8')
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Frozen ANN source changed: {rel}')
    return sha(LOCK)


def manifests():
    lock_sha = verify_baseline()
    raw, feat = read(RAW/'manifest.json'), read(FEATURES/'manifest.json')
    index = read(ANN_ROOT/'data/index.json')
    if (raw['status'] != 'complete' or feat['status'] != 'complete'
            or raw['max_windows'] is not None or raw['baseline_lock_sha256'] != lock_sha
            or feat['identity']['baseline_lock_sha256'] != lock_sha
            or feat['identity']['teacher_manifest_sha256'] != sha(RAW/'manifest.json')):
        raise ValueError('Incomplete or incompatible development caches')
    sources = {r['path']: r['sha256'] for r in index['files']}
    for role in ALLOWED:
        expected = {'train/'+Path(p).name for p in index['roles'][role]}
        rr, ff = raw['roles'][role], feat['roles'][role]
        if (set(r['sequence'] for r in rr) != expected or len(rr) != len(expected)
                or set(r['sequence'] for r in ff) != expected or len(ff) != len(expected)):
            raise ValueError('Missing, duplicated or wrong-role sequence')
        for row in rr + ff:
            if row['source_sha256'] != sources[row['sequence']]:
                raise ValueError('Source provenance mismatch')
    return raw, feat


def identity():
    raw, _ = manifests()
    sources = ['protocol.json','run.py','src/__init__.py','src/core.py','src/workflow.py',
               'tests/test_dynamic_window.py']
    return dict(version='h2pro_dynamic_window_v1', baseline_lock_sha256=sha(LOCK),
        raw_manifest_sha256=sha(RAW/'manifest.json'), feature_manifest_sha256=sha(FEATURES/'manifest.json'),
        sources={p:sha(EXPERIMENT/p) for p in sources},
        policies=[p.__dict__ for p in POLICIES],
        sequences={role:[r['sequence'] for r in raw['roles'][role]] for role in ALLOWED},
        python=platform.python_version(), numpy=np.__version__, blas_threads=1,
        training_started=False, reserved_sequences_read=False)


def preservation_snapshot():
    """Metadata for all old implementation files, content hashes for code/inputs.

    Large unrelated historical checkpoints are not all rehashed. Their path,
    byte size and nanosecond modification time are checked, and this limitation
    is explicit in the report. README is the sole authorized existing-file edit.
    """
    result = {}
    for root in (ANN_ROOT, SNN_ROOT):
        for p in root.rglob('*'):
            if not p.is_file() or '__pycache__' in p.parts or p == SNN_ROOT/'README.md':
                continue
            stat = p.stat()
            row = dict(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if ('software' in p.relative_to(root).parts or p.parent in (RAW, FEATURES)
                    or p == ANN_ROOT/'data/index.json'):
                row['sha256'] = sha(p)
            result[str(p)] = row
    return result


def load_sequence(role, row, frow):
    if role not in ALLOWED:
        raise ValueError('Reserved roles cannot be opened by this experiment')
    if row['sequence'] != frow['sequence']:
        raise ValueError('Wrong feature sequence')
    rp, fp = RAW/row['cache_file'], FEATURES/frow['cache_file']
    for p, expected in ((rp,row['cache_sha256']), (fp,frow['cache_sha256'])):
        if p.parent not in (RAW, FEATURES) or sha(p) != expected:
            raise ValueError('Cache path or hash mismatch')
    with np.load(rp,allow_pickle=False) as z:
        meta = json.loads(str(z['metadata'].item()))
        e = np.empty(len(z['t']), EVENT_DTYPE)
        for key in e.dtype.names:
            e[key] = z[key]
        labels, candidate, teacher = z['label'].copy(), z['candidate'].copy(), z['teacher'].copy()
    if (meta['role'] != role or meta['sequence'] != row['sequence']
            or meta['source_sha256'] != row['source_sha256'] or meta['max_windows'] is not None
            or meta['baseline_lock_sha256'] != sha(LOCK)
            or meta['published_events'] != len(e) or meta['beyond_prefix_events'] != 0
            or meta['input_events'] != len(e)+meta['trailing_events']
            or not np.isin(labels,[0,1]).all()):
        raise ValueError('Cache coverage/label/role mismatch')
    with np.load(fp,allow_pickle=False) as z:
        if not np.array_equal(candidate,z['candidate']) or not np.array_equal(labels,z['labels']):
            raise ValueError('Independent cache alignment failed')
        features = z['features'].copy()
    return e, labels, candidate, teacher, features, meta


def sequence_stem(sequence):
    return Path(sequence).stem


def verified_record(out, role, sequence, run_identity, policy_names):
    path = Path(out)/role/(sequence_stem(sequence)+'.json')
    if not path.exists():
        return None
    r = read(path)
    if (r['identity_sha256'] != run_identity or r['sequence'] != sequence or r['role'] != role
            or list(r['policies']) != policy_names):
        raise ValueError('Completed record has a different identity')
    payload = path.parent/r['payload']
    if payload.parent != path.parent or sha(payload) != r['payload_sha256']:
        raise ValueError('Completed prediction payload changed')
    return r


def evaluate_sequence(out, role, row, frow, policy_names, run_identity):
    resumed = verified_record(out,role,row['sequence'],run_identity,policy_names)
    if resumed is not None:
        return resumed, True
    started = time.perf_counter()
    e, labels, frozen_c, frozen_p, frozen_f, meta = load_sequence(role,row,frow)
    context = SequenceContext(e, meta['t0_ms'], meta['closed_windows'])
    payload = {'event_index':np.arange(len(e),dtype=np.int64)}
    summary = {}
    for name in policy_names:
        a = context.run(POLICY_MAP[name])
        if name == 'fixed100':
            if (not np.array_equal(a['candidate'],frozen_c)
                    or not np.array_equal(a['probabilities'],frozen_p)
                    or not np.array_equal(a['features'],frozen_f)
                    or a['cost']['proposal_current_sparse_mac'] != frow['proposal_sparse_mac']
                    or a['cost']['proposal_current_dense_mac'] != frow['proposal_dense_mac']):
                raise AssertionError(f"Frozen baseline replay mismatch: {row['sequence']}")
        if not np.isfinite(a['probabilities']).all():
            raise FloatingPointError('Nonfinite prediction')
        for key in ('candidate','probabilities','windows'):
            payload[f'{name}__{key}'] = a[key]
        durations = (a['windows'][:,1]-a['windows'][:,0])*25
        summary[name] = dict(metrics=metrics(counts(labels,a['probabilities'],a['candidate'])),
            cost=a['cost'],wait_ms=a['wait_ms'],replay_seconds=a['replay_seconds'],
            windows=len(durations),window_histogram_ms={str(d):int((durations==d).sum()) for d in (25,50,75,100)})
    directory = Path(out)/role
    directory.mkdir(parents=True,exist_ok=True)
    final = directory/(sequence_stem(row['sequence'])+'.npz')
    if final.exists():
        # An interrupted record can leave a complete payload. Recompute and
        # compare content; do not replace it based on zip timestamps.
        with np.load(final,allow_pickle=False) as z:
            if set(z.files) != set(payload) or any(not np.array_equal(z[k],v) for k,v in payload.items()):
                raise ValueError('Orphan payload differs; use a new output directory')
    else:
        partial = final.with_suffix('.partial.npz')
        np.savez_compressed(partial,**payload)
        os.replace(partial,final)
    result = dict(role=role,sequence=row['sequence'],identity_sha256=run_identity,
        source_sha256=row['source_sha256'],cache_sha256=row['cache_sha256'],
        feature_cache_sha256=frow['cache_sha256'],metadata=meta,
        baseline_exact=True,policies=summary,payload=final.name,payload_sha256=sha(final),
        memoized_intervals=context.cache_misses,reused_intervals=context.cache_hits,
        total_replay_seconds=time.perf_counter()-started)
    save(final.with_suffix('.json'),result)
    return result, False


def aggregate(records, policy_names):
    output = {}
    count_keys = ('events','target_events','candidate_events','candidate_targets','tp','fp','fn','tn','proposal_fn','refiner_fn')
    for name in policy_names:
        parts = [r['policies'][name] for r in records]
        total = {k:sum(p['metrics'][k] for p in parts) for k in count_keys}
        hd = {k:sum(r['policies'][name]['metrics'][k] for r in records if r['metadata']['high_density']) for k in count_keys}
        cost = {k:sum(p['cost'][k] for p in parts) for k in parts[0]['cost']}
        output[name] = dict(metrics=metrics(total),high_density=metrics(hd),cost=cost,
            mean_publication_wait_ms=sum(p['wait_ms']['mean']*p['metrics']['events'] for p in parts)/total['events'],
            maximum_publication_wait_ms=max(p['wait_ms']['maximum'] for p in parts),
            windows=sum(p['windows'] for p in parts),
            window_histogram_ms={str(d):sum(p['window_histogram_ms'][str(d)] for p in parts) for d in (25,50,75,100)})
    return dict(sequences=len(records),high_density_sequences=sum(r['metadata']['high_density'] for r in records),
        original_events=sum(r['metadata']['original_events'] for r in records),
        published_events=sum(r['metadata']['published_events'] for r in records),
        common_unpublished_tail_events=sum(r['metadata']['trailing_events'] for r in records),policies=output)


def select(summary):
    p = summary['policies']
    def iou_key(n):
        v = p[n]
        return v['metrics']['iou'],v['high_density']['recall'],v['metrics']['precision'],-v['cost']['mac_sparse_proposal']
    def recall_key(n):
        v = p[n]
        return v['high_density']['recall'],v['metrics']['iou'],v['metrics']['precision'],-v['cost']['mac_sparse_proposal']
    adaptive = [n for n in p if n.startswith('adaptive')]
    choices = {f'best_iou_{scale}':max([n for n in adaptive if n.endswith('_'+scale)],key=iou_key)
               for scale in ('raw','rate100')}
    choices['best_high_density_recall'] = max(adaptive,key=recall_key)
    names = list(dict.fromkeys(['fixed100','fixed50_raw','fixed25_raw']+list(choices.values())))
    return dict(choices=choices,dev_policies=names,reason='Main-only ranking; no zero-regression filter; all Main alternatives retained')


def evidence_hashes(out,role):
    directory = Path(out)/role
    return {str(p.relative_to(out)):sha(p) for p in sorted(directory.iterdir()) if p.suffix in ('.json','.npz') and '.partial' not in p.name}


def validate_seal(out):
    out = Path(out)
    seal = read(out/'selection.json')
    if (seal['identity_sha256'] != digest(read(out/'identity.json'))
            or seal['main_summary_sha256'] != sha(out/'train_main_summary.json')
            or seal['main_evidence'] != evidence_hashes(out,'train_main')
            or seal['selection'] != select(read(out/'train_main_summary.json'))):
        raise ValueError('Main evidence or selection seal changed')
    return seal


def run_stage(out,role,workers=4):
    if role not in ALLOWED:
        raise ValueError('Only Main selection and Dev verification are supported')
    out = Path(out)
    current = identity()
    if role == 'train_dev' and not (out/'selection.json').exists():
        raise ValueError('Complete and seal Main before opening Dev payloads')
    save(out/'identity.json',current)
    if not (out/'preservation_before.json').exists():
        save(out/'preservation_before.json',preservation_snapshot())
    raw, feat = manifests()
    policy_names = [p.name for p in POLICIES] if role=='train_main' else validate_seal(out)['selection']['dev_policies']
    feature_rows = {r['sequence']:r for r in feat['roles'][role]}
    rows = raw['roles'][role]
    futures=[]; records=[]
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in rows:
            futures.append(pool.submit(evaluate_sequence,str(out),role,row,feature_rows[row['sequence']],policy_names,digest(current)))
        for future in as_completed(futures):
            record,resumed = future.result()
            records.append(record)
            print(json.dumps(dict(stage=role,complete=len(records),total=len(rows),sequence=record['sequence'],
                resumed=resumed,sequence_seconds=round(record['total_replay_seconds'],3),elapsed_seconds=round(time.perf_counter()-started,2))),flush=True)
    records.sort(key=lambda r:r['sequence'])
    summary = aggregate(records,policy_names)
    save(out/(role+'_summary.json'),summary)
    if role == 'train_main':
        seal = dict(identity_sha256=digest(current),main_summary_sha256=sha(out/'train_main_summary.json'),
                    main_evidence=evidence_hashes(out,role),selection=select(summary),dev_opened_before_seal=False)
        save(out/'selection.json',seal)
        print(json.dumps(seal['selection']),flush=True)
    return summary


def paired_bootstrap(records,names,draws=10000):
    """Paired sequence bootstrap, preserving the observed HD/non-HD stratum sizes."""
    rng=np.random.default_rng(42)
    hd=[i for i,r in enumerate(records) if r['metadata']['high_density']]
    normal=[i for i,r in enumerate(records) if not r['metadata']['high_density']]
    indices=np.concatenate([rng.choice(group,(draws,len(group))) for group in (hd,normal) if group],axis=1)
    result={}
    for name in names:
        differences=[]
        for arm in (name,'fixed100'):
            a=np.asarray([[r['policies'][arm]['metrics'][k] for k in ('tp','fp','fn')] for r in records])
            v=a[indices].sum(1)
            differences.append(v[:,0]/np.maximum(v.sum(1),1))
        delta=differences[0]-differences[1]
        result[name]=dict(iou_difference_95_percentile_interval=np.quantile(delta,[.025,.975]).tolist(),
            bootstrap_positive_fraction=float((delta>0).mean()),draws=draws,
            caution='Exploratory paired stratified sequence bootstrap; only two HD Dev sequences, not a held-out test significance claim')
    return result


def audit(out):
    """Re-read ALL stored predictions; independently rebuild confusion counts."""
    out=Path(out); current=identity()
    if read(out/'identity.json') != current:
        raise ValueError('Runtime identity changed')
    seal=validate_seal(out)
    raw,feat=manifests();checked_events=0;checked_predictions=0;records_by_role={}
    for role in ALLOWED:
        names=[p.name for p in POLICIES] if role=='train_main' else seal['selection']['dev_policies']
        frows={r['sequence']:r for r in feat['roles'][role]};records=[]
        for row in raw['roles'][role]:
            r=verified_record(out,role,row['sequence'],digest(current),names)
            if r is None:raise ValueError('Missing completed record')
            e,y,c,p,f,meta=load_sequence(role,row,frows[row['sequence']])
            with np.load(out/role/r['payload'],allow_pickle=False) as z:
                if not np.array_equal(z['event_index'],np.arange(len(e))):raise AssertionError('Event identity changed')
                for name in names:
                    pc,pp,ww=(z[f'{name}__{k}'] for k in ('candidate','probabilities','windows'))
                    if (pp.shape!=y.shape or pc.shape!=y.shape or not np.isfinite(pp).all()
                            or np.any((pp<0)|(pp>1)) or np.any(pp[~pc]!=0)):
                        raise AssertionError('Prediction contract violation')
                    pred=pp>=.35;positive=y==1
                    # Different confusion implementation from core.counts.
                    matrix=np.bincount((positive.astype(int)*2+pred.astype(int)),minlength=4)
                    tn,fp,fn,tp=map(int,matrix)
                    direct=dict(events=len(y),target_events=int(positive.sum()),candidate_events=int(pc.sum()),
                        candidate_targets=int(np.count_nonzero(positive[pc])),tp=tp,fp=fp,fn=fn,tn=tn,
                        proposal_fn=int(np.count_nonzero(positive[~pc])),refiner_fn=int(np.count_nonzero(positive&pc&~pred)))
                    if metrics(direct)!=r['policies'][name]['metrics']:raise AssertionError('Metric audit failed')
                    plan=schedule(e['t'],meta['t0_ms'],meta['closed_windows'],POLICY_MAP[name])
                    last=0;wait_total=0.
                    for w,(start,end,count) in zip(ww,plan):
                        lo,hi=np.searchsorted(e['t'],meta['t0_ms']+np.array([start,end])*25,side='left')
                        expected=[start,end,int(lo),int(hi),int(pc[lo:hi].sum())]
                        if list(w)!=expected or lo!=last or hi-lo!=count:raise AssertionError('Window coverage audit failed')
                        last=hi;wait_total+=float(np.sum(meta['t0_ms']+end*25-e['t'][lo:hi]))
                    if len(ww)!=len(plan) or last!=len(y):raise AssertionError('Incomplete window coverage')
                    if not np.isclose(wait_total/len(y),r['policies'][name]['wait_ms']['mean'],rtol=0,atol=1e-10):
                        raise AssertionError('Waiting time audit failed')
                    cost=r['policies'][name]['cost'];dense_ref=int(pc.sum())*4480
                    if (cost['refiner_mac']!=dense_ref or cost['neural_ac']!=0
                            or cost['mac_sparse_proposal']!=dense_ref+cost['proposal_current_sparse_mac']+cost['proposal_history_extra_sparse_mac']
                            or cost['mac_dense_proposal']!=dense_ref+cost['proposal_current_dense_mac']+cost['proposal_history_extra_dense_mac']):
                        raise AssertionError('Cost arithmetic failed')
                    # Independently count nonempty current/extra anchor inference.
                    full=set(int(w[0]//4) for w in ww if w[1]-w[0]==4)
                    anchor=np.searchsorted(e['t'],meta['t0_ms']+np.arange(meta['closed_windows']+1)*100.)
                    anchor_calls=sum(anchor[m+1]>anchor[m] for m in range(meta['closed_windows']) if m not in full)
                    if (cost['proposal_current_dense_mac']!=int(np.count_nonzero(ww[:,3]-ww[:,2]))*2580480
                            or cost['proposal_history_extra_dense_mac']!=int(anchor_calls)*2580480):
                        raise AssertionError('Anchor/current dense MAC audit failed')
                    if name=='fixed100' and (not np.array_equal(pc,c) or not np.array_equal(pp,p)):
                        raise AssertionError('Baseline probabilities or candidates differ')
                    checked_predictions+=len(y)
            checked_events+=len(y);records.append(r)
        records.sort(key=lambda r:r['sequence']);records_by_role[role]=records
        if aggregate(records,names)!=read(out/(role+'_summary.json')):raise AssertionError('Aggregate audit failed')
    preservation=read(out/'preservation_before.json');after=preservation_snapshot()
    changed=[p for p,v in preservation.items() if after.get(p)!=v]
    added=[p for p in after if p not in preservation]
    save(out/'preservation_after.json',dict(original_files=len(preservation),content_hashed_files=sum('sha256' in v for v in preservation.values()),
        changed_or_missing=changed,added=added,large_unrelated_files_checked_by='size and nanosecond mtime',readme_excluded=True))
    if changed:raise AssertionError(f'Original files changed: {changed[:5]}')
    dev=read(out/'train_dev_summary.json');base=dev['policies']['fixed100'];comparison={}
    for name,v in dev['policies'].items():
        comparison[name]=dict(iou_pp=100*(v['metrics']['iou']-base['metrics']['iou']),
            precision_pp=100*(v['metrics']['precision']-base['metrics']['precision']),
            recall_pp=100*(v['metrics']['recall']-base['metrics']['recall']),
            high_density_recall_pp=100*(v['high_density']['recall']-base['high_density']['recall']),
            mac_sparse_proposal_ratio=v['cost']['mac_sparse_proposal']/base['cost']['mac_sparse_proposal'],
            mac_dense_proposal_ratio=v['cost']['mac_dense_proposal']/base['cost']['mac_dense_proposal'],
            mean_publication_wait_ms=v['mean_publication_wait_ms'])
    save(out/'comparison.json',dict(selection=seal['selection'],dev=dev,differences_from_fixed100=comparison,
        paired_bootstrap=paired_bootstrap(records_by_role['train_dev'],seal['selection']['dev_policies'][1:]),
        training_started=False,reserved_sequences_read=False,weights_updated=False,
        cost_caveat='Neural MAC equivalent; actual replay uses dense NumPy, shared interval cache. Feature construction/routing/memory/scalar arithmetic excluded; AC0. Publication waiting time is not measured compute latency.'))
    report=dict(status='passed',sequences=89,unique_published_events=checked_events,policy_event_predictions_checked=checked_predictions,
        all_original_files_unchanged=True,full_fixed100_candidate_probability_and_feature_equality=True,
        selection_sealed_before_dev=True,training_started=False,reserved_sequences_read=False)
    save(out/'audit.json',report)
    print(json.dumps(report),flush=True)
    return report
