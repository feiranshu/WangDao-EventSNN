"""Independent Main seal, Dev replay, and resumable actual spike-add checks."""
from pathlib import Path
from ..h2pro_adaptive_t_v2.core import Policy
from ..h2pro_adaptive_t_v2.scoring import score, policy_id
from ..h2pro_adaptive_t_v2.experiment import bundle, verify_policy
from ..h2pro_adaptive_t_v1.experiment import emit, ann_reference
from ..h2pro_r1_train_v1.data import manifest
from .common import (DEFAULT_RUN,DEFAULT_CACHE,SOURCE,REFERENCE,PRIMARY_V2,EXTRA_V2,
    identity,output_path,lock_directory,read_json,sha256,atomic_json,sealed_json)
from .selection import select_main
from .analysis import paired_analysis


def seal_main(directory, selection, receipts):
    sealed_json(directory/'selection.json',selection)
    sealed_json(directory/'main_seal.json',dict(role='train_main',dev_used_for_selection=False,
        identity_sha256=sha256(directory/'identity.json'),selection_sha256=sha256(directory/'selection.json'),
        traces=receipts))


def require_seal(directory):
    if not (directory/'main_seal.json').exists():
        raise ValueError('Seal Main before reading any Dev prefixes')
    seal=read_json(directory/'main_seal.json');selection=read_json(directory/'selection.json')
    if (seal['role']!='train_main' or selection['role']!='train_main' or seal['dev_used_for_selection']
            or selection['dev_used_for_selection'] or sha256(directory/'identity.json')!=seal['identity_sha256']
            or sha256(directory/'selection.json')!=seal['selection_sha256']):
        raise ValueError('Main seal changed')
    for row in seal['traces']:
        if not row['relative'].startswith('traces/train_main/') or sha256(REFERENCE/row['relative'])!=row['sha256']:
            raise ValueError('Original Main prefix changed')
    return selection


def run(action='run',directory=DEFAULT_RUN,cache=DEFAULT_CACHE,batch_size=8192):
    directory,cache=output_path(directory),Path(cache).resolve()
    if action=='status':
        return read_json(directory/'progress.json') if (directory/'progress.json').exists() else {'stage':'not_started'}
    if action not in ('main','dev','run'):raise ValueError('Unknown action')
    if type(batch_size) is not int or batch_size<1:raise ValueError('Positive batch size required')
    for old in (SOURCE,REFERENCE,PRIMARY_V2,EXTRA_V2,cache):
        old=old.resolve()
        if directory==old or directory in old.parents or old in directory.parents:
            raise ValueError('Use an independent output directory')
    ident,layers=identity(cache)
    with lock_directory(directory,ident):
        if (directory/'comparison.json').exists():
            require_seal(directory);done=read_json(directory/'comparison.json')
            for name,digest in done['artifact_sha256'].items():
                if sha256(directory/name)!=digest:raise ValueError('Completed artifact changed')
            return dict(stage='complete',reused=True,comparison=str(directory/'comparison.json'))
        m=manifest(cache)
        if action in ('main','run'):
            if not (directory/'main_seal.json').exists():
                data,receipts=bundle(directory,cache,m['roles']['train_main'])
                selection=select_main(data,lambda v:emit(directory,v))
                if selection['split']!=read_json(REFERENCE/'selection.json')['split']:
                    raise AssertionError('Original Main split must remain unchanged')
                seal_main(directory,selection,receipts)
                emit(directory,dict(stage='main_sealed',selected={k:{n:r['policy'] for n,r in v['selected'].items()}
                                                               for k,v in selection['arms'].items()}))
                del data
            else:selection=require_seal(directory)
            if action=='main':return dict(stage='main_sealed',selection=str(directory/'selection.json'))
        selection=require_seal(directory)
        original=read_json(REFERENCE/'comparison.json');expected=original['artifact_sha256']
        data,_=bundle(directory,cache,m['roles']['train_dev'],expected)
        policies=selection['evaluation_policies'];results={};verified={}
        fixed=score(data,fixed_step=16,details=True);ann=ann_reference(cache,m['roles']['train_dev'])
        prior=read_json(PRIMARY_V2/'comparison.json');extra=read_json(EXTRA_V2/'comparison.json')
        if ann!=prior['frozen_ann']:raise AssertionError('Frozen ANN changed')
        if any(fixed[k]!=prior['fixed_t16'][k] for k in ('overall','ordinary','high_density','cost','sequences')):
            raise AssertionError('Frozen T16 changed')
        for name,spec in policies.items():
            policy=Policy(**spec);key=policy_id(policy)
            if key not in verified:
                verified[key]=verify_policy(directory,cache,m['roles']['train_dev'],layers,policy,expected,batch_size)
            results[name]=score(data,policy,details=True);results[name]['full_explicit_verification']=key
        for current,old in ((results['shared'],prior['shared_control']),(results['combined'],extra['adaptive']['accuracy'])):
            if any(current[k]!=old[k] for k in ('overall','ordinary','high_density','cost','sequences')):
                raise AssertionError('Old shared/combined control changed')
        analysis=paired_analysis(data,policies)
        artifacts={p.relative_to(directory).as_posix():sha256(p) for p in directory.rglob('*') if p.is_file()
                   and p.suffix in ('.json','.npz') and p.name not in ('comparison.json','progress.json')}
        out=dict(status='complete',identity=ident,selection_sha256=sha256(directory/'selection.json'),
                 fixed_t16=fixed,frozen_ann=ann,adaptive=results,paired_errors=analysis,verification=verified,
                 artifact_sha256=artifacts,controls_reproduced_exactly=True,weights_updated=False,
                 training_started=False,reserved_splits_read=False,data_role='train_dev')
        atomic_json(directory/'comparison.json',out)
        emit(directory,dict(stage='complete',comparison=str(directory/'comparison.json'),
                            unique_policies=len(verified),weights_updated=False,training_started=False))
        return dict(stage='complete',reused=False,comparison=str(directory/'comparison.json'))
