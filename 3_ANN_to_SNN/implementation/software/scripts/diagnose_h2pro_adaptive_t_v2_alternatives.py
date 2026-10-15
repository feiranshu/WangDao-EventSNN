"""Supplemental Main-chosen distinct gate diagnostics; preserve the primary seal.

This supplement was motivated by the primary Main-only selection returning the
shared control. It is exploratory after Main inspection, sealed BEFORE this
round's Dev validation, and never replaces the primary winner retrospectively.
"""
from pathlib import Path
import argparse,json,sys,time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wangdao.h2pro_adaptive_t_v2.common import (DEFAULT_RUN as PRIMARY,DEFAULT_CACHE,REFERENCE,SOURCE,DESIGN,
    identity as primary_identity,output_path,lock_directory,read_json,sha256,atomic_json,sealed_json)
from wangdao.h2pro_adaptive_t_v2.core import Policy,CONTROL
from wangdao.h2pro_adaptive_t_v2.scoring import score,subset,rank,policy_id
from wangdao.h2pro_adaptive_t_v2.experiment import require_seal,bundle,verify_policy
from wangdao.h2pro_adaptive_t_v1.experiment import emit
from wangdao.h2pro_r1_train_v1.data import manifest

DEFAULT_RUN=PRIMARY.parent/'h2pro_adaptive_t_v2_alternatives_s42_ep159'
RULE='Among the already shortlisted non-shared policies, select the best per original Main-validation objective; preserve primary/control'


def read_seal(run):
    s=read_json(run/'selection.json');seal=read_json(run/'main_seal.json')
    if (sha256(run/'identity.json')!=seal['identity_sha256'] or sha256(run/'selection.json')!=seal['selection_sha256']
            or sha256(PRIMARY/'main_seal.json')!=seal['primary_main_seal_sha256'] or s['dev_used_for_selection']):
        raise ValueError('Supplemental Main seal changed')
    require_seal(PRIMARY)
    return s


def run(action='run',directory=DEFAULT_RUN,batch_size=8192):
    directory=output_path(directory)
    if action=='status':return read_json(directory/'progress.json') if (directory/'progress.json').exists() else {'stage':'not_started'}
    if action not in ('main','dev','run'):raise ValueError('Unknown action')
    if type(batch_size) is not int or batch_size<1:raise ValueError('Positive batch size')
    for old in (PRIMARY,REFERENCE,SOURCE,DEFAULT_CACHE):
        if directory==old or directory in old.parents or old in directory.parents:raise ValueError('Use independent supplement output')
    primary,layers=primary_identity(DEFAULT_CACHE)
    if primary!=read_json(PRIMARY/'identity.json'):raise ValueError('Primary identity changed')
    selection=require_seal(PRIMARY)
    ident=dict(primary_identity=primary,primary_selection_sha256=sha256(PRIMARY/'selection.json'),
        primary_main_seal_sha256=sha256(PRIMARY/'main_seal.json'),script_sha256=sha256(Path(__file__)),
        rule=RULE,scope='supplemental exploratory Main diagnosis, not replacement primary selection',
        weights_updated=False,training_started=False)
    with lock_directory(directory,ident):
        if (directory/'comparison.json').exists():
            read_seal(directory);done=read_json(directory/'comparison.json')
            for name,h in done['artifact_sha256'].items():
                if sha256(directory/name)!=h:raise ValueError('Completed diagnostic changed')
            return dict(stage='complete',reused=True,comparison=str(directory/'comparison.json'))
        m=manifest(DEFAULT_CACHE)
        if action in ('main','run'):
            if not (directory/'main_seal.json').exists():
                main,_=bundle(directory,DEFAULT_CACHE,m['roles']['train_main'])
                val=subset(main,selection['split']['validation']);baseline=score(val,fixed_step=16)
                candidates={i:r for i,r in selection['validation_candidates'].items()
                    if (r['policy']['negative4'],r['policy']['positive4'])!=(r['policy']['negative8'],r['policy']['positive8'])}
                chosen={}
                for name,coef in DESIGN['objectives'].items():
                    key=max(candidates,key=lambda i:(rank(candidates[i],coef,baseline),-int(i)))
                    item=candidates[key];policy=Policy(**item['policy'])
                    chosen[name]=dict(grid_index=int(key),policy=policy.as_dict(),policy_id=policy_id(policy),coefficient=coef,
                        fit=selection['fit_shortlist'][key],validation=item,full_main=score(main,policy,details=True))
                extra=dict(role='train_main',rule=RULE,primary_winner_unchanged=True,
                    registered_after_primary_main_inspection=True,dev_used_for_selection=False,
                    parent_dev_already_complete_on_selection=(PRIMARY/'comparison.json').exists(),
                    eligible_grid_indices=sorted(map(int,candidates)),selected=chosen)
                sealed_json(directory/'selection.json',extra)
                sealed_json(directory/'main_seal.json',dict(identity_sha256=sha256(directory/'identity.json'),
                    selection_sha256=sha256(directory/'selection.json'),primary_main_seal_sha256=sha256(PRIMARY/'main_seal.json')))
                emit(directory,dict(stage='main_sealed',selected={k:v['policy'] for k,v in chosen.items()},
                                    primary_dev_complete=extra['parent_dev_already_complete_on_selection']))
                del main,val
            if action=='main':return dict(stage='main_sealed',selection=str(directory/'selection.json'))
        extra=read_seal(directory)
        original=read_json(REFERENCE/'comparison.json');expected=original['artifact_sha256']
        dev,_=bundle(directory,DEFAULT_CACHE,m['roles']['train_dev'],expected)
        fixed=score(dev,fixed_step=16,details=True);shared=score(dev,CONTROL,details=True)
        for result,old in ((fixed,original['fixed']['16']),(shared,original['adaptive']['balanced'])):
            if any(result[k]!=old[k] for k in ('overall','ordinary','high_density','cost','sequences')):
                raise AssertionError('Supplement controls differ from original frozen experiment')
        results={};verified={}
        for name,item in extra['selected'].items():
            policy=Policy(**item['policy']);key=policy_id(policy)
            if key not in verified:verified[key]=verify_policy(directory,DEFAULT_CACHE,m['roles']['train_dev'],layers,policy,expected,batch_size)
            results[name]=score(dev,policy,details=True);results[name]['full_explicit_verification']=key
        artifacts={p.relative_to(directory).as_posix():sha256(p) for p in directory.rglob('*') if p.is_file()
                   and p.suffix in ('.json','.npz') and p.name not in ('progress.json','comparison.json')}
        out=dict(status='complete',scope=ident['scope'],identity=ident,selection_sha256=sha256(directory/'selection.json'),
            fixed_t16=fixed,shared_control=shared,adaptive=results,verification=verified,artifact_sha256=artifacts,
            controls_reproduced_exactly=True,weights_updated=False,training_started=False,reserved_splits_read=False)
        atomic_json(directory/'comparison.json',out)
        emit(directory,dict(stage='complete',comparison=str(directory/'comparison.json'),weights_updated=False,training_started=False))
        return dict(stage='complete',reused=False,comparison=str(directory/'comparison.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('run','main','dev','status'),default='run')
    p.add_argument('--run-directory',type=Path,default=DEFAULT_RUN)
    p.add_argument('--batch-size',type=int,default=8192)
    a=p.parse_args();print(json.dumps(run(a.action,a.run_directory,a.batch_size),ensure_ascii=False,indent=2))
