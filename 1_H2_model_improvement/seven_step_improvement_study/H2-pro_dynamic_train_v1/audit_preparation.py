"""Independent recount of prepared inputs/epoch-zero artifacts; never trains."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import argparse
from pathlib import Path
import numpy as np
import torch
from dynamic_ann.common import (ROOT,DEFAULT_CACHE,DEFAULT_SUITE,ARMS,PRIOR,FLOAT_SOURCE,
                               read,sha,atomic_json,Config,run_identity)
from dynamic_ann.data import verify_cache
from dynamic_ann.model import Refiner,load_export
from dynamic_ann.runtime import setup_device,engine,suite_status
from dynamic_ann.storage import load_checkpoint
from dynamic_ann.synthetic import equal_tree
from dynamic_ann.preservation import preservation_report


def recount(y,p,c):
    positive=y.astype(bool);pred=p>=.35
    return dict(events=len(y),target_events=int(positive.sum()),candidate_events=int(c.sum()),
        candidate_targets=int((positive&c).sum()),tp=int((positive&pred).sum()),
        fp=int((~positive&pred).sum()),fn=int((positive&~pred).sum()),tn=int((~positive&~pred).sum()),
        proposal_fn=int((positive&~c).sum()),refiner_fn=int((positive&c&~pred).sum()))


def audit():
    manifest=verify_cache(DEFAULT_CACHE);suite=DEFAULT_SUITE;ready=read(suite/'ready.json')
    identity=run_identity(DEFAULT_CACHE,Config(),engine(setup_device('cuda')))
    assert ready['status']=='passed' and ready['identity']==identity
    initial=load_checkpoint(suite/'initial.pt')
    assert initial['identity']==identity and initial['real_data_optimizer_steps']==0
    assert initial['optimizer']['state']=={}
    original=torch.load(FLOAT_SOURCE,map_location='cpu',weights_only=True)['state_dict']
    equal_tree(initial['model'],original)
    assert sha(suite/'initial.pt')==ready['initial_checkpoint_sha256']
    assert initial['optimizer']['param_groups'][0]['lr']==1e-5
    dev_predictions={}
    for arm in ARMS:
        assert not (suite/arm/'last.pt').exists(),'Preparation audit cannot be run after formal training'
        assert not (suite/arm/'epochs').exists()
        with np.load(suite/arm/'initial_dev_predictions.npz',allow_pickle=False) as z:
            dev_predictions[arm]={k:z[k].copy() for k in ('float','int8')}
        model=Refiner();model.load_state_dict(initial['model'])
        for pair,expected in zip(load_export(suite/arm/'initial_int8.npz'),model.export_layers()):
            for a,b in zip(pair,expected):np.testing.assert_array_equal(a,b)
    counts={a:{k:{g:{} for g in ('overall','high_density')} for k in ('float','int8')} for a in ARMS}
    offsets={a:0 for a in ARMS};sequence_count=payloads=events=0
    positives={r:{a:0 for a in ARMS} for r in manifest['roles']}
    for role,rows in manifest['roles'].items():
        for row in rows:
            sequence_count+=1;labels=None
            previous=read(PRIOR/role/(Path(row['sequence']).stem+'.json'))
            with np.load(PRIOR/role/previous['payload'],allow_pickle=False) as frozen:
                for arm,policy in ARMS.items():
                    entry=row['arms'][arm]
                    with np.load(DEFAULT_CACHE/entry['file'],allow_pickle=False) as z:
                        y=z['full_labels'];ids=z['candidate_ids'];x=z['features'];p=z['initial_probability']
                        n=row['metadata']['published_events']
                        assert y.dtype==np.uint8 and y.shape==(n,) and np.isin(y,[0,1]).all()
                        assert ids.dtype==np.int32 and (np.diff(ids)>0).all()
                        assert x.shape==(len(ids),36) and x.dtype==np.uint8
                        np.testing.assert_array_equal(ids,np.flatnonzero(frozen[policy+'__candidate']).astype(np.int32))
                        np.testing.assert_array_equal(p,frozen[policy+'__probabilities'])
                        np.testing.assert_array_equal(z['windows'],frozen[policy+'__windows'])
                        if labels is not None:np.testing.assert_array_equal(y,labels)
                        else:labels=y.copy();events+=n
                        positives[role][arm]+=int(y[ids].sum());payloads+=1
                        assert int(y[ids].sum())==entry['candidate_targets']
                        c=np.zeros(n,bool);c[ids]=True
                        if role=='train_dev':
                            start=offsets[arm];end=start+len(ids)
                            for kind in ('float','int8'):
                                part=dev_predictions[arm][kind][start:end]
                                assert np.isfinite(part).all() and ((part>=0)&(part<=1)).all()
                                full=np.zeros(n,np.float32);full[ids]=part
                                if kind=='int8':np.testing.assert_array_equal(full,p)
                                value=recount(y,full,c)
                                groups=['overall']+(['high_density'] if row['metadata']['high_density'] else [])
                                for group in groups:
                                    acc=counts[arm][kind][group]
                                    for k,v in value.items():acc[k]=acc.get(k,0)+v
                            offsets[arm]=end
    for arm in ARMS:
        for kind in ('float','int8'):
            assert offsets[arm]==len(dev_predictions[arm][kind])
            for group,values in counts[arm][kind].items():
                reference=ready['arms'][arm]['initial'][kind][group]
                for k,v in values.items():assert v==reference[k],(arm,kind,group,k)
                for k,v in dict(iou=values['tp']/max(1,values['tp']+values['fp']+values['fn']),
                                precision=values['tp']/max(1,values['tp']+values['fp']),
                                recall=values['tp']/max(1,values['tp']+values['fn'])).items():
                    assert v==reference[k],(arm,kind,group,k)
        old=read(PRIOR/'train_dev_summary.json')['policies'][ARMS[arm]]
        for name in ('mac_sparse_proposal','mac_dense_proposal'):
            assert ready['arms'][arm]['initial']['int8']['cost'][name]==old['cost'][name]
        assert ready['arms'][arm]['initial']['int8']['cost']['refiner_mac']==offsets[arm]*4480
        assert ready['arms'][arm]['initial']['int8']['cost']['ac']==0
    status=suite_status(suite)
    assert all(a['status']=='ready_not_trained' for a in status['arms'].values())
    report=dict(status='passed',sequences=sequence_count,paired_feature_payloads=payloads,
        unique_source_events=events,dev_candidates=offsets,candidate_targets=positives,
        cached_candidates_features_labels_windows_valid=True,all_prior_candidate_probabilities_exact=True,
        full_dev_float_int8_metrics_independently_recounted=True,initial_weights_and_int8_export_exact=True,
        shared_adam_empty=True,shared_rng_stored=True,source_batches_per_epoch=ready['steps_per_epoch'],
        steps_per_arm_at_5_epochs=ready['steps_per_epoch']*5,real_data_optimizer_steps=0,
        formal_status=status,preservation=preservation_report(),ready_sha256=sha(suite/'ready.json'))
    atomic_json(ROOT/'results/checks/preparation_audit.json',report)
    return report


def reuse_snapshot():
    return {str(p):dict(sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            for root in (DEFAULT_CACHE,DEFAULT_SUITE) for p in root.rglob('*')
            if p.is_file() and p.name!='.run.lock'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('audit','snapshot','verify-reuse'),nargs='?',default='audit')
    action=parser.parse_args().action
    if action=='audit':result=audit()
    elif action=='snapshot':
        result=reuse_snapshot();atomic_json(ROOT/'results/checks/reuse_before.json',result,immutable=True)
        result=dict(files=len(result),status='snapshotted')
    else:
        before=read(ROOT/'results/checks/reuse_before.json');after=reuse_snapshot()
        assert before==after,'Prepared cache/run artifact was rewritten or changed'
        result=dict(status='passed',files=len(before),hash_size_mtime_unchanged=True)
        atomic_json(ROOT/'results/checks/reuse_verified.json',result)
    import json
    print(json.dumps(result,ensure_ascii=False,indent=2))
