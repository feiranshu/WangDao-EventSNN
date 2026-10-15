"""Read-only epoch-5 review; CPU recount/export checks, no model inference or training."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[name]='1'
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from dynamic_ann.common import (ROOT,DEFAULT_CACHE,DEFAULT_SUITE,ARMS,FLOAT_SOURCE,INT8_SOURCE,
                               Config,read,sha,run_identity,atomic_json)
from dynamic_ann.data import verify_cache
from dynamic_ann.model import Refiner,load_export
from dynamic_ann.storage import load_checkpoint
from dynamic_ann.synthetic import equal_tree

OUT=ROOT/'results/reviews/epoch5_20261015'
KEYS=('events','target_events','candidate_events','candidate_targets','tp','fp','fn','tn','proposal_fn','refiner_fn')


def fingerprint(root):
    return {p.relative_to(root).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def confusion(y,p,c):
    y=y.astype(bool);pred=p>=.35
    return dict(events=len(y),target_events=int(y.sum()),candidate_events=int(c.sum()),candidate_targets=int((y&c).sum()),
                tp=int((y&pred).sum()),fp=int((~y&pred).sum()),fn=int((y&~pred).sum()),tn=int((~y&~pred).sum()),
                proposal_fn=int((y&~c).sum()),refiner_fn=int((y&c&~pred).sum()))


def metrics(c):
    tp,fp,fn=c['tp'],c['fp'],c['fn']
    return dict(c,iou=tp/max(1,tp+fp+fn),precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),
                f1=2*tp/max(1,2*tp+fp+fn),proposal_recall=c['candidate_targets']/max(1,c['target_events']))


def delta(a,b):
    return dict(iou_pp=100*(a['overall']['iou']-b['overall']['iou']),
                precision_pp=100*(a['overall']['precision']-b['overall']['precision']),
                recall_pp=100*(a['overall']['recall']-b['overall']['recall']),
                hd_recall_pp=100*(a['high_density']['recall']-b['high_density']['recall']),
                tp=a['overall']['tp']-b['overall']['tp'],fp=a['overall']['fp']-b['overall']['fp'])


def main():
    torch.set_num_threads(2)
    OUT.mkdir(parents=True,exist_ok=True)
    suite=DEFAULT_SUITE;before=fingerprint(suite)
    atomic_json(OUT/'run_before.json',before,immutable=True)
    saved=OUT/'five_epoch_snapshot'
    if not saved.exists():shutil.copytree(suite,saved,copy_function=shutil.copy2)
    assert fingerprint(saved)==before,'Snapshot does not match completed five-epoch run'
    manifest=verify_cache(DEFAULT_CACHE);ready=read(suite/'ready.json');identity=ready['identity']
    assert identity==run_identity(DEFAULT_CACHE,Config(),identity['engine'])
    initial=load_checkpoint(suite/'initial.pt')
    assert initial['identity']==identity and not initial['optimizer']['state']
    equal_tree(initial['model'],torch.load(FLOAT_SOURCE,map_location='cpu',weights_only=True)['state_dict'])
    assert sha(suite/'initial.pt')==ready['initial_checkpoint_sha256']
    assert sha(FLOAT_SOURCE)==identity['source_float_sha256'] and sha(INT8_SOURCE)==identity['source_int8_sha256']
    datasets={};main_counts={}
    for arm in ARMS:
        seqs=[];ys=[];cs=[];offset=0;hd=[]
        main_counts[arm]=dict(candidates=sum(x['arms'][arm]['candidate_events'] for x in manifest['roles']['train_main']),
                              targets=sum(x['arms'][arm]['candidate_targets'] for x in manifest['roles']['train_main']))
        main_counts[arm]['background']=main_counts[arm]['candidates']-main_counts[arm]['targets']
        main_counts[arm]['target_fraction']=main_counts[arm]['targets']/main_counts[arm]['candidates']
        for row in manifest['roles']['train_dev']:
            with np.load(DEFAULT_CACHE/row['arms'][arm]['file'],allow_pickle=False) as z:
                y=z['full_labels'];ids=z['candidate_ids'];p=z['initial_probability']
                c=np.zeros(len(y),bool);c[ids]=True
                seqs.append(dict(sequence=row['sequence'],high_density=row['metadata']['high_density'],
                                 lo=offset,hi=offset+len(y),ids=ids.copy(),candidate_initial=p[ids].copy(),row=row))
                ys.append(y.copy());cs.append(c);hd.append(np.full(len(y),row['metadata']['high_density'],bool));offset+=len(y)
        datasets[arm]=dict(y=np.concatenate(ys),candidate=np.concatenate(cs),hd=np.concatenate(hd),sequences=seqs)
    np.testing.assert_array_equal(datasets['fixed100']['y'],datasets['dynamic2000']['y'])
    np.testing.assert_array_equal(datasets['fixed100']['hd'],datasets['dynamic2000']['hd'])
    all_predictions={};arms={};records=0
    for arm in ARMS:
        directory=suite/arm;training=read(directory/'training.json');logs=[json.loads(l) for l in (directory/'progress.jsonl').read_text().splitlines() if l.strip()]
        records+=len(logs)
        assert training['identity']==identity and training['global_step']==1095 and len(training['epochs'])==5
        assert [e['epoch'] for e in training['epochs']]==list(range(1,6))
        assert logs[-1]['status']=='complete' and logs[-1]['global_step']==1095
        assert all(r['status'] not in ('failed','paused') for r in logs) and not list(directory.glob('failure*.json'))
        assert all(a['global_step']<=b['global_step'] for a,b in zip(logs,logs[1:]))
        assert {r['data']['candidate_events'] for r in logs}=={main_counts[arm]['candidates']}
        assert all(r['data']['per_batch_disk_reads']==r['data']['per_batch_input_h2d_bytes']==0 for r in logs)
        data=datasets[arm];ys=data['y'];candidate=data['candidate'];hd=data['hd'];rows=[];states={}
        for epoch in range(6):
            evaluation=training['initial'] if epoch==0 else training['epochs'][epoch-1]['evaluation']
            npz=directory/'initial_dev_predictions.npz' if epoch==0 else directory/'epochs'/f'epoch_{epoch:03}_dev.npz'
            with np.load(npz,allow_pickle=False) as z:prob={k:z[k].copy() for k in ('float','int8')}
            for kind,p in prob.items():
                assert len(p)==candidate.sum() and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
                full=np.zeros(len(ys),np.float32);full[candidate]=p
                all_predictions[arm,epoch,kind]=full
                for group,mask in (('overall',np.ones(len(ys),bool)),('high_density',hd),('ordinary',~hd)):
                    m=metrics(confusion(ys[mask],full[mask],candidate[mask]))
                    assert m==evaluation[kind][group],(arm,epoch,kind,group)
                saved_rows=evaluation[kind]['sequences'];assert len(saved_rows)==len(data['sequences'])
                off=0
                for seq,saved_row in zip(data['sequences'],saved_rows):
                    sl=slice(seq['lo'],seq['hi']);m=metrics(confusion(ys[sl],full[sl],candidate[sl]))
                    assert dict(sequence=seq['sequence'],high_density=seq['high_density'],**m)==saved_row
                    if epoch==0 and kind=='int8':
                        n=len(seq['ids']);np.testing.assert_array_equal(p[off:off+n],seq['candidate_initial']);off+=n
                cost=evaluation[kind]['cost']
                for suffix in ('sparse','dense'):
                    proposal=sum(seq['row']['arms'][arm]['cost'][f'proposal_current_{suffix}_mac']+
                                 seq['row']['arms'][arm]['cost'][f'proposal_history_extra_{suffix}_mac'] for seq in data['sequences'])
                    assert cost[f'mac_{suffix}_proposal']==proposal+int(candidate.sum())*4480
                assert cost['ac']==0 and cost['refiner_mac']==int(candidate.sum())*4480
            if epoch:
                entry=training['epochs'][epoch-1];t=entry['training']
                assert read(directory/'epochs'/f'epoch_{epoch:03}.json')==entry
                assert t['source_events']==7154631 and t['candidate_events']==main_counts[arm]['candidates']
                assert t['empty_candidate_batches']==0 and entry['global_step']==219*epoch
                cp=load_checkpoint(directory/'epochs'/f'epoch_{epoch:03}.pt');states[epoch]=cp
                assert cp['resumable'] and cp['arm']==arm and cp['identity']==identity
                st=cp['state'];assert st['cursor']==dict(epoch=epoch+1,batch=0,phase='train') and st['global_step']==219*epoch
                assert st['history']==training['epochs'][:epoch]
                assert len(cp['optimizer']['state'])==4 and cp['optimizer']['param_groups'][0]['lr']==1e-5
                for value in cp['optimizer']['state'].values():
                    assert int(value['step'])==219*epoch
                    assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
                changed={k:int(torch.count_nonzero(w-initial['model'][k])) for k,w in cp['model'].items()}
                assert all(v>0 for v in changed.values()) and all(torch.isfinite(w).all() for w in cp['model'].values())
                model=Refiner();model.load_state_dict(cp['model'])
                for pair,expected in zip(load_export(directory/'epochs'/f'epoch_{epoch:03}_int8.npz'),model.export_layers()):
                    for actual,wanted in zip(pair,expected):np.testing.assert_array_equal(actual,wanted)
                memory=entry['memory'];loss=t['mean_loss']
            else:changed={};memory=ready['arms'][arm]['memory'];loss=None
            rows.append(dict(epoch=epoch,evaluation=evaluation,mean_loss=loss,memory=memory,
                             memory_stage='preflight' if epoch==0 else 'training_cumulative_peak',changed_parameters=changed))
        rank=lambda r:(r['evaluation']['int8']['overall']['iou'],r['evaluation']['int8']['overall']['recall'])
        best=max(rows,key=rank)['epoch'];assert best==training['best_epoch']
        equal_tree(load_checkpoint(directory/'last.pt'),states[5]);equal_tree(load_checkpoint(directory/'best.pt'),states[best])
        for pair,expected in zip(load_export(directory/'best_int8.npz'),load_export(directory/'epochs'/f'epoch_{best:03}_int8.npz')):
            for actual,wanted in zip(pair,expected):np.testing.assert_array_equal(actual,wanted)
        losses=[r['mean_loss'] for r in rows[1:]]
        arms[arm]=dict(best_epoch=best,last_epoch=5,epochs=rows,updates=1095,
            candidate_events_per_epoch=main_counts[arm]['candidates'],candidate_visits=5*main_counts[arm]['candidates'],
            loss_relative_change_percent=100*(losses[-1]/losses[0]-1),
            training_memory={k:max(r['memory'][k] for r in rows[1:]) for k in ('peak_allocated_bytes','peak_reserved_bytes')},
            log_records=len(logs),log_wall_seconds=logs[-1]['updated_unix_s']-logs[0]['updated_unix_s'],
            train_seconds=sum(e['training']['train_seconds'] for e in training['epochs']),
            evaluation_seconds=sum(e['evaluation']['evaluation_seconds'] for e in training['epochs']),
            load_seconds=logs[-1]['data']['load_seconds'],gradient_norm_max=max(e['training']['gradient_norm_max'] for e in training['epochs']),
            max_abs_logit=max(e['training']['max_abs_logit'] for e in training['epochs']),failures=0,pauses=0,
            delta_best_to_initial=delta(rows[best]['evaluation']['int8'],rows[0]['evaluation']['int8']),
            delta_last_to_initial=delta(rows[5]['evaluation']['int8'],rows[0]['evaluation']['int8']))
    frozen=arms['fixed100']['epochs'][0]['evaluation']['int8']
    dy=arms['dynamic2000'];fx=arms['fixed100'];d0=dy['epochs'][0]['evaluation']['int8'];d5=dy['epochs'][5]['evaluation']['int8']
    paired=dict(dynamic_best_to_frozen=delta(d5,frozen),
                dynamic_last_to_fixed_last=delta(d5,fx['epochs'][5]['evaluation']['int8']),
                dynamic_best_to_fixed_best=delta(d5,fx['epochs'][fx['best_epoch']]['evaluation']['int8']),
                dynamic_iou_gain_minus_fixed_iou_gain_pp=dy['delta_last_to_initial']['iou_pp']-fx['delta_last_to_initial']['iou_pp'],
                dynamic_mac_increase_percent=100*(d5['cost']['mac_sparse_proposal']/frozen['cost']['mac_sparse_proposal']-1),
                initial_iou_gap_recovered_percent=100*(d5['overall']['iou']-d0['overall']['iou'])/(frozen['overall']['iou']-d0['overall']['iou']),
                wait_reduction_percent=100*(1-d5['mean_publication_wait_ms']/frozen['mean_publication_wait_ms']))
    y=datasets['fixed100']['y'].astype(bool);hd=datasets['fixed100']['hd']
    f0=all_predictions['fixed100',0,'int8']>=.35;d0p=all_predictions['dynamic2000',0,'int8']>=.35;d5p=all_predictions['dynamic2000',5,'int8']>=.35
    recovered=y&~f0&d0p;lost=y&f0&~d0p
    transitions=dict(initial_new_targets=int(recovered.sum()),initial_new_targets_retained=int((recovered&d5p).sum()),
        initial_old_targets_lost=int(lost.sum()),initial_old_targets_recovered_after_training=int((lost&d5p).sum()),
        new_vs_frozen_targets_final=int((y&~f0&d5p).sum()),lost_vs_frozen_targets_final=int((y&f0&~d5p).sum()),
        dynamic_training_tp_gained=int((y&~d0p&d5p).sum()),dynamic_training_tp_lost=int((y&d0p&~d5p).sum()),
        dynamic_training_fp_removed=int((~y&d0p&~d5p).sum()),dynamic_training_fp_added=int((~y&~d0p&d5p).sum()))
    sequences=[]
    for seq in datasets['fixed100']['sequences']:
        mask=np.zeros(len(y),bool);mask[seq['lo']:seq['hi']]=True
        sequences.append(dict(sequence=seq['sequence'],high_density=seq['high_density'],
            tp_change=int((y&d5p&mask).sum()-(y&d0p&mask).sum()),
            fp_change=int((~y&d5p&mask).sum()-(~y&d0p&mask).sum()),
            targets=int((y&mask).sum())))
    summary=dict(status='passed',scope='Existing full Train-dev predictions and completed training logs/checkpoints, CPU-only; no inference/training',
        data=dict(main_sequences=79,main_source_events=7154631,dev_sequences=10,dev_source_events=len(y),dev_targets=int(y.sum()),
                  high_density_sequences=2,high_density_targets=int((y&hd).sum()),main_candidate_composition=main_counts),
        verification=dict(source_identity_matches=True,all_178_cache_payloads_hash_verified=True,full_predictions_recounted=24,
            per_sequence_metric_records_verified=240,epoch_checkpoints_verified=10,initial_and_best_last_identity_verified=True,
            all_epochs_full_coverage=True,each_optimizer_steps=1095,log_records=records,formal_source_files=len(identity['sources']),
            real_inference_calls=0,real_optimizer_updates=0),arms=arms,paired=paired,transitions=transitions,
        dynamic_training_per_sequence=sequences,
        recommendation='Continue both arms unchanged to cumulative 20 epochs from last.pt, then review at 10/20; current dynamic IoU improves monotonically, but precision gain costs recall. Do not treat current results as replacing frozen ANN. If recall remains weak, independently test class balance or fixed-weight Main threshold diagnosis; do not mix changes now.',
        preservation=dict(original_run_files=len(before),bytes=sum(v['bytes'] for v in before.values()),
                          five_epoch_snapshot=str(saved),all_original_run_hashes_sizes_mtimes_unchanged=fingerprint(suite)==before))
    assert summary['preservation']['all_original_run_hashes_sizes_mtimes_unchanged']
    atomic_json(OUT/'review.json',summary)
    draw(summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='arms'},ensure_ascii=False,indent=2))
    for a,v in arms.items():
        print(a,json.dumps({k:w for k,w in v.items() if k!='epochs'},ensure_ascii=False))


def draw(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,3,figsize=(13.4,4.2),layout='constrained')
    for arm,color,label in [('fixed100','#2463a8','100 ms control'),('dynamic2000','#c16724','Dynamic 2000')]:
        rows=report['arms'][arm]['epochs'];x=[r['epoch'] for r in rows]
        for ax,(group,key,title) in zip(axs,[('overall','iou','Event IoU (%)'),('overall','precision','Precision (%)'),('high_density','recall','High-density recall (%)')]):
            ax.plot(x,[100*r['evaluation']['int8'][group][key] for r in rows],color=color,marker='o',label=label)
            ax.plot(x,[100*r['evaluation']['float'][group][key] for r in rows],color=color,linestyle=':',alpha=.65)
            ax.set(xlabel='Completed epochs',ylabel=title,xticks=x);ax.grid(axis='y',alpha=.17)
    for ax,(group,key) in zip(axs,[('overall','iou'),('overall','precision'),('high_density','recall')]):
        base=report['arms']['fixed100']['epochs'][0]['evaluation']['int8'][group][key]*100
        ax.axhline(base,color='#555555',linestyle='--',linewidth=1,label='Frozen H2-pro (int8)')
    axs[0].legend(fontsize=8,loc='center right')
    fig.suptitle('Dynamic-window adaptation: 5 complete epochs per arm\nTrain-dev; fixed threshold 0.35 | solid: exported int8, dotted: float',fontsize=12)
    fig.savefig(OUT/'learning_curves.png',dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
