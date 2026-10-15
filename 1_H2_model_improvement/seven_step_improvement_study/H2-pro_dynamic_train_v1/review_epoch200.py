"""CPU-only review of saved 200-epoch artifacts. No network inference or updates."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import torch
import review_epoch5 as previous
from dynamic_ann.common import (ROOT,DEFAULT_CACHE,DEFAULT_SUITE,ARMS,FLOAT_SOURCE,INT8_SOURCE,
                               Config,read,sha,run_identity,atomic_json)
from dynamic_ann.data import verify_cache
from dynamic_ann.model import Refiner,load_export
from dynamic_ann.storage import load_checkpoint
from dynamic_ann.synthetic import equal_tree

OUT=ROOT/'results/reviews/epoch200_20261016'
GROUPS=('overall','high_density','ordinary')


def load_data(manifest,arm):
    rows=[];ys=[];cs=[];hs=[];offset=0
    for row in manifest['roles']['train_dev']:
        with np.load(DEFAULT_CACHE/row['arms'][arm]['file'],allow_pickle=False) as z:
            y=z['full_labels'];ids=z['candidate_ids'];c=np.zeros(len(y),bool);c[ids]=True
            rows.append(dict(row=row,lo=offset,hi=offset+len(y),initial=z['initial_probability'].copy()))
            ys.append(y.copy());cs.append(c);hs.append(np.full(len(y),row['metadata']['high_density'],bool));offset+=len(y)
    return dict(y=np.concatenate(ys),candidate=np.concatenate(cs),hd=np.concatenate(hs),sequences=rows)


def recount(data,p,expected):
    assert len(p)==int(data['candidate'].sum()) and np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    full=np.zeros(len(data['y']),np.float32);full[data['candidate']]=p
    totals={g:{k:0 for k in previous.KEYS} for g in GROUPS}
    assert len(expected['sequences'])==10 and expected['threshold']==.35
    for seq,saved in zip(data['sequences'],expected['sequences']):
        sl=slice(seq['lo'],seq['hi']);row=seq['row']
        c=previous.confusion(data['y'][sl],full[sl],data['candidate'][sl])
        m=previous.metrics(c)
        assert dict(sequence=row['sequence'],high_density=row['metadata']['high_density'],**m)==saved
        for group in ('overall','high_density' if row['metadata']['high_density'] else 'ordinary'):
            for k,v in c.items():totals[group][k]+=v
    for g in GROUPS:assert previous.metrics(totals[g])==expected[g],g
    return full


def compact(epoch,evaluation,memory,training=None):
    return dict(epoch=epoch,float=evaluation['float'],int8=evaluation['int8'],
                quantization=evaluation['quantization_difference'],memory=memory,training=training)


def transition(y,hd,a,b):
    a=a>=.35;b=b>=.35;y=y.astype(bool)
    result={}
    for group,mask in (('overall',np.ones(len(y),bool)),('high_density',hd),('ordinary',~hd)):
        v=dict(tp_gained=int((mask&y&~a&b).sum()),tp_lost=int((mask&y&a&~b).sum()),
               fp_added=int((mask&~y&~a&b).sum()),fp_removed=int((mask&~y&a&~b).sum()))
        v.update(net_tp=v['tp_gained']-v['tp_lost'],net_fp=v['fp_added']-v['fp_removed'])
        result[group]=v
    return result


def phase_stats(rows,lo,hi):
    selected=[r for r in rows if lo<=r['epoch']<=hi]
    result=dict(first_epoch=lo,last_epoch=hi)
    for kind in ('float','int8'):
        result[kind]={}
        for group,key in [('overall','iou'),('overall','precision'),('overall','recall'),('high_density','recall')]:
            values=np.array([r[kind][group][key] for r in selected])*100
            result[kind][group+'_'+key]=dict(mean=float(values.mean()),std=float(values.std()),min=float(values.min()),max=float(values.max()))
    result['int8_minus_float_iou_pp']=float(np.mean([100*(r['int8']['overall']['iou']-r['float']['overall']['iou']) for r in selected]))
    return result


def main():
    torch.set_num_threads(2);OUT.mkdir(parents=True,exist_ok=True)
    print('Fingerprinting original 200-epoch run and existing five-epoch snapshot',flush=True)
    before=previous.fingerprint(DEFAULT_SUITE);old_snapshot=previous.OUT/'five_epoch_snapshot'
    old_before=previous.fingerprint(old_snapshot)
    assert old_before==read(previous.OUT/'run_before.json')
    atomic_json(OUT/'run_before.json',before,immutable=True)
    atomic_json(OUT/'five_epoch_snapshot_before.json',old_before,immutable=True)
    manifest=verify_cache(DEFAULT_CACHE);ready=read(DEFAULT_SUITE/'ready.json');identity=ready['identity']
    assert identity==run_identity(DEFAULT_CACHE,Config(),identity['engine'])
    initial=load_checkpoint(DEFAULT_SUITE/'initial.pt')
    assert sha(DEFAULT_SUITE/'initial.pt')==ready['initial_checkpoint_sha256'] and initial['identity']==identity
    assert initial['optimizer']['state']=={} and initial['real_data_optimizer_steps']==0
    equal_tree(initial['model'],torch.load(FLOAT_SOURCE,map_location='cpu',weights_only=True)['state_dict'])
    assert sha(FLOAT_SOURCE)==identity['source_float_sha256'] and sha(INT8_SOURCE)==identity['source_int8_sha256']
    datasets={a:load_data(manifest,a) for a in ARMS}
    for key in ('y','hd'):np.testing.assert_array_equal(datasets['fixed100'][key],datasets['dynamic2000'][key])
    predictions={};arms={};total_logs=0
    for arm in ARMS:
        directory=DEFAULT_SUITE/arm;training=read(directory/'training.json');old=read(old_snapshot/arm/'training.json')
        assert training['identity']==identity and training['global_step']==43800 and len(training['epochs'])==200
        assert training['initial']==old['initial'] and training['epochs'][:5]==old['epochs']
        assert [e['epoch'] for e in training['epochs']]==list(range(1,201))
        best=training['best_epoch'];float_best=max(training['epochs'],key=lambda e:e['evaluation']['float']['overall']['iou'])['epoch']
        high_recall=max(training['epochs'],key=lambda e:e['evaluation']['int8']['high_density']['recall'])['epoch']
        near=[e for e in training['epochs'] if e['evaluation']['int8']['overall']['iou']>=training['epochs'][best-1]['evaluation']['int8']['overall']['iou']-.0015]
        near_recall=max(near,key=lambda e:e['evaluation']['int8']['high_density']['recall'])['epoch']
        selected={0,5,20,50,100,200,best,float_best,high_recall,near_recall}
        logs=[json.loads(l) for l in (directory/'progress.jsonl').read_text().splitlines() if l.strip()];total_logs+=len(logs)
        assert logs[-1]['status']=='complete' and logs[-1]['completed_epochs']==200 and logs[-1]['global_step']==43800
        assert not list(directory.glob('failure*.json')) and all(x['status'] not in ('failed','paused') for x in logs)
        assert all(a['global_step']<=b['global_step'] for a,b in zip(logs,logs[1:]))
        assert sum(x['status']=='epoch_complete' for x in logs)==200
        source_count=manifest['stats']['train_main']['source_events'];candidate_count=manifest['stats']['train_main']['candidates'][arm]
        assert {r['data']['candidate_events'] for r in logs}=={candidate_count}
        assert all(r['data']['per_batch_disk_reads']==r['data']['per_batch_input_h2d_bytes']==0 for r in logs)
        data=datasets[arm];rows=[];changed_summary=[];quantitative={};model=Refiner()
        expected_cost=ready['arms'][arm]['initial']['int8']['cost']
        # Independent recomputation from cached Proposal current/history work.
        for suffix in ('sparse','dense'):
            proposal=sum(seq['row']['arms'][arm]['cost'][f'proposal_current_{suffix}_mac']+
                         seq['row']['arms'][arm]['cost'][f'proposal_history_extra_{suffix}_mac'] for seq in data['sequences'])
            assert expected_cost[f'mac_{suffix}_proposal']==proposal+int(data['candidate'].sum())*4480
        for epoch in range(201):
            entry=None if epoch==0 else training['epochs'][epoch-1]
            evaluation=training['initial'] if entry is None else entry['evaluation']
            npz=directory/'initial_dev_predictions.npz' if epoch==0 else directory/'epochs'/f'epoch_{epoch:03}_dev.npz'
            with np.load(npz,allow_pickle=False) as z:prob={k:z[k].copy() for k in ('float','int8')}
            full={}
            for kind,p in prob.items():
                full[kind]=recount(data,p,evaluation[kind])
                assert evaluation[kind]['cost']==expected_cost
                if epoch==0 and kind=='int8':
                    for seq in data['sequences']:np.testing.assert_array_equal(full[kind][seq['lo']:seq['hi']],seq['initial'])
            diff=np.abs(prob['float']-prob['int8'])
            quant=dict(mean_abs_probability=float(diff.mean()),max_abs_probability=float(diff.max()),
                       decision_flips=int(np.count_nonzero((prob['float']>=.35)!=(prob['int8']>=.35))))
            assert quant==evaluation['quantization_difference']
            if epoch in selected:
                predictions[arm,epoch]=full
                quantitative[epoch]=dict(int8_vs_float=transition(data['y'],data['hd'],full['float'],full['int8']),
                    int8_minus_float_iou_pp=100*(evaluation['int8']['overall']['iou']-evaluation['float']['overall']['iou']),
                    changed_decisions=quant['decision_flips'])
            if epoch:
                assert read(directory/'epochs'/f'epoch_{epoch:03}.json')==entry
                assert entry['global_step']==219*epoch
                t=entry['training'];assert t['source_events']==source_count and t['candidate_events']==candidate_count and t['empty_candidate_batches']==0
                assert np.isfinite([t['mean_loss'],t['gradient_norm_max'],t['max_abs_logit']]).all()
                cp=load_checkpoint(directory/'epochs'/f'epoch_{epoch:03}.pt')
                assert cp['resumable'] and cp['identity']==identity and cp['arm']==arm
                state=cp['state'];assert state['cursor']==dict(epoch=epoch+1,batch=0,phase='train') and state['global_step']==219*epoch
                assert state['history']==training['epochs'][:epoch]
                assert set(cp['rng'])=={'python','numpy','torch_cpu','torch_cuda'}
                assert len(cp['optimizer']['state'])==4 and cp['optimizer']['param_groups'][0]['lr']==1e-5
                for value in cp['optimizer']['state'].values():
                    assert int(value['step'])==219*epoch
                    assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
                changed={k:int(torch.count_nonzero(w-initial['model'][k])) for k,w in cp['model'].items()}
                assert all(v>0 for v in changed.values()) and all(torch.isfinite(w).all() for w in cp['model'].values())
                changed_summary.append(dict(epoch=epoch,changed_weights=changed))
                model.load_state_dict(cp['model'])
                for a,b in zip(model.export_layers(),load_export(directory/'epochs'/f'epoch_{epoch:03}_int8.npz')):
                    for actual,wanted in zip(a,b):np.testing.assert_array_equal(actual,wanted)
                if epoch==best:equal_tree(cp,load_checkpoint(directory/'best.pt'))
                if epoch==200:equal_tree(cp,load_checkpoint(directory/'last.pt'))
                if epoch<=5:
                    for suffix in ('.pt','.json','_int8.npz','_dev.npz'):
                        name=f'epoch_{epoch:03}'+suffix
                        assert sha(directory/'epochs'/name)==sha(old_snapshot/arm/'epochs'/name)
                rows.append(compact(epoch,evaluation,entry['memory'],entry['training']))
            else:rows.append(compact(0,evaluation,ready['arms'][arm]['memory']))
            if epoch%25==0:print(f'{arm}: {epoch}/200 checkpoints, float/int8 predictions and exports checked',flush=True)
        rank=lambda r:(r['int8']['overall']['iou'],r['int8']['overall']['recall'])
        assert max(rows,key=rank)['epoch']==best
        for a,b in zip(load_export(directory/'best_int8.npz'),load_export(directory/'epochs'/f'epoch_{best:03}_int8.npz')):
            for actual,wanted in zip(a,b):np.testing.assert_array_equal(actual,wanted)
        sessions=[];start=logs[0]['updated_unix_s']
        for i,l in enumerate(logs):
            if l['status']=='complete':
                sessions.append(dict(completed_epochs=l['completed_epochs'],seconds=l['updated_unix_s']-start))
                if i+1<len(logs):start=logs[i+1]['updated_unix_s']
        assert [s['completed_epochs'] for s in sessions]==[5,200]
        latest=rows[-1];train_rows=rows[1:]
        arms[arm]=dict(best_epoch=best,float_best_epoch=float_best,high_recall_epoch=high_recall,
            near_best_iou_high_recall_epoch=near_recall,diagnostic_reference_note='Recall/float alternatives are descriptive Dev observations; original int8-best rule is unchanged.',
            rows=rows,selected_epochs=sorted(selected),quantization_at_selected=quantitative,weight_changes=changed_summary,
            mean_windows=[phase_stats(rows,*p) for p in [(1,5),(6,20),(21,50),(51,100),(101,150),(151,200),(181,200)]],
            completed_epochs=200,updates=43800,additional_updates_since5=42705,candidate_visits=200*candidate_count,
            source_events_per_epoch=source_count,candidates_per_epoch=candidate_count,log_records=len(logs),sessions=sessions,
            losses=dict(first=train_rows[0]['training']['mean_loss'],epoch5=rows[5]['training']['mean_loss'],last=latest['training']['mean_loss'],
                        relative_change_percent=100*(latest['training']['mean_loss']/train_rows[0]['training']['mean_loss']-1)),
            times=dict(train_seconds=sum(r['training']['train_seconds'] for r in train_rows),evaluation_seconds=sum(e['evaluation']['evaluation_seconds'] for e in training['epochs']),
                       log_interval_seconds=sum(s['seconds'] for s in sessions)),
            memory_peak={k:max(r['memory'][k] for r in train_rows) for k in ('peak_allocated_bytes','peak_reserved_bytes')},
            gradient_norm_max=max(r['training']['gradient_norm_max'] for r in train_rows),max_abs_logit=max(r['training']['max_abs_logit'] for r in train_rows),
            failures=0,pauses=0,delta_best_to_initial=previous.delta(rows[best]['int8'],rows[0]['int8']),
            delta_best_to5=previous.delta(rows[best]['int8'],rows[5]['int8']),delta_last_to_initial=previous.delta(latest['int8'],rows[0]['int8']),
            delta_last_to5=previous.delta(latest['int8'],rows[5]['int8']))
    frozen=arms['fixed100']['rows'][0]['int8'];dyn=arms['dynamic2000'];fix=arms['fixed100'];dbest=dyn['rows'][dyn['best_epoch']]['int8']
    paired=dict(dynamic_best_to_frozen=previous.delta(dbest,frozen),
        fixed_best_to_frozen=previous.delta(fix['rows'][fix['best_epoch']]['int8'],frozen),
        dynamic_best_to_fixed_best=previous.delta(dbest,fix['rows'][fix['best_epoch']]['int8']),
        dynamic_last_to_fixed_last=previous.delta(dyn['rows'][200]['int8'],fix['rows'][200]['int8']),
        dynamic_best_initial_gap_recovered_percent=100*(dbest['overall']['iou']-dyn['rows'][0]['int8']['overall']['iou'])/(frozen['overall']['iou']-dyn['rows'][0]['int8']['overall']['iou']),
        mac_increase_percent=100*(dbest['cost']['mac_sparse_proposal']/frozen['cost']['mac_sparse_proposal']-1),
        publication_wait_reduction_percent=100*(1-dbest['mean_publication_wait_ms']/frozen['mean_publication_wait_ms']))
    y=datasets['fixed100']['y'];hd=datasets['fixed100']['hd'];f0=predictions['fixed100',0]['int8'];d0=predictions['dynamic2000',0]['int8']
    recovered=y.astype(bool)&(f0<.35)&(d0>=.35);lost=y.astype(bool)&(f0>=.35)&(d0<.35)
    transitions={}
    for epoch in dyn['selected_epochs']:
        final=predictions['dynamic2000',epoch]['int8']
        transitions[epoch]=dict(from_initial=transition(y,hd,d0,final),from_frozen=transition(y,hd,f0,final),
            initial345_recovered_targets_retained=int((recovered&(final>=.35)).sum()),initial467_lost_targets_recovered=int((lost&(final>=.35)).sum()))
    after=previous.fingerprint(DEFAULT_SUITE);assert after==before and previous.fingerprint(old_snapshot)==old_before
    report=dict(status='passed',date='2026-10-16',data_role='full Train-dev (10 sequences); Main training logs; no reserved splits',
        threshold=.35,main_source_events=7154631,dev_source_events=len(y),dev_target_events=int(y.sum()),high_density_sequences=2,
        verification=dict(original_run_files=len(before),original_run_hash_size_mtime_unchanged=True,
            old_snapshot_files=len(old_before),old_snapshot_unchanged=True,first5_epoch_artifacts_match_snapshot=True,
            source_files=len(identity['sources']),source_identity_unchanged=True,feature_payloads_verified=178,
            epoch_checkpoints_and_exports_verified=400,full_prediction_arrays_recounted=804,per_sequence_records_verified=8040,
            quantization_records_recounted=402,log_records=total_logs,full_coverage=True,best_and_last_verified=True,
            model_inference_calls=0,optimizer_updates=0),arms=arms,paired=paired,transitions=transitions,
        next_step='Preserve dynamic epoch118 int8 best, epoch192 float-best and epoch200. First isolate weight/activation quantization with fixed weights and Main diagnostics; then prepare matched original FP32 vs QAT fine-tuning if confirmed. Keep window, frozen Proposal, architecture and threshold fixed. Do not start training. Separate later class-balance/recall objectives.')
    atomic_json(OUT/'review.json',report)
    print(json.dumps(dict(verification=report['verification'],paired=paired),ensure_ascii=False,indent=2),flush=True)
    draw(report)


def draw(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    axes=axs.ravel()
    def avg(x):return np.convolve(np.asarray(x),np.ones(10)/10,mode='valid')
    for arm,color,label in [('fixed100','#2463a8','100 ms control'),('dynamic2000','#c16724','Dynamic 2000')]:
        rows=report['arms'][arm]['rows'];x=np.arange(201)
        series=[np.array([100*r[kind][group][key] for r in rows]) for kind,group,key in
                [('int8','overall','iou'),('float','overall','iou'),('int8','overall','precision'),('int8','overall','recall'),('int8','high_density','recall')]]
        series.append(series[0]-series[1])
        for ax,v in zip(axes,series):
            ax.plot(x,v,color=color,alpha=.25,linewidth=.7)
            ax.plot(x[9:],avg(v),color=color,linewidth=1.8,label=label)
            ax.set(xlabel='Completed epochs',xticks=[0,20,50,100,150,200]);ax.grid(axis='y',alpha=.17)
        best=report['arms'][arm]['best_epoch'];axes[0].scatter([best],[series[0][best]],color=color,s=28,zorder=5)
        axes[0].annotate(f'best {best}',(best,series[0][best]),xytext=(8,6),textcoords='offset points',fontsize=9,color=color)
    titles=['Event IoU: exported int8 (%)','Event IoU: float (%)','Precision: exported int8 (%)',
            'Recall: exported int8 (%)','HD recall: exported int8 (%)','Int8 minus float IoU (pp)']
    frozen=report['arms']['fixed100']['rows'][0]
    for i,(ax,title) in enumerate(zip(axes,titles)):
        ax.set_ylabel(title)
        if i<5:
            kind,group,key=[('int8','overall','iou'),('float','overall','iou'),('int8','overall','precision'),('int8','overall','recall'),('int8','high_density','recall')][i]
            ax.axhline(100*frozen[kind][group][key],color='#555555',ls='--',lw=.8,label='Frozen reference')
        else:ax.axhline(0,color='#555555',ls='--',lw=.8)
    axes[0].legend(loc='upper right',bbox_to_anchor=(1,.76),fontsize=8)
    fig.suptitle('Dynamic-window adaptation: 200 complete epochs per arm\nFull Train-dev, threshold 0.35 | thin: each epoch; bold: trailing 10-epoch mean; dots: int8 best',fontsize=13)
    fig.savefig(OUT/'learning_curves.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
