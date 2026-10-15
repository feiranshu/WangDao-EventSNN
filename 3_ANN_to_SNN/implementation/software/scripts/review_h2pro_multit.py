"""CPU-only review of saved multi-T predictions/checkpoints; never trains or reads Test."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'software'))
import numpy as np
import torch
from wangdao.h2pro_longtrain_v1.storage import load
from wangdao.h2pro_multit_v1.common import digest, sources
from wangdao.h2pro_multit_v1.model import R1Multi
from wangdao.h2pro_multit_v1.evaluation import load_export


def read(path):
    return json.loads(path.read_text('utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def probabilities(raw):
    logits = raw.reshape(-1, 2, 2).max(axis=2)
    ex = np.exp((logits-logits.max(axis=1, keepdims=True)).astype(np.float64))
    return (ex[:, 1]/ex.sum(axis=1)).astype(np.float32)


def recount(y, pred):
    y = y.astype(bool)
    tp, fp, fn, tn = (int(v.sum()) for v in (y & pred, ~y & pred, y & ~pred, ~y & ~pred))
    divide = lambda numerator, denominator: numerator/denominator if denominator else None
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, iou=divide(tp,tp+fp+fn), precision=divide(tp,tp+fp),
                recall=divide(tp,tp+fn), f1=divide(2*tp,2*tp+fp+fn))


def compact(e):
    return dict(iou=e['overall']['iou'], precision=e['overall']['precision'], recall=e['overall']['recall'],
                hd_recall=e['high_density']['recall'], tp=e['overall']['tp'], fp=e['overall']['fp'],
                hd_tp=e['high_density']['tp'], mac=e['cost']['inference_equivalent_mac_sparse_proposal'],
                ac=e['cost']['inference_equivalent_ac'], average_steps=e['cost']['average_steps'])


def modes(e):
    return {**e['fixed'], 'adaptive': e['adaptive']}


def changes(y, pred, previous, hd):
    return {group: dict(recovered=int((y[mask].astype(bool) & pred[mask] & ~previous[mask]).sum()),
                        newly_missed=int((y[mask].astype(bool) & ~pred[mask] & previous[mask]).sum()),
                        added_fp=int((~y[mask].astype(bool) & pred[mask] & ~previous[mask]).sum()),
                        removed_fp=int((~y[mask].astype(bool) & ~pred[mask] & previous[mask]).sum()))
            for group, mask in (('overall', np.ones(len(y), bool)), ('high_density', hd), ('ordinary', ~hd))}


def audit_predictions(path, evaluation, seqs, y, candidates, hd):
    with np.load(path, allow_pickle=False) as z:
        raw, spikes = z['raw'], z['spikes']
    n = int(candidates.sum())
    assert raw.shape == (3, n, 4) and spikes.shape == (3, n)
    assert np.isfinite(raw).all() and (spikes >= 0).all()
    p = np.stack([probabilities(v) for v in raw])
    exit_ix = np.full(n, 2, np.int64)
    exit_ix[(p[0] <= .01) | (p[0] >= .99)] = 0
    exit_ix[(exit_ix == 2) & ((p[1] <= .10) | (p[1] >= .95))] = 1
    all_preds = {}
    proposal = sum(s['row']['proposal_sparse_mac'] for s in seqs)
    for mode, ix in [(str(t), np.full(n, i)) for i, t in enumerate((4, 8, 16))] + [('adaptive', exit_ix)]:
        pred = np.zeros(len(y), bool)
        pred[candidates] = p[ix, np.arange(n)] >= .35
        e = modes(evaluation)[mode]
        for group, mask in (('overall', np.ones(len(y), bool)), ('high_density', hd), ('ordinary', ~hd)):
            observed = recount(y[mask], pred[mask])
            assert all(e[group][k] == v for k, v in observed.items()), (path, mode, group)
        for s, reference in zip(seqs, e['sequences']):
            assert s['row']['sequence'] == reference['sequence']
            observed = recount(y[s['lo']:s['hi']], pred[s['lo']:s['hi']])
            assert all(reference['counts'][k] == observed[k] for k in ('tp', 'fp', 'fn', 'tn'))
        readouts = n if mode != 'adaptive' else n + int((ix >= 1).sum()) + int((ix == 2).sum())
        expected_mac = proposal + n*2304 + readouts*128
        expected_ac = int(spikes[ix, np.arange(n)].sum())*32
        assert expected_mac == e['cost']['inference_equivalent_mac_sparse_proposal']
        assert expected_ac == e['cost']['inference_equivalent_ac']
        assert float(np.array([4, 8, 16])[ix].mean()) == e['cost']['average_steps']
        assert e['cost']['exit_counts'] == {str(t): int((ix == i).sum()) for i, t in enumerate((4, 8, 16))}
        all_preds[mode] = pred
    return all_preds


def review(suite, epochs, output):
    torch.set_num_threads(2)
    started = time.time()
    directory = ROOT / f'results/runs/h2pro_multit_v1_epoch{epochs}_review_20261016'
    directory.mkdir(parents=True, exist_ok=True)
    before = {p.relative_to(suite).as_posix(): dict(sha256=sha(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
              for p in sorted(suite.rglob('*')) if p.is_file()}
    write(directory/'preservation_before.json', before)
    ready = read(suite/'ready.json'); identity = read(suite/'identity.json')
    assert ready['identity'] == identity and identity['sources'] == sources()
    source = Path(identity['paths']['source']); cache = Path(identity['paths']['cache'])
    origin = load(source/'epochs/epoch_159.pt')
    for name, expected in identity['source_files_sha256'].items():
        assert sha(source/name) == expected, name
    manifest = read(cache/'manifest.json')
    assert set(manifest['roles']) == {'train_main', 'train_dev'}
    seqs, ys, cs, hds, teachers = [], [], [], [], []
    offset = 0
    for role, rows in manifest['roles'].items():
        for row in rows:
            assert sha(cache/row['cache_file']) == row['cache_sha256']
            if role != 'train_dev':
                continue
            with np.load(cache/row['cache_file'], allow_pickle=False) as z:
                yy, cc = z['labels'], z['candidate']
                ys.append(yy.copy()); cs.append(cc.copy()); hds.append(np.full(len(yy), row['high_density'], bool))
                teachers.append(z['teacher'].copy())
            seqs.append(dict(row=row, lo=offset, hi=offset+len(yy))); offset += len(yy)
    y, c, hd = map(np.concatenate, (ys, cs, hds))
    assert len(y) == 1221626 and int(c.sum()) == 438966 and int(y.sum()) == 30254
    ann_pred = np.zeros(len(y), bool); ann_pred[c] = probabilities(np.concatenate(teachers)) >= .35
    for group, mask in (('overall', np.ones(len(y), bool)), ('ordinary', ~hd), ('high_density', hd)):
        assert all(ready['frozen_ann'][group][k] == v for k, v in recount(y[mask], ann_pred[mask]).items())
    initial_preds = audit_predictions(suite/'initial_dev_predictions.npz', ready['initial_student'], seqs, y, c, hd)
    result = dict(schema_version=1, status='complete', experiment_id=f'h2pro_multit_v1_epoch{epochs}_review_20261016',
                  stage='saved_training_review', data_role='train_main/train_dev', real_updates_by_review=0,
                  reserved_splits_read=False, source_epoch=159, initial_student=ready['initial_student'],
                  frozen_ann=ready['frozen_ann'], arms={}, paired_endpoints={}, protocol=identity['protocol'])
    histories, predictions, logs_all = {}, {}, []
    model = R1Multi()
    for arm in ('control', 'multit'):
        folder = suite/arm; training = read(folder/'training.json'); history = training['epochs'][:epochs]
        assert len(history) == epochs
        for name in ('initial', 'best', 'best_adaptive'):
            checkpoint = load(folder/(name+'.pt'))
            if name == 'initial' or checkpoint['state']['best_epoch' if name == 'best' else 'best_adaptive_epoch'] == 0:
                assert all(digest(checkpoint[k]) == digest(origin[k]) for k in ('model', 'optimizer', 'rng'))
        predictions[arm] = {}; replay = []
        for added, entry in enumerate(history, 1):
            base = folder/'epochs'/f'added_{added:03}'
            assert read(base.with_suffix('.json')) == entry
            assert entry['effective_epoch'] == 159+added and entry['global_step'] == 43089+271*added
            assert entry['training']['events'] == 2219864
            assert sha(Path(str(base)+'_predictions.npz')) == entry['predictions_sha256']
            pred = audit_predictions(Path(str(base)+'_predictions.npz'), entry['evaluation'], seqs, y, c, hd)
            predictions[arm][added] = pred
            checkpoint = load(base.with_suffix('.pt'))
            assert checkpoint['state']['history'][-1] == entry
            assert checkpoint['state']['cursor'] == dict(epoch=added+1, batch=0, phase='train')
            assert all(float(s['step']) == entry['global_step'] for s in checkpoint['optimizer']['state'].values())
            assert all(g['lr'] == 1e-4 for g in checkpoint['optimizer']['param_groups'])
            model.load_state_dict(checkpoint['model']); model.verify_frozen_scales()
            for actual, exported in zip(model.export_layers(), load_export(Path(str(base)+'_int8.npz'))):
                for a, b in zip(actual, exported):
                    np.testing.assert_array_equal(a, b)
            if arm == 'control':
                old_path = source/'epochs'/f'epoch_{159+added:03}.pt'
                if old_path.exists():
                    old = load(old_path)
                    assert all(digest(checkpoint[k]) == digest(old[k]) for k in ('model', 'optimizer', 'rng'))
                    assert entry['historical_control_replay']['all_exact']
                    replay.append(159+added)
        last = load(folder/'last.pt'); final = load(folder/'epochs'/f'added_{epochs:03}.pt')
        assert all(digest(last[k]) == digest(final[k]) for k in ('model', 'optimizer', 'rng', 'state'))
        logs = [json.loads(line) for line in (folder/'progress.jsonl').read_text('utf-8').splitlines()]
        logs_all += logs
        assert not list(folder.glob('failure*.json'))
        assert not any(l['status'] == 'failed' for l in logs)
        for l in logs:
            assert all(np.isfinite(l['totals'][k]) for k in ('gradient_norm_max', 'logit_absmax', 'train_seconds'))
        current, active_seconds = None, 0.
        for l in logs:
            if current is None and l['status'] == 'running':
                current = l['updated_unix_s']
            if l['status'] in ('complete', 'paused', 'failed') and current is not None:
                active_seconds += l['updated_unix_s'] - current; current = None
        best = {}
        for mode in ('4', '8', '16', 'adaptive'):
            best_entry = max(history, key=lambda e: (modes(e['evaluation'])[mode]['overall']['iou'], modes(e['evaluation'])[mode]['overall']['recall']))
            best[mode] = dict(added_epoch=best_entry['added_epoch'], **compact(modes(best_entry['evaluation'])[mode]))
        result['arms'][arm] = dict(history=history, best_trained_by_mode=best,
            best_including_source_fixed=training['best_added_epoch'], best_including_source_adaptive=training['best_adaptive_added_epoch'],
            endpoint={k:compact(v) for k,v in modes(history[-1]['evaluation']).items()},
            epoch5_vs_source_events={m:changes(y,predictions[arm][epochs][m],initial_preds[m],hd) for m in initial_preds},
            epoch5_vs_ann_events={m:changes(y,predictions[arm][epochs][m],ann_pred,hd) for m in initial_preds},
            logs=dict(records=len(logs),statuses=sorted({l['status'] for l in logs}),active_wall_seconds=active_seconds,
                training_seconds=sum(e['training']['train_seconds'] for e in history),
                evaluation_seconds=sum(e['evaluation']['inference_seconds'] for e in history),
                max_gradient_norm=max(l['totals']['gradient_norm_max'] for l in logs),
                max_abs_logit=max(l['totals']['logit_absmax'] for l in logs)),
            memory_peak={k:max(e['memory'][k] for e in history) for k in history[0]['memory']},
            historical_control_replay_epochs=replay)
        histories[arm] = history
    for m in initial_preds:
        a,b = (result['arms'][arm]['endpoint'][m] for arm in ('control','multit'))
        result['paired_endpoints'][m] = dict(delta_pp={k:100*(b[k]-a[k]) for k in ('iou','precision','recall','hd_recall')},
            ac_relative_percent=100*(b['ac']/a['ac']-1), mac_relative_percent=100*(b['mac']/a['mac']-1),
            event_changes=changes(y,predictions['multit'][epochs][m],predictions['control'][epochs][m],hd))
    for added in (1, epochs):
        comparison = read(suite/f'comparison_added_{added:03}.json')
        for arm in ('control','multit'):
            assert comparison['arms'][arm]['endpoint'] == histories[arm][added-1]['evaluation']
    after = {p.relative_to(suite).as_posix():dict(sha256=sha(p),size=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
             for p in sorted(suite.rglob('*')) if p.is_file()}
    assert after == before
    result['checks'] = dict(run_files_unchanged=len(before),full_predictions=1+2*epochs,
        fixed_and_adaptive_mode_evaluations=4*(1+2*epochs),checkpoint_exports=2*epochs,
        cache_files_verified=89,events_per_dev_evaluation=len(y),candidate_events=int(c.sum()),
        target_events=int(y.sum()),hd_target_events=int(y[hd].sum()),logs_verified=len(logs_all),
        logged_wall_interval_seconds=max(l['updated_unix_s'] for l in logs_all)-min(l['updated_unix_s'] for l in logs_all),
        review_seconds=time.time()-started)
    result['primary_metrics'] = {a:result['arms'][a]['endpoint'] for a in result['arms']}
    result['decision'] = 'Continue the unchanged paired experiment to 50 total additional epochs, retain comparisons at 10/20/50; user starts training.'
    result['conclusion'] = 'T4 accuracy and loss improve, while T16 and the fixed exit policy regress; longer observation is warranted, with unequal temporal loss weights retained as a separate next experiment.'
    write(output,result)
    plot(result,output.with_name(output.stem.replace('review','curves')).with_suffix('.png'))
    print(json.dumps(dict(checks=result['checks'],paired_endpoints=result['paired_endpoints'],
                         best={a:result['arms'][a]['best_trained_by_mode'] for a in result['arms']}),indent=2))


def plot(result, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13.5,7.5),layout='constrained')
    colors={'control':'#2c65a1','multit':'#c06526'}
    for axis,mode in zip(axes.flat[:4],('4','8','16','adaptive')):
        for arm in colors:
            history=result['arms'][arm]['history']
            vals=[modes(result['initial_student'])[mode]['overall']['iou']]+[modes(e['evaluation'])[mode]['overall']['iou'] for e in history]
            axis.plot(range(len(vals)),np.array(vals)*100,'o-',label=arm,color=colors[arm])
        axis.axhline(result['frozen_ann']['overall']['iou']*100,color='#777777',ls='--',label='Frozen ANN')
        axis.set(title=('Fixed T'+mode if mode!='adaptive' else 'Fixed exit policy')+' | Dev IoU',ylabel='IoU (%)',xlabel='Additional epoch')
    for arm in colors:
        history=result['arms'][arm]['history']
        vals=[result['initial_student']['adaptive']['high_density']['recall']]+[e['evaluation']['adaptive']['high_density']['recall'] for e in history]
        axes[1,1].plot(range(len(vals)),np.array(vals)*100,'o-',label=arm,color=colors[arm])
    axes[1,1].set(title='Fixed exit policy | High-density recall',ylabel='Recall (%)',xlabel='Additional epoch')
    for mode in ('4','8','16'):
        h=result['arms']['multit']['history']
        axes[1,2].plot([e['added_epoch'] for e in h],[e['training']['prefix_mean_losses'][mode][0] for e in h],'o-',label='T'+mode)
    axes[1,2].set(title='Multi-T | Main CE+KD loss',ylabel='Loss (within-mode comparisons)',xlabel='Additional epoch')
    for axis in axes.flat:
        axis.grid(alpha=.2);axis.legend(fontsize=8)
        axis.xaxis.set_major_locator(plt.MaxNLocator(integer=True,nbins=6))
    fig.suptitle('Paired continuation from epoch159 | Full Main training / Full Dev evaluation',fontsize=13)
    fig.savefig(destination,dpi=170);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',type=Path,default=ROOT/'results/runs/h2pro_multit_v1_s42_ep159')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--output',type=Path,default=ROOT/'results/h2pro_multit_v1_epoch5_review_20261016.json')
    args=parser.parse_args();review(args.suite,args.epochs,args.output)
