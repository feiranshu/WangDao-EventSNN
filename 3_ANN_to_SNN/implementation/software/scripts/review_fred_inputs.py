"""Independent inspection/aggregation of frozen FRED input controls; no inference."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import diagnose_fred_inputs as d
from wangdao.fred_adapter_v1.core import frame_members, parse_annotations

OUT,ROOT,DATA,BASELINE=d.OUT,d.ROOT,d.DATA,d.BASELINE


def read(path):return d.frozen.read_json(path)


def quantile(hist,q):
    if not int(hist.sum()):return None
    return int(np.searchsorted(np.cumsum(hist),q*hist.sum(),side='left'))


def summarize_histograms(hist):
    result={}
    for gi,g in enumerate(d.GROUPS):
        cols=[]
        for col,h in enumerate(hist[gi]):
            n=int(h.sum())
            cols.append({'column':col,'events':n,'median':quantile(h,.5),'p90':quantile(h,.9),
                'p99':quantile(h,.99),'mean':float(np.dot(h,np.arange(256))/n) if n else None,
                'zero_fraction':float(h[0]/n) if n else None,
                'saturation_255_fraction':float(h[255]/n) if n else None})
        result[g]=cols
    return result


def profile_reference(output):
    cache=ROOT/'results/runs/h2pro_distill_v1_features'
    roles={}
    for role in ('train_main','train_dev'):
        hist=np.zeros((len(d.GROUPS),36,256),np.int64)
        rows=[]
        for p in sorted(cache.glob(f'{role}_train_*.json')):
            meta=read(p)
            q=cache/meta['cache_file']
            assert d.frozen.sha256(q)==meta['cache_sha256']
            with np.load(q,allow_pickle=False) as z:
                x=z['features'];y=z['labels'][z['candidate']]>0
                predicted=d.frozen.probabilities(z['teacher'])>=.35
                d.feature_histograms(hist,x,y,np.ones(len(x),bool),predicted)
            rows.append({'sequence':meta['sequence'],'events':meta['events'],
                         'candidate_events':len(x),'source_sha256':meta['source_sha256'],
                         'cache_sha256':meta['cache_sha256']})
        np.savez_compressed(output/f'evuav_{role}_features.npz',histogram=hist,groups=np.array(d.GROUPS))
        roles[role]={'sequences':rows,'candidate_events':sum(r['candidate_events'] for r in rows),
                     'features':summarize_histograms(hist)}
    d.frozen.atomic_json(output/'evuav_feature_reference.json',{'roles':roles,
        'scope':'unchanged cached candidate features, no reserved split and no new inference',
        'training_started':False})


def make_cases(output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    directory=output/'cases';directory.mkdir(exist_ok=True)
    records=[]
    index=read(DATA/'dataset/index.json')
    for seq in index['sequences']:
        name=seq['source_sequence']
        raw=np.load(str(output/'source'/name/'events.npy'),mmap_mode='r',allow_pickle=False)
        bits=np.load(str(BASELINE/name/'predictions.npy'),mmap_mode='r',allow_pickle=False)
        windows=[json.loads(x) for x in (BASELINE/name/'windows.jsonl').read_text().splitlines()]
        positive=[w for w in windows if w['counts']['h2pro_ann']['all']['target_events']>=100]
        recall=lambda w:w['counts']['h2pro_ann']['all']['tp']/w['counts']['h2pro_ann']['all']['target_events']
        ordered=sorted(positive,key=recall)
        opening=[w for w in windows if w['right_boundary_ms']<5000]
        selections=[('opening_fp',max(opening,key=lambda w:w['counts']['h2pro_ann']['all']['fp'])),
                    ('largest_fn',max(positive,key=lambda w:w['counts']['h2pro_ann']['all']['fn'])),
                    ('median_recall',ordered[len(ordered)//2]),('high_recall',ordered[-1])]
        with zipfile.ZipFile(DATA/'downloads'/f'{name}.zip') as archive:
            members=frame_members(archive.namelist())
            times=np.asarray(list(members),np.int64)
            ann=parse_annotations((DATA/'dataset/source'/name/'coordinates.txt').read_text('utf-8'),times)
            fig,axes=plt.subplots(4,3,figsize=(13.2,11.6),dpi=145,layout='constrained')
            for ri,(kind,w) in enumerate(selections):
                stamp=int(times[np.argmin(abs(times-(w['right_boundary_ms']-50)*1000))])
                fi=int(np.searchsorted(times,stamp))
                with archive.open(members[stamp]) as stream:
                    picture=np.asarray(Image.open(io.BytesIO(stream.read())).convert('RGB'))
                boxes=list(ann['boxes'][fi].values())
                lo=int(np.searchsorted(raw['t'],stamp-33333,side='left'))
                hi=int(np.searchsorted(raw['t'],stamp,side='right'))
                block=raw[lo:hi];b=bits[lo:hi]
                valid=block['valid']&((b&64)!=0)
                target=block['label'].astype(bool);prediction=(b&2)!=0
                canvas=np.full((720,1280,3),247,np.uint8)
                # At repeated pixels the ordering highlights the presence of FN;
                # all numerical counts are event-based, not read from this image.
                groups=[(valid&~target&~prediction,[145,160,174]),(valid&~target&prediction,[166,68,187]),
                        (valid&target&prediction,[19,151,95]),(valid&target&~prediction,[226,99,40])]
                for take,color in groups:
                    canvas[block['y'][take],block['x'][take]]=color
                axes[ri,0].imshow(picture)
                axes[ri,1].imshow(picture)
                axes[ri,2].imshow(canvas)
                for box in boxes:
                    for ax in axes[ri]:
                        ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],
                                               fill=False,edgecolor='#187bcc',linewidth=1.2))
                    cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
                    bw,bh=box[2]-box[0],box[3]-box[1]
                    for ax in axes[ri,1:]:
                        ax.add_patch(Rectangle((cx-bw/4,cy-bh/4),bw/2,bh/2,fill=False,
                                              edgecolor='#aa2c99',linestyle='--',linewidth=1))
                if boxes:
                    box=boxes[0];cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
                    radius=max(55,(box[2]-box[0])*.8,(box[3]-box[1])*.8)
                    for ax in axes[ri,1:]:
                        ax.set_xlim(max(0,cx-radius),min(1280,cx+radius))
                        ax.set_ylim(min(720,cy+radius),max(0,cy-radius))
                c=d.frozen.counts(target,prediction,(b&1)!=0,valid)
                metrics=d.frozen.summarize(c)
                rr='n/a' if metrics['recall'] is None else f"{metrics['recall']*100:.1f}%"
                axes[ri,0].set_title(f'{kind} | {stamp/1e6:.3f} s',fontsize=10)
                axes[ri,1].set_title('Official frame / source box',fontsize=10)
                axes[ri,2].set_title(f'ANN events: TP {c["tp"]:,} / FN {c["fn"]:,} | R {rr}',fontsize=9)
                for ax in axes[ri]:ax.set_xticks([]);ax.set_yticks([])
                Image.fromarray(picture).save(directory/f'{name}_{kind}_official.png')
                records.append({'sequence':name,'selection':kind,'selected_window':w['window'],
                    'frame_us':stamp,'source_png':members[stamp],'source_event_span':[lo,hi],
                    'source_boxes':[v.tolist() for v in boxes],'frame_span_counts':metrics,
                    'selection_scope':'diagnostic examples selected using existing baseline errors; not an unbiased accuracy estimate'})
            fig.suptitle(f'{name} | FRED weak-label / Refiner case audit\nGreen: TP; orange: FN; purple: FP; gray: TN. Dashed rectangle: central half-box.',fontsize=12)
            fig.savefig(directory/f'{name}_case_audit.png');plt.close(fig)
            # Inspect the assumption that an existing frame without an annotation
            # is background, concentrating on the largest post-opening FP cases.
            empty=[]
            for w in sorted(windows,key=lambda z:z['counts']['h2pro_ann']['all']['fp'],reverse=True):
                if w['right_boundary_ms']<5000 or w['counts']['h2pro_ann']['all']['target_events']:
                    continue
                fi=int(np.argmin(abs(times-(w['right_boundary_ms']-50)*1000)))
                if ann['boxes'][fi] or any(abs(int(times[fi])-s[0])<1_000_000 for s in empty):continue
                empty.append((int(times[fi]),w))
                if len(empty)==2:break
            if empty:
                fig,axes=plt.subplots(len(empty),2,figsize=(12,3.6*len(empty)),dpi=145,
                                      layout='constrained',squeeze=False)
                for ri,(stamp,w) in enumerate(empty):
                    with archive.open(members[stamp]) as stream:
                        picture=np.asarray(Image.open(io.BytesIO(stream.read())).convert('RGB'))
                    lo=int(np.searchsorted(raw['t'],stamp-33333,side='left'))
                    hi=int(np.searchsorted(raw['t'],stamp,side='right'))
                    b=bits[lo:hi];block=raw[lo:hi]
                    take=block['valid']&((b&64)!=0)&((b&2)!=0)
                    marked=picture.copy();marked[block['y'][take],block['x'][take]]=[230,65,180]
                    axes[ri,0].imshow(picture);axes[ri,1].imshow(marked)
                    axes[ri,0].set_title(f'No annotation | {stamp/1e6:.3f} s')
                    axes[ri,1].set_title(f'ANN-positive pixels (pink), {int(take.sum()):,} events')
                    for ax in axes[ri]:ax.set_xticks([]);ax.set_yticks([])
                    Image.fromarray(picture).save(directory/f'{name}_no_box_{ri}_official.png')
                    records.append({'sequence':name,'selection':'no_box_post5_high_fp',
                        'selected_window':w['window'],'frame_us':stamp,'source_png':members[stamp],
                        'source_boxes':[],'source_event_span':[lo,hi],'ann_positive_events':int(take.sum()),
                        'selection_scope':'diagnostic no-box examples, not proof that all absent annotations are true background'})
                fig.suptitle(f'{name} | Additional audit of unannotated source frames',fontsize=12)
                fig.savefig(directory/f'{name}_no_box_audit.png');plt.close(fig)
    d.frozen.atomic_json(directory/'case_manifest.json',{'cases':records,
        'frame_event_interval':'[frame timestamp -33333us, frame timestamp]',
        'pixel_overplot_note':'multiple events can occupy one pixel; numbers use all individual events',
        'training_started':False})


def add_model_reports(target,row):
    for m in d.frozen.MODELS:
        for scope in ('retained','paired_baseline','full_source'):
            for segment in ('all','after_5s','opening_5s'):
                key=(m,scope,segment)
                if key not in target:target[key]=d.frozen.zero_counts()
                d.frozen.add_counts(target[key],row[m][scope][segment])


def weak_label_review(output):
    index=read(DATA/'dataset/index.json');rows=[];total={}
    for seq in index['sequences']:
        name=seq['source_sequence'];src=DATA/'dataset/source'/name
        times=np.asarray(read(src/'frame_timestamps.json'),np.int64)
        annotations=parse_annotations((src/'coordinates.txt').read_text('utf-8'),times)
        present=np.array([bool(x) for x in annotations['boxes']])
        raw=np.load(str(output/'source'/name/'events.npy'),mmap_mode='r',allow_pickle=False)
        bits=np.load(str(BASELINE/name/'predictions.npy'),mmap_mode='r',allow_pickle=False)
        groups={}
        for lo in range(0,len(raw),500_000):
            b=bits[lo:lo+500_000];r=raw[lo:lo+500_000]
            ix=np.searchsorted(times,r['t'],side='right')-1
            left=np.clip(ix,0,len(times)-1);right=np.minimum(left+1,len(times)-1)
            exact=(ix>=0)&(r['t']==times[left])
            has_box=(present[left]&present[right])|(exact&present[left])
            no_box=~present[left]&~present[right]
            valid=((b&16)!=0)&((b&64)!=0)
            for mi,m in enumerate(d.frozen.MODELS):
                for group,mask in [('annotated_brackets',has_box),('no_box_brackets',no_box),
                    ('no_box_brackets_after5',no_box&(r['t']>=5_000_000))]:
                    key=f'{m}/{group}'
                    if key not in groups:groups[key]=d.frozen.zero_counts()
                    c=d.frozen.counts((b&32)!=0,(b&(2<<mi))!=0,(b&1)!=0,valid&mask)
                    d.frozen.add_counts(groups[key],c)
        for k,c in groups.items():
            if k not in total:total[k]=d.frozen.zero_counts()
            d.frozen.add_counts(total[k],c)
        rows.append({'sequence':name,'frames':len(times),'frames_with_box':int(present.sum()),
                     'groups':{k:d.frozen.summarize(v) for k,v in groups.items()}})
    visual=[
        {'sequence':'train_000','frame_us':107065596,
         'observation':'Compact object-shaped event cluster is visible without a box; next box appears at 107165595 us. This is a suspected omitted target interval, not established pixel ground truth.'},
        {'sequence':'train_000','frame_us':19166475,
         'observation':'A partially visible compact cluster touches the left sensor edge without a box. Border visibility/annotation policy may explain part of these counted false positives.'},
        {'sequence':'train_001','frame_us':78265884,
         'observation':'Compact object-shaped cluster near the lower image boundary has no box; both event and RGB annotation files omit this timestamp. Suspected annotation coverage issue.'},
        {'sequence':'train_001','frame_us':68665980,
         'observation':'A thin isolated streak is visible; object identity cannot be established from this event frame. Do not relabel it as a drone automatically.'},
    ]
    result={'status':'reviewed_with_material_label_limitations','sequences':rows,
        'aggregate':{k:d.frozen.summarize(v) for k,v in total.items()},
        'visual_review':visual,'images_inspected':[
            'cases/train_000_case_audit.png','cases/train_001_case_audit.png',
            'cases/train_000_no_box_audit.png','cases/train_001_no_box_audit.png'],
        'annotated_fn_case_observation':'Six annotated diagnostic examples show event clusters within the source boxes. The largest-FN examples contain dense object cores, not merely box-border background.',
        'risk':'No annotation is not verified absence. Some correct target detections may be counted as false positives by the preserved pilot rule.',
        'scope_limit':'Examples selected using baseline errors cannot estimate the dataset-wide omission rate. Box-derived positives are still not pixel-perfect truth.',
        'original_labels_or_masks_changed':False,'training_started':False,
        'official_sources':['https://github.com/miccunifi/FRED','https://arxiv.org/html/2506.05163v1']}
    d.frozen.atomic_json(output/'weak_label_review.json',result)


def review(output):
    arms={};audits=[]
    original=read(BASELINE/'comparison.json')
    for arm in d.ARMS:
        done=read(output/arm/'completed.json')
        totals={};cost={m:{} for m in d.frozen.MODELS};hist=np.zeros((len(d.GROUPS),36,256),np.int64)
        profiles={k:np.zeros(4096,np.int64) for k in ('source','mapped')}
        strata={}
        for seq_i,row in enumerate(done['sequences']):
            name=row['sequence'];directory=output/arm/name
            for f,h in row['artifact_sha256'].items():assert d.frozen.sha256(directory/f)==h
            raw=np.load(str(output/'source'/name/'events.npy'),mmap_mode='r',allow_pickle=False)
            bits=np.load(str(directory/'predictions.npy'),mmap_mode='r',allow_pickle=False)
            old=np.load(str(BASELINE/name/'predictions.npy'),mmap_mode='r',allow_pickle=False)
            actual={};retained=closed=0
            for lo in range(0,len(raw),500_000):
                hi=min(lo+500_000,len(raw));b=bits[lo:hi];old_b=old[lo:hi]
                is_closed=(b&64)!=0;keep=(b&256)!=0
                np.testing.assert_array_equal(keep,d.keep_events(np.arange(lo,hi),d.ARMS[arm]['retain_denominator'],seq_i))
                np.testing.assert_array_equal((b&(16+32+64+128)).astype(np.uint8),old_b&(16+32+64+128))
                assert not np.any(((b&15)!=0)&~keep)
                retained+=int(np.count_nonzero(keep&is_closed));closed+=int(is_closed.sum())
                # A separate bincount implementation reconstructs every confusion matrix.
                label=(b&32)!=0;valid=(b&16)!=0;opening=(b&128)!=0
                for mi,m in enumerate(d.frozen.MODELS):
                    pred=(b&(2<<mi))!=0;cand=(b&1)!=0
                    for scope,pp,cc,take0 in [('retained',pred,cand,keep),
                        ('paired_baseline',(old_b&(2<<mi))!=0,(old_b&1)!=0,keep),
                        ('full_source',pred,cand,np.ones(len(b),bool))]:
                        assert not np.any(pp&~cc)
                        for segment,mask in [('all',np.ones(len(b),bool)),('after_5s',~opening),('opening_5s',opening)]:
                            take=is_closed&valid&take0&mask
                            tn,fp,fn,tp=map(int,np.bincount((label[take].astype(np.int8)<<1)+pp[take],minlength=4))
                            pf=int(np.count_nonzero(take&label&~cc))
                            count=dict(events=tp+fp+fn+tn,target_events=tp+fn,candidate_events=int(np.count_nonzero(take&cc)),
                                candidate_targets=tp+fn-pf,proposal_fn=pf,refiner_fn=fn-pf,tp=tp,fp=fp,fn=fn,tn=tn)
                            key=(m,scope,segment)
                            if key not in actual:actual[key]=d.frozen.zero_counts()
                            d.frozen.add_counts(actual[key],count)
                    # Region and geometry-seam diagnostics never alter headline denominators.
                    rr=raw[lo:hi]
                    near_seam=(abs(rr['x'].astype(np.int32)-640)<16)|(abs(rr['y'].astype(np.int32)-360)<16)
                    for group,mask in [('core50',rr['region']==3),('inner80',rr['region']>=2),
                                      ('outer20',rr['region']==1),('away_from_seam',~near_seam),
                                      ('near_seam',near_seam),('polarity0',rr['p']==0),('polarity1',rr['p']==1)]:
                        take=valid&is_closed&keep&mask
                        key=f'{m}/{group}'
                        if key not in strata:strata[key]=d.frozen.zero_counts()
                        d.frozen.add_counts(strata[key],d.frozen.counts(label,pred,cand,take))
            assert d.metric_report(actual)==row['models']
            assert retained==row['retained_closed_events'] and closed==row['closed_source_events']
            add_model_reports(totals,row['models'])
            window_rows=[json.loads(x) for x in (directory/'windows.jsonl').read_text().splitlines()]
            assert len(window_rows)==row['closed_windows']
            w_totals={};wp={};wr={m:{} for m in d.frozen.MODELS}
            for w in window_rows:
                add_model_reports(w_totals,w['models']);d.sum_values(wp,w['proposal'])
                for m in d.frozen.MODELS:d.sum_values(wr[m],w['refiner_cost'][m])
            assert d.metric_report(w_totals)==row['models']
            for m in d.frozen.MODELS:
                c=d.frozen.cost_report(wr[m],wp,m,row['candidate_events'])
                assert c==row['cost'][m]
                assert c['refiner_mac']==row['candidate_events']*dict(zip(d.frozen.MODELS,[4480,2432,2304]))[m]
                for k,v in c.items():
                    if type(v) is int:cost[m][k]=cost[m].get(k,0)+v
            with np.load(directory/'features.npz',allow_pickle=False) as z:
                hist+=z['histogram']
                for k in profiles:profiles[k]+=z[f'{k}_pixel_density']
            if arm=='baseline':
                assert row['baseline_replayed_events']==row['closed_source_events']
                for m in d.frozen.MODELS:
                    expected=next(r for r in original['sequences'] if r['sequence']==name)['models'][m]
                    assert row['models'][m]['retained']==expected['metrics']
                    assert row['cost'][m]==expected['cost']
            audits.append({'arm':arm,'sequence':name,'source_ids_verified':len(raw),
                'confusions_recounted':True,'all_window_counts_and_costs_recounted':True,
                'baseline_exact':arm=='baseline','artifacts_match_hashes':True})
        np.savez_compressed(output/arm/'feature_histograms.npz',histogram=hist,groups=np.array(d.GROUPS))
        metrics=d.metric_report(totals)
        delta={m:{segment:{k:100*(metrics[m]['retained'][segment][k]-metrics[m]['paired_baseline'][segment][k])
                for k in ('iou','precision','recall','proposal_recall')}
                for segment in ('all','after_5s')} for m in d.frozen.MODELS}
        arms[arm]={'models':metrics,'paired_delta_pp':delta,'cost':cost,
            'source_events':sum(r['source_events'] for r in done['sequences']),
            'closed_source_events':sum(r['closed_source_events'] for r in done['sequences']),
            'retained_closed_events':sum(r['retained_closed_events'] for r in done['sequences']),
            'closed_windows':sum(r['closed_windows'] for r in done['sequences']),
            'features':summarize_histograms(hist),'strata':{k:d.frozen.summarize(v) for k,v in strata.items()},
            'pixel_density_2ms':{k:{'median':quantile(v,.5),'p90':quantile(v,.9),'p99':quantile(v,.99),
                                   'histogram':v.tolist()} for k,v in profiles.items()},
            'sequences':[{k:v for k,v in r.items() if k not in ('identity','artifact_sha256')} for r in done['sequences']]}
    prior=read(output/'preservation_before.json')
    changed=[p for p,h in prior.items() if not Path(p).is_file() or d.frozen.sha256(p)!=h['sha256']]
    noted_path=output/'concurrent_unrelated_changes.json'
    noted=read(noted_path) if noted_path.exists() else {'observed_files':[]}
    # A separately observed change in another direction's unused module is not
    # silently counted as unchanged. Preserve and disclose it; still fail on
    # any actual evaluated-model change or an unreviewed file change.
    known_unrelated={r['path'] for r in noted['observed_files'] if not r['imported_by_evaluated_models']}
    loaded={str(Path(m.__file__).absolute()) for m in __import__('sys').modules.values() if getattr(m,'__file__',None)}
    unexpected=[p for p in changed if p not in known_unrelated or str(Path(p).absolute()) in loaded]
    model_identity_unchanged=(d.frozen.FrozenModels().identity(DATA/'dataset')==read(output/'experiment_identity.json')['frozen_models'])
    preservation={'checked_files':len(prior),'changed_files':changed,'unchanged':not changed,
        'unchanged_files':len(prior)-len(changed),'evaluated_model_identity_unchanged':model_identity_unchanged,
        'unrelated_external_changes':[{'path':p,'before_sha256':prior[p]['sha256'],
            'after_sha256':d.frozen.sha256(p),'imported_by_evaluated_models':False} for p in changed if p in known_unrelated],
        'unexpected_changes':unexpected}
    d.frozen.atomic_json(output/'preservation_after.json',preservation)
    assert model_identity_unchanged and not unexpected
    result={'status':'complete_verified','protocol':d.PROTOCOL,'arms':arms,'independent_audits':audits,
        'source_audit':read(output/'source_audit.json'),'preservation':preservation,
        'reviewed_at':datetime.now(timezone.utc).isoformat(),'training_started':False}
    d.frozen.atomic_json(output/'comparison.json',result)
    print(json.dumps({'status':result['status'],'arms':{a:{m:v['models'][m]['retained']['all']['iou']
                     for m in d.frozen.MODELS} for a,v in arms.items()},'preservation':preservation}),flush=True)
    return result


def deliver(output):
    """Create an executed, read-only companion and reviewed inline evidence."""
    import contextlib
    report=read(output/'comparison.json')
    assert report['status']=='complete_verified'
    reference=read(output/'evuav_feature_reference.json')
    labels=read(output/'weak_label_review.json')
    source=report['source_audit']
    core=sum(r['region_diagnostics']['3']['events'] for r in source['sequences'])
    core_tp=sum(r['region_diagnostics']['3']['ann_tp'] for r in source['sequences'])
    flips=sum(r['alternative_label_sensitivity']['nearest']['flipped'] for r in source['sequences'])
    rows=[]
    for arm,value in report['arms'].items():
        for m in d.frozen.MODELS:
            v=value['models'][m]
            rows.append({'input':arm,'model':m,'retained_iou_pct':v['retained']['all']['iou']*100,
                'retained_precision_pct':v['retained']['all']['precision']*100,
                'retained_recall_pct':v['retained']['all']['recall']*100,
                'paired_baseline_recall_pct':v['paired_baseline']['all']['recall']*100,
                'full_source_recall_pct':v['full_source']['all']['recall']*100,
                'proposal_recall_retained_pct':v['retained']['all']['proposal_recall']*100})
    feature_rows=[]
    for col in (4,10,16,22,30):
        feature_rows.append({'column':col,
            'evuav_main_median':reference['roles']['train_main']['features']['weak_target'][col]['median'],
            **{a+'_median':v['features']['weak_target'][col]['median'] for a,v in report['arms'].items()},
            'fred_baseline_255_fraction':report['arms']['baseline']['features']['weak_target'][col]['saturation_255_fraction']})
    compact={'status':'complete_verified','rows':rows,'feature_rows':feature_rows,
        'core_half_width_height_target_events':core,'core_ann_recall':core_tp/core,
        'nearest_frame_label_changes':flips,'source_formula_mismatches':0,
        'label_caveat':labels['risk'],'preservation':report['preservation'],
        'training_started':False,'evaluated_model_files_or_calibration_changed':False}
    d.frozen.atomic_json(output/'readout.json',compact)

    # Execute precisely the notebook's read-only cells and embed their real outputs.
    code_cells=[
        "from pathlib import Path\nimport json\nrun = Path.cwd()\nif not (run / 'comparison.json').exists():\n    run = run / 'results/runs/fred_input_diagnosis_v1'\nr = json.loads((run / 'comparison.json').read_text(encoding='utf-8'))\nsummary = json.loads((run / 'readout.json').read_text(encoding='utf-8'))\nassert r['status'] == 'complete_verified'\nprint('Complete controls:', list(r['arms']))\nprint('Source events per arm:', r['arms']['baseline']['source_events'])",
        "for row in summary['rows']:\n    print(row['input'], row['model'], 'IoU/P/R retained:', *(round(row[k], 4) for k in ['retained_iou_pct', 'retained_precision_pct', 'retained_recall_pct']), 'paired baseline R:', round(row['paired_baseline_recall_pct'], 4), 'full source R:', round(row['full_source_recall_pct'], 4))",
        "print('Core weak target events:', summary['core_half_width_height_target_events'])\nprint('ANN recall within core:', summary['core_ann_recall'])\nprint('Nearest-frame label flips:', summary['nearest_frame_label_changes'])\nprint('Formula mismatches:', summary['source_formula_mismatches'])\nlabels = json.loads((run / 'weak_label_review.json').read_text(encoding='utf-8'))\nprint('No-box label limitation:', labels['risk'])\nprint('Known annotated brackets:', labels['aggregate']['h2pro_ann/annotated_brackets'])",
        "for row in summary['feature_rows']:\n    print(row)\nfor arm, result in r['arms'].items():\n    print(arm, 'retained closed events:', result['retained_closed_events'])\n    for model, cost in result['cost'].items():\n        print(model, 'equivalent MAC G:', cost['neural_mac_dense_proposal']/1e9, 'equivalent AC G:', cost['neural_ac']/1e9)",
        "assert r['preservation']['evaluated_model_identity_unchanged']\nassert not r['preservation']['unexpected_changes']\nassert not r['training_started']\nassert all(x['confusions_recounted'] and x['all_window_counts_and_costs_recounted'] for x in r['independent_audits'])\nprint(r['preservation'])\nprint('All completed source-ID/metric/cost audits:', len(r['independent_audits']))"
    ]
    cells=[{'cell_type':'markdown','metadata':{},'source':[
        '# FRED frozen-model input diagnosis\n',
        'Two official train recordings; source-event IDs are the grain. Weights, calibration and 0.35 threshold are fixed. No training.\n',
        'Density controls report retained-event metrics paired to the same baseline IDs, plus full-source rejection accounting. These are diagnostic weak-label results, not an official FRED benchmark.\n',
        'Authoritative experiment and source code: `experiment_identity.json`, `source_audit.json`, and the standalone scripts recorded in the delivery receipt. The project README is the maintained implementation-status document.\n']}]
    environment={}
    original_cwd=Path.cwd()
    try:
        import os
        os.chdir(output)
        for i,code in enumerate(code_cells,1):
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):exec(compile(code,f'notebook_cell_{i}','exec'),environment)
            cells.append({'cell_type':'code','execution_count':i,'metadata':{},'source':code.splitlines(True),
                'outputs':[{'output_type':'stream','name':'stdout','text':stream.getvalue().splitlines(True)}]})
    finally:
        os.chdir(original_cwd)
    cells.append({'cell_type':'markdown','metadata':{},'source':[
        '## Inspected cases\n',
        'Selections are diagnostic, not an unbiased sample of annotation quality. Orange denotes false negatives under the preserved weak labels.\n',
        '![Annotated cases 0](cases/train_000_case_audit.png)\n',
        '![Annotated cases 1](cases/train_001_case_audit.png)\n',
        '![No-box cases 0](cases/train_000_no_box_audit.png)\n',
        '![No-box cases 1](cases/train_001_no_box_audit.png)\n']})
    notebook={'nbformat':4,'nbformat_minor':5,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
        'language_info':{'name':'python','version':'3.10'}},'cells':cells}
    for i,cell in enumerate(cells):cell['id']=f'fred-diagnosis-{i}'
    d.frozen.atomic_json(output/'FRED_diagnosis.ipynb',notebook)

    metric_source={'label':'FRED frozen input controls','files':[{'label':'comparison.json'},{'label':'experiment_identity.json'}],
        'metricDefinitions':[{'label':'Retained-event recall','definition':'Recall counts correctly predicted weak target events among the retained, valid, closed-window events. Each control is paired to original predictions on the same source-event IDs.'},
            {'label':'Full-source recall','definition':'Full-source recall keeps the original valid target denominator and treats discarded input events as predicted background.'}],
        'filters':['Recordings: train/0 and train/1','Decision threshold: 0.35','Weights and calibration: frozen'],
        'caveats':['Box-derived event labels are not pixel-exact truth.','Thinning improvements on retained events are not improvements in full-source recall.',
                   'The 2x geometry control partitions the source into four fixed views, so tile boundary context changes with geometry.'],
        'evidenceFlow':[{'kind':'validation','title':'Independent recount','detail':'All eight arm/sequence outputs were checked against source IDs and independently recounted from saved predictions and window cost records.'}]}
    items=[{'id':'input-density-and-geometry','title':'Frozen input controls and matched-event results',
        'queries':[{'id':'paired-metrics','source':metric_source,'rows':rows,
            'columns':list(rows[0]),'preview':{'kind':'aggregate','note':'Three frozen models across four fixed input conditions.'}}]},
        {'id':'feature-distribution','title':'Target feature distribution differs from EV-UAV',
         'queries':[{'id':'feature-histograms','source':{'label':'Target candidate feature histograms',
            'files':[{'label':'evuav_feature_reference.json'},{'label':'comparison.json'}],
            'metricDefinitions':[{'label':'Feature median','definition':'Integer medians use candidate target events and the unchanged 36-column H2-pro feature definition.'}],
            'filters':['Reference: all 79 Train-main cached sequences','External data: both FRED pilot sequences'],
            'caveats':['FRED targets are derived from boxes; EV-UAV labels and camera statistics differ.']},
            'rows':feature_rows,'columns':list(feature_rows[0]),'preview':{'kind':'aggregate','note':'Selected count/context columns; all 36 columns are saved in the evidence.'}}]},
        {'id':'weak-label-quality','title':'Weak-label implementation matches; annotation absence remains uncertain',
         'queries':[{'id':'source-audit','source':{'label':'FRED source and visual label audit',
            'files':[{'label':'source_audit.json'},{'label':'weak_label_review.json'},{'label':'case_manifest.json'}],
            'links':[{'label':'Official annotation format','url':'https://github.com/miccunifi/FRED'}],
            'caveats':['Agreement of two label formulas does not establish pixel-level ground truth.',
                       'Four unannotated frames were selected by baseline errors; suspected omissions cannot establish a dataset-wide omission rate.']},
            'rows':[{'audited_events':source['audited_events'],'formula_mismatches':0,'nearest_frame_label_changes':flips,
                     'central_half_box_target_events':core,'central_half_box_ann_recall':core_tp/core}],
            'columns':['audited_events','formula_mismatches','nearest_frame_label_changes','central_half_box_target_events','central_half_box_ann_recall'],
            'summary':labels['risk']}]},
        {'id':'preservation','title':'Evaluated models were preserved; one unrelated change was recorded',
         'queries':[{'id':'hash-audit','source':{'label':'Before/after preservation hashes',
            'files':[{'label':'preservation_before.json'},{'label':'preservation_after.json'},{'label':'regression.log'}],
            'caveats':['One independent direction-2 calibration module changed during the run. It was not edited by this task or imported by the evaluated models; the report records it separately.']},
            'rows':[{'checked_files':report['preservation']['checked_files'],'changed_files':len(report['preservation']['changed_files']),
                     'training_started':False}], 'columns':['checked_files','changed_files','training_started']}]}]
    d.frozen.atomic_json(output/'sources_receipt.json',{'schemaVersion':1,'items':items})
    print(json.dumps({'status':'written','notebook':'FRED_diagnosis.ipynb','source_cards':len(items)}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['cases','reference','labels','review','deliver'])
    p.add_argument('--output',type=Path,default=OUT)
    args=p.parse_args()
    {'cases':make_cases,'reference':profile_reference,'labels':weak_label_review,'review':review,'deliver':deliver}[args.action](args.output)
