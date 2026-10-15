"""Independently recount full-source results; generate inspectable evidence, no training."""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np

import adapt_fred_statistics as run
from wangdao.h2.constants import EVENT_FEATURE_SPECS
from wangdao.h2.features import NEIGHBOR_OFFSETS, compress_counts

f=run.frozen


def recount(bits,totals):
    valid=((bits&16)!=0)&((bits&64)!=0)
    label,candidate,opening,updates=(bits&32)!=0,(bits&1)!=0,(bits&128)!=0,(bits&256)!=0
    groups={'all':valid,'opening_5s':valid&opening,'after_5s':valid&~opening,
            'statistics_updates':valid&updates,'query_only':valid&~updates}
    for mi,m in enumerate(f.MODELS):
        p=(bits&(2<<mi))!=0
        assert not np.any(p&~candidate)
        for group,take in groups.items():
            tn,fp,fn,tp=map(int,np.bincount(2*label[take].astype(np.int8)+p[take],minlength=4))
            pf=int(np.count_nonzero(take&label&~candidate))
            c=dict(events=tn+fp+fn+tp,target_events=fn+tp,candidate_events=int(np.count_nonzero(take&candidate)),
                candidate_targets=tp+fn-pf,proposal_fn=pf,refiner_fn=fn-pf,tp=tp,fp=fp,fn=fn,tn=tn)
            key=(m,group)
            if key not in totals:totals[key]=f.zero_counts()
            f.add_counts(totals[key],c)


def oracle(context,q,is_update,t0):
    values=[]
    for cell,ms in EVENT_FEATURE_SPECS:
        x,y,p=int(q['x'])//cell,int(q['y'])//cell,int(q['p'])
        t=int(np.floor((q['t']-t0)/ms))
        cx,cy=context['x']//cell,context['y']//cell
        ct=np.floor((context['t']-t0)/ms).astype(np.int64)
        at=(cx==x)&(cy==y)&(ct==t)
        values.extend([int(np.count_nonzero(at&(context['p']==p)))+int(not is_update),
                       int(np.count_nonzero(at&(context['p']!=p)))])
        for lag in (1,2):
            for polarity in (p,1-p):
                n=0
                for dx,dy in NEIGHBOR_OFFSETS:
                    n+=int(np.count_nonzero((cx==x+dx)&(cy==y+dy)&(ct==t-lag)&(context['p']==polarity)))
                values.append(n)
    return compress_counts(np.asarray(values,np.int64))


def review(output):
    output=Path(output);done=f.read_json(output/'completed.json')
    identity=f.read_json(output/'experiment_identity.json')
    assert identity['frozen_models']==f.FrozenModels().identity(run.DATA/'dataset')
    assert identity['runner_sha256']==f.sha256(Path(run.__file__))
    assert identity['adapter_sha256']==f.sha256(Path(run.adapter.__file__))
    totals={};before={};costs={m:{} for m in f.MODELS};audits=[]
    hist=np.zeros((len(run.GROUPS),36,256),np.int64)
    changed=np.zeros(36,np.int64)
    core={};transitions={m:np.zeros(4,np.int64) for m in f.MODELS}
    for seq_i,row in enumerate(done['sequences']):
        name=row['sequence'];directory=output/name
        assert row['identity']==identity
        for file,h in row['artifact_sha256'].items():assert f.sha256(directory/file)==h
        raw=np.load(str(run.SOURCE/'source'/name/'events.npy'),mmap_mode='r',allow_pickle=False)
        old=np.load(str(run.BASELINE/name/'predictions.npy'),mmap_mode='r',allow_pickle=False)
        bits=np.load(str(directory/'predictions.npy'),mmap_mode='r',allow_pickle=False)
        assert len(bits)==len(raw)==len(old)==row['source_events']
        actual={};previous={};closed=nc=updates=0
        for lo in range(0,len(raw),500_000):
            hi=min(lo+500_000,len(raw));b=bits[lo:hi];ob=old[lo:hi];r=raw[lo:hi]
            keep=run.prior.keep_events(np.arange(lo,hi),run.DENOMINATOR,seq_i)
            np.testing.assert_array_equal((b&256)!=0,keep)
            np.testing.assert_array_equal((b&(1+16+32+64+128)).astype(np.uint8),ob&(1+16+32+64+128))
            np.testing.assert_array_equal((b&16)!=0,r['valid'])
            np.testing.assert_array_equal((b&32)!=0,r['label'].astype(bool))
            assert not np.any(b&np.uint16(0xFE00))
            published=(b&64)!=0
            assert not np.any(((b&15)!=0)&~published)
            closed+=int(published.sum());nc+=int(((b&1)!=0).sum());updates+=int((keep&published).sum())
            recount(b,actual);recount(ob.astype(np.uint16)|(keep.astype(np.uint16)*256),previous)
            valid=published&((b&16)!=0);label=(b&32)!=0
            for mi,m in enumerate(f.MODELS):
                pred=(b&(2<<mi))!=0;old_pred=(ob&(2<<mi))!=0
                transitions[m]+=np.bincount((2*old_pred[valid].astype(np.int8)+pred[valid]),minlength=4)
                key=(m,'core_half_box')
                if key not in core:core[key]=f.zero_counts()
                f.add_counts(core[key],f.counts(label,pred,(b&1)!=0,valid&(r['region']==3)))
        assert run.report_metrics(actual)==row['models']
        assert run.report_metrics(previous)==row['paired_baseline']
        for key,c in actual.items():
            if key not in totals:totals[key]=f.zero_counts()
            f.add_counts(totals[key],c)
        for key,c in previous.items():
            if key not in before:before[key]=f.zero_counts()
            f.add_counts(before[key],c)
        assert closed==row['closed_source_events'] and nc==row['candidate_events'] and updates==row['statistics_update_events']
        assert row['discarded_query_events']==0
        windows=[json.loads(x) for x in (directory/'windows.jsonl').read_text().splitlines()]
        old_windows=[json.loads(x) for x in (run.BASELINE/name/'windows.jsonl').read_text().splitlines()]
        assert len(windows)==len(old_windows)==row['closed_windows']
        wr={m:{} for m in f.MODELS};wp={};wm={};offset=0
        for w,ow in zip(windows,old_windows):
            assert w['first_source_index']==offset==ow['first_event_index']
            assert w['source_events']==ow['events'] and w['proposal']==ow['proposal']
            assert w['right_boundary_ms']==ow['right_boundary_ms']
            offset+=w['source_events']
            run.prior.sum_values(wp,w['proposal'])
            for mi,m in enumerate(f.MODELS):
                detail=w['refiner_cost'][m]
                assert detail['rate_network_mac']==w['candidate_events']*[4480,2432,2304][mi]
                assert detail['executed_synaptic_mac']==w['candidate_events']*4480
                assert detail['executed_synaptic_ac']==0
                ac,spikes=detail['rate_network_ac'],detail['unit_spikes']
                if mi==0:assert ac==spikes==0
                if mi==1:assert ac==32*spikes and spikes<=w['candidate_events']*64*16
                if mi==2:
                    first,rem=divmod(ac-4*spikes,28);second=spikes-first
                    assert rem==0 and 0<=first<=w['candidate_events']*64*8 and 0<=second<=w['candidate_events']*32*8
                run.prior.sum_values(wr[m],detail)
                for s,c in w['models'][m].items():
                    key=(m,s)
                    if key not in wm:wm[key]=f.zero_counts()
                    f.add_counts(wm[key],c)
        assert offset==closed and run.report_metrics(wm)==row['models']
        for m in f.MODELS:
            assert f.cost_report(wr[m],wp,m,nc)==row['cost'][m]
            for key,v in row['cost'][m].items():
                if type(v) is int:costs[m][key]=costs[m].get(key,0)+v
        with np.load(directory/'features.npz',allow_pickle=False) as z:
            hist+=z['histogram'];changed+=z['changed_columns']
            assert not np.any(z['changed_columns'][run.adapter.UNCHANGED_COLUMNS])
            with np.load(run.SOURCE/'baseline'/name/'features.npz',allow_pickle=False) as old_features:
                # Target/background groups have identical candidate membership before and after.
                np.testing.assert_array_equal(z['histogram'][:2,run.adapter.UNCHANGED_COLUMNS],
                    old_features['histogram'][:2,run.adapter.UNCHANGED_COLUMNS])
            np.testing.assert_array_equal(z['sample_original'][:,run.adapter.UNCHANGED_COLUMNS],z['sample_adapted'][:,run.adapter.UNCHANGED_COLUMNS])
            checked=0
            t0=f.read_json(run.BASELINE/name/'result.json')['t0_ms']
            for wi in np.unique(z['sample_windows']):
                w=windows[int(wi)];lo=w['first_source_index'];hi=lo+w['source_events']
                previous_lo=windows[int(wi)-1]['first_source_index'] if wi else lo
                packet,_=run.prior.transform(raw[previous_lo:hi],1)
                keep=run.prior.keep_events(np.arange(previous_lo,hi),run.DENOMINATOR,seq_i)
                context=packet[keep]
                for pos in np.flatnonzero(z['sample_windows']==wi)[::8]:
                    qid=int(z['sample_ids'][pos]);q=packet[qid-previous_lo]
                    np.testing.assert_array_equal(oracle(context,q,bool(z['sample_updates'][pos]),t0),z['sample_adapted'][pos,4:28])
                    checked+=1
        audits.append({'sequence':name,'source_ids_verified':len(raw),'all_query_coverage_preserved':True,
            'confusions_independently_recounted':True,'all_window_costs_recounted':True,
            'original_candidates_and_label_bits_exact':True,'real_feature_oracle_queries':checked,
            'baseline_ann_replay_events':row['baseline_replayed_events'],
            'explicit_snn_checks':row['explicit_snn_checks']})
    baseline=f.read_json(run.BASELINE/'comparison.json')
    metrics=run.report_metrics(totals);paired=run.report_metrics(before)
    for m in f.MODELS:
        for s in ('all','opening_5s','after_5s'):assert paired[m][s]==baseline['models'][m]['metrics'][s]
        for k in ('neural_mac_dense_proposal','neural_mac_zero_skip_proposal','proposal_dense_mac','refiner_mac'):
            assert costs[m][k]==baseline['models'][m]['cost'][k]
    preservation=f.preservation(output)
    comparison={
        'status':'complete_verified','protocol':run.PROTOCOL,'identity':identity,
        'source_events':sum(r['source_events'] for r in done['sequences']),
        'closed_source_events':sum(r['closed_source_events'] for r in done['sequences']),
        'source_tail_events':sum(r['source_tail_events'] for r in done['sequences']),
        'candidate_events':sum(r['candidate_events'] for r in done['sequences']),
        'closed_windows':sum(r['closed_windows'] for r in done['sequences']),
        'statistics_update_events':sum(r['statistics_update_events'] for r in done['sequences']),
        'query_only_closed_events':sum(r['query_only_closed_events'] for r in done['sequences']),
        'discarded_query_events':0,'models':metrics,'baseline':paired,'cost':costs,
        'baseline_cost':{m:baseline['models'][m]['cost'] for m in f.MODELS},
        'deltas_pp':{m:{k:100*(metrics[m]['all'][k]-paired[m]['all'][k]) for k in ('iou','precision','recall','proposal_recall')} for m in f.MODELS},
        'core_recall':{m:f.summarize(core[(m,'core_half_box')]) for m in f.MODELS},
        'prediction_transitions_order':['both_negative','new_positive','new_negative','both_positive'],
        'prediction_transitions':{m:v.tolist() for m,v in transitions.items()},
        'changed_feature_values_by_column':changed.tolist(),'independent_audits':audits,
        'preservation':preservation,'sequences':done['sequences'],
        'reviewed_at':datetime.now(timezone.utc).isoformat(),'training_started':False}
    np.savez_compressed(output/'feature_histograms.npz',histogram=hist,groups=np.array(run.GROUPS))
    reference={r['column']:r['evuav_main_median'] for r in f.read_json(run.SOURCE/'readout.json')['feature_rows']}
    feature_rows=[]
    with np.load(run.SOURCE/'baseline/feature_histograms.npz',allow_pickle=False) as z:
        original_hist=z['histogram']
    for col in (4,10,16,22,30):
        row={'column':col,'evuav_main_median':reference[col]}
        for label,h in [('baseline',original_hist[0,col]),('adapted',hist[0,col])]:
            row[label+'_median']=int(np.searchsorted(np.cumsum(h),(int(h.sum())+1)//2))
            row[label+'_255_fraction']=float(h[255]/h.sum())
        feature_rows.append(row)
    comparison['target_candidate_feature_profiles']=feature_rows
    comparison['evuav_reference_path']=str(run.SOURCE/'evuav_feature_reference.json')
    f.atomic_json(output/'comparison.json',comparison)
    print(json.dumps({'status':'complete_verified','metrics':{m:metrics[m]['all'] for m in f.MODELS},
        'deltas_pp':comparison['deltas_pp'],'preservation':preservation},ensure_ascii=False))
    return comparison


def deliver(output):
    output=Path(output);c=f.read_json(output/'comparison.json')
    rows=[]
    for m in f.MODELS:
        for phase,key,ckey in [('before','baseline','baseline_cost'),('adapted','models','cost')]:
            v=c[key][m]['all'];cost=c[ckey][m]
            rows.append({'model':m,'input':phase,'iou_pct':100*v['iou'],'precision_pct':100*v['precision'],
                'recall_pct':100*v['recall'],'proposal_recall_pct':100*v['proposal_recall'],
                'neural_mac_g':cost['neural_mac_dense_proposal']/1e9,'neural_ac_g':cost['neural_ac']/1e9})
    f.atomic_json(output/'readout.json',{'status':c['status'],'rows':rows,'deltas_pp':c['deltas_pp'],
        'coverage':{k:c[k] for k in ('source_events','closed_source_events','source_tail_events','candidate_events','statistics_update_events','query_only_closed_events','discarded_query_events')},
        'query_only':{m:c['models'][m]['query_only'] for m in f.MODELS},'training_started':False})
    code=[
        "import json\nfrom pathlib import Path\nc=json.loads(Path('comparison.json').read_text(encoding='utf-8'))\nr=json.loads(Path('readout.json').read_text(encoding='utf-8'))\nassert c['status']=='complete_verified' and c['discarded_query_events']==0\nprint(r['coverage'])",
        "for row in r['rows']:\n    print(row)",
        "for model in c['models']:\n    print(model, 'query-only', c['models'][model]['query_only'])\n    print('after 5 s', c['models'][model]['after_5s'])\n    print('paired change in percentage points', c['deltas_pp'][model])",
        "assert c['preservation']['unchanged']\nassert not c['training_started']\nassert all(c['changed_feature_values_by_column'][i]==0 for i in list(range(4))+list(range(28,36)))\nprint('Preservation:',c['preservation'])\nprint('Independent audits:',c['independent_audits'])"
    ]
    cells=[{'cell_type':'markdown','metadata':{},'source':[
        '# FRED neighborhood-only adaptation\n',
        'All original events are presented to the unchanged Proposal, and all original candidates are classified. Only count columns 4:28 use sparse context updates; each query includes itself once.\n',
        'The threshold is 0.35, all weights/calibration are frozen, and no training occurred. These two previously inspected official training recordings use unchanged box-derived weak labels; they are not an independent benchmark test.\n',
        'The project README is the maintained status document. This notebook reads the independently verified artifacts; it does not rerun inference.\n']}]
    env={};old=Path.cwd()
    try:
        import os
        os.chdir(output)
        for i,cell in enumerate(code,1):
            stream=io.StringIO()
            with contextlib.redirect_stdout(stream):exec(compile(cell,f'cell_{i}','exec'),env)
            cells.append({'cell_type':'code','execution_count':i,'metadata':{},'source':cell.splitlines(True),
                'outputs':[{'output_type':'stream','name':'stdout','text':stream.getvalue().splitlines(True)}]})
    finally:os.chdir(old)
    for i,cell in enumerate(cells):cell['id']=f'fred-statistics-{i}'
    f.atomic_json(output/'FRED_statistics_adaptation.ipynb',{'nbformat':4,'nbformat_minor':5,
        'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.10'}},'cells':cells})
    source={'label':'Frozen full-coverage FRED statistics adaptation','files':[{'label':'comparison.json'},{'label':'experiment_identity.json'}],
        'metricDefinitions':[{'label':'Event metrics','definition':'IoU, precision and recall use the same valid source-event IDs and weak target denominator before and after adaptation.'},
                             {'label':'Compute','definition':'MAC and AC count neural operations over all original closed-window candidates; SNN AC is an equivalent derived from spike counts.'}],
        'filters':['Recordings: FRED train/0 and train/1','Decision threshold: 0.35','Statistics updates: deterministic 1/16 source-event subset'],
        'caveats':['The original incomplete final windows remain unpublished. No additional events are discarded.',
                   'The two recordings and the 1/16 setting were examined in prior diagnostics; these results are development evidence.',
                   'Box-derived labels have uncertain no-box negatives.','CPU inference uses dense count_reference; operation counts are not measured hardware energy or latency.'],
        'evidenceFlow':[{'kind':'validation','title':'Full-source reconciliation','detail':'Saved predictions were independently recounted and matched to the original candidate and label bits. Every window cost was reconciled.'}]}
    items=[{'id':'full-source-accuracy-and-cost','title':'Full-source metrics and neural operations before and after adaptation',
        'queries':[{'id':'three-model-comparison','source':source,'rows':rows,'columns':list(rows[0]),
                    'preview':{'kind':'aggregate','note':'Three frozen models on the same two complete source recordings, under the original closed-window scoring rule.'}}]},
        {'id':'all-query-coverage','title':'Events outside the statistics subset still receive classification',
         'queries':[{'id':'coverage-and-query-only','source':{'label':'Source-ID and feature-column audits',
            'files':[{'label':'comparison.json'},{'label':'regression.log'},{'label':'preservation_after.json'}],
            'caveats':['The statistics subset controls neighborhood state only; it does not select the classification or metric population.']},
            'rows':[{'model':m,'query_only_scored_events':c['models'][m]['query_only']['events'],
                     'query_only_target_events':c['models'][m]['query_only']['target_events'],
                     'query_only_recall_pct':100*c['models'][m]['query_only']['recall'],'discarded_query_events':0} for m in f.MODELS],
            'columns':['model','query_only_scored_events','query_only_target_events','query_only_recall_pct','discarded_query_events']}]}]
    f.atomic_json(output/'sources_receipt.json',{'schemaVersion':1,'items':items})
    print(json.dumps({'status':'written','rows':rows},ensure_ascii=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('review','deliver'))
    p.add_argument('--output',type=Path,default=run.OUT)
    args=p.parse_args()
    {'review':review,'deliver':deliver}[args.action](args.output)
