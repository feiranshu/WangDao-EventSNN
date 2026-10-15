"""Exhaustive four-dimensional gate search using exact sufficient statistics.

Bins preserve each threshold's equality case separately: this is not approximate
probability binning. Labels participate only in offline Main parameter choice.
"""
from itertools import product
import numpy as np
from .common import DESIGN,NEGATIVE,POSITIVE,split_main
from .core import Policy,CONTROL,THRESHOLD
from .scoring import score,subset,policy_id,rank,summarize_counts

HIST_COLUMNS=('candidates','positives','tp16','fp16','spikes4','spikes8','spikes16')
SEARCH_COLUMNS=('tp','fp','spikes','exit4','exit8','exit16')


def grid_codes(negative=NEGATIVE,positive=POSITIVE):
    return np.array(list(product(range(len(negative)),range(len(positive)),
                                 range(len(negative)),range(len(positive)))),np.int16)


def policy_from_code(code,negative=NEGATIVE,positive=POSITIVE):
    a,b,c,d=map(int,code)
    return Policy(negative[a],positive[b],negative[c],positive[d]).validate()


def histogram(bundle,negative=NEGATIVE,positive=POSITIVE):
    if bundle['role']!='train_main':raise ValueError('Search histogram must use Main only')
    p=bundle['p'];y=bundle['labels'];spikes=bundle['spikes']
    if (p.ndim!=2 or p.shape[0]!=3 or len(y)!=p.shape[1] or spikes.shape!=p.shape
            or p.dtype!=np.float32 or not np.isfinite(p).all() or np.any((p<0)|(p>1))
            or not np.isin(y,[0,1]).all() or np.any(spikes<0) or np.any(np.diff(spikes,axis=0)<0)):
        raise ValueError('Invalid original prefix arrays')
    edges=np.unique(np.array([v for v in negative+positive if v is not None]+[THRESHOLD],np.float32))
    bins=2*len(edges)+1
    indexes=[np.searchsorted(edges,row,'left')+np.searchsorted(edges,row,'right') for row in p[:2]]
    flat=indexes[0]*bins+indexes[1]
    hist=np.zeros((2,len(HIST_COLUMNS),bins,bins),np.int64);meta=[]
    for high in (False,True):
        use=np.zeros(len(y),bool);rows=[]
        for seq in bundle['sequences']:
            if seq['row']['high_density']==high:
                use[seq['lo']:seq['hi']]=True;rows.append(seq['row'])
        if not rows:raise ValueError('Both ordinary and HD sequence groups are needed')
        ids=flat[use];positive_y=y[use]==1;pred16=p[2,use]>=THRESHOLD
        values=[np.ones(len(ids),np.int64),positive_y,positive_y&pred16,~positive_y&pred16,
                spikes[0,use],spikes[1,use],spikes[2,use]]
        for i,value in enumerate(values):
            if int(value.sum(dtype=np.int64))>=2**53:raise ValueError('Exact histogram sum exceeds FP64 integer domain')
            hist[int(high),i]=np.bincount(ids,weights=value,minlength=bins*bins).reshape(bins,bins).astype(np.int64)
        meta.append({k:sum(r[k] for r in rows) for k in ('events','target_events','candidate_events','candidate_targets',
                                                       'proposal_sparse_mac','proposal_dense_mac')})
        if meta[-1]['candidate_events']!=len(ids) or meta[-1]['candidate_targets']!=int(positive_y.sum()):
            raise ValueError('Histogram coverage differs from frozen candidate manifest')
    return dict(hist=hist,edges=edges,meta=meta)


def all_statistics(table,negative=NEGATIVE,positive=POSITIVE):
    h,edges=table['hist'],table['edges'];bins=h.shape[-1]
    prefix=np.pad(h,((0,0),(0,0),(1,0),(1,0))).cumsum(axis=2).cumsum(axis=3)
    def low_end(v):return 0 if v is None else 2*int(np.searchsorted(edges,np.float32(v)))+2
    def high_start(v):return bins if v is None else 2*int(np.searchsorted(edges,np.float32(v)))+1
    def rect(x0,x1,y0,y1):
        y0,y1=np.broadcast_arrays(y0,y1)
        return prefix[:,:,x1,y1]-prefix[:,:,x0,y1]-prefix[:,:,x1,y0]+prefix[:,:,x0,y0]
    pairs=list(product(negative,positive));lo8=np.array([low_end(a) for a,b in pairs]);hi8=np.array([high_start(b) for a,b in pairs])
    result=np.empty((len(pairs)**2,2,len(SEARCH_COLUMNS)),np.int64)
    for j,(neg4,pos4) in enumerate(pairs):
        lo4,hi4=low_end(neg4),high_start(pos4)
        low4,high4=rect(0,lo4,0,bins),rect(hi4,bins,0,bins)
        low8,high8=rect(lo4,hi4,0,lo8),rect(lo4,hi4,hi8,bins)
        end=rect(lo4,hi4,lo8,hi8)
        tp=high4[:,1,None]+high8[:,1,:]+end[:,2,:]
        fp=(high4[:,0]-high4[:,1])[:,None]+high8[:,0,:]-high8[:,1,:]+end[:,3,:]
        spikes=(low4[:,4]+high4[:,4])[:,None]+low8[:,5,:]+high8[:,5,:]+end[:,6,:]
        exit4=np.broadcast_to((low4[:,0]+high4[:,0])[:,None],tp.shape)
        exit8=low8[:,0,:]+high8[:,0,:];exit16=end[:,0,:]
        result[j*len(pairs):(j+1)*len(pairs)]=np.stack([tp,fp,spikes,exit4,exit8,exit16],axis=-1).transpose(1,0,2)
    if np.any(result<0) or not np.all(result[:,:,3:].sum(axis=2)==np.array([r['candidate_events'] for r in table['meta']])):
        raise AssertionError('Grid sufficient statistics do not partition candidate coverage')
    return result


def compact_result(stats,meta):
    rows=[]
    for m,v in zip(meta,stats):
        tp,fp=map(int,v[:2]);fn=m['target_events']-tp
        rows.append(dict(events=m['events'],target_events=m['target_events'],candidate_events=m['candidate_events'],
            candidate_targets=m['candidate_targets'],proposal_fn=m['target_events']-m['candidate_targets'],
            refiner_fn=m['candidate_targets']-tp,tp=tp,fp=fp,fn=fn,tn=m['events']-tp-fp-fn))
    r=dict(overall=summarize_counts(rows),ordinary=summarize_counts([rows[0]]),high_density=summarize_counts([rows[1]]))
    total=stats.sum(axis=0);calls=int(total[3]+2*total[4]+3*total[5]);n=sum(m['candidate_events'] for m in meta)
    r['cost']=dict(inference_equivalent_ac=int(total[2])*32,
        inference_equivalent_mac_sparse_proposal=sum(m['proposal_sparse_mac'] for m in meta)+n*2304+calls*128)
    return r


def check_direct(stats,meta,direct):
    compact=compact_result(stats,meta)
    for g in ('overall','ordinary','high_density'):
        if compact[g]!=direct[g]:raise AssertionError('Histogram/direct full-event metric mismatch: '+g)
    for k,v in compact['cost'].items():
        if direct['cost'][k]!=v:raise AssertionError('Histogram/direct cost mismatch: '+k)
    if [direct['cost']['exit_counts'][str(t)] for t in (4,8,16)]!=stats.sum(axis=0)[3:].tolist():
        raise AssertionError('Histogram/direct exit counts differ')


def select_main(bundle,progress=lambda x:None):
    if bundle['role']!='train_main':raise ValueError('Never select exit parameters on Dev')
    split=split_main([s['row'] for s in bundle['sequences']])
    fit,valid=(subset(bundle,split[k]) for k in ('fit','validation'))
    base_fit,base_val=(score(b,fixed_step=16) for b in (fit,valid))
    table=histogram(fit);stats=all_statistics(table);codes=grid_codes()
    records=[compact_result(v,table['meta']) for v in stats]
    progress(dict(stage='main_full_grid',completed=len(codes),total=len(codes)))
    control_index=next(i for i,code in enumerate(codes) if policy_from_code(code)==CONTROL)
    shortlist={control_index}
    for coefficient in DESIGN['objectives'].values():
        shortlist.update(sorted(range(len(codes)),key=lambda i:(rank(records[i],coefficient,base_fit),-i),reverse=True)[:8])
    validation={};fit_shortlist={}
    for i in sorted(shortlist):
        policy=policy_from_code(codes[i]);fit_shortlist[str(i)]=score(fit,policy)
        check_direct(stats[i],table['meta'],fit_shortlist[str(i)])
        validation[str(i)]=score(valid,policy)
    selected={}
    for name,coefficient in DESIGN['objectives'].items():
        i=max(sorted(shortlist),key=lambda j:(rank(validation[str(j)],coefficient,base_val),-j))
        policy=policy_from_code(codes[i])
        selected[name]=dict(grid_index=i,policy_id=policy_id(policy),policy=policy.as_dict(),coefficient=coefficient,
            fit=fit_shortlist[str(i)],validation=validation[str(i)],full_main=score(bundle,policy,details=True))
    result=dict(design=DESIGN,role='train_main',split=split,grid_size=len(codes),shortlist=sorted(shortlist),
        fit_shortlist=fit_shortlist,validation_candidates=validation,selected=selected,
        fixed_main=score(bundle,fixed_step=16,details=True),shared_main=score(bundle,CONTROL,details=True),
        control_grid_index=control_index,fit_group_metadata=table['meta'],fit_histogram_columns=list(HIST_COLUMNS),
        fit_search_columns=list(SEARCH_COLUMNS),fit_archive='fit_search.npz',
        dev_used_for_selection=False,weights_updated=False,selection_sealed=True)
    arrays=dict(grid_codes=codes,statistics=stats,histogram=table['hist'],edges=table['edges'])
    return result,arrays
