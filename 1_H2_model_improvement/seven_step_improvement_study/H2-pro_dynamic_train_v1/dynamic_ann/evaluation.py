import time
import numpy as np
import torch
from .common import counts,metrics,atomic_npz
from .model import exact_layers_on_device,exact_forward,require_finite
from wangdao.h2.numpy_model import _refiner_probability

COUNT_KEYS=('events','target_events','candidate_events','candidate_targets','tp','fp','fn','tn','proposal_fn','refiner_fn')


def score(data,probabilities):
    totals={g:{k:0 for k in COUNT_KEYS} for g in ('overall','high_density','ordinary')}
    rows=[]
    proposal_keys=('proposal_current_sparse_mac','proposal_current_dense_mac',
                   'proposal_history_extra_sparse_mac','proposal_history_extra_dense_mac')
    cost={k:0 for k in proposal_keys};cost['refiner_mac']=0
    wait_sum=0.
    for seq in data.sequences['train_dev']:
        row=seq['row'];entry=row['arms'][data.arm];n=len(seq['full_labels'])
        c=np.zeros(n,bool);c[seq['candidate_ids']]=True
        p=np.zeros(n,np.float32);p[seq['candidate_ids']]=probabilities[seq['lo']:seq['hi']]
        item=counts(seq['full_labels'],p,c)
        group='high_density' if row['metadata']['high_density'] else 'ordinary'
        for g in ('overall',group):
            for k in COUNT_KEYS:totals[g][k]+=item[k]
        rows.append(dict(sequence=row['sequence'],high_density=row['metadata']['high_density'],**metrics(item)))
        for k in proposal_keys:cost[k]+=entry['cost'][k]
        cost['refiner_mac']+=len(seq['candidate_ids'])*4480
        wait_sum+=entry['waiting_ms']['mean']*n
    cost.update(mac_sparse_proposal=cost['proposal_current_sparse_mac']+cost['proposal_history_extra_sparse_mac']+cost['refiner_mac'],
                mac_dense_proposal=cost['proposal_current_dense_mac']+cost['proposal_history_extra_dense_mac']+cost['refiner_mac'],ac=0,
                actual_int8_reference_backend='dense FP64 exact integer-valued dot; zero-skip MAC is equivalent',
                feature_control_memory_included=False)
    return {**{g:metrics(v) for g,v in totals.items()},'sequences':rows,'cost':cost,
            'mean_publication_wait_ms':wait_sum/totals['overall']['events'],'threshold':.35}


@torch.no_grad()
def evaluate(model,data,batch_size,pause=lambda:None,keep_raw=False):
    started=time.perf_counter();model.eval();features=data.roles['train_dev']['features']
    arrays={k:[] for k in ('float','int8')};layers=model.export_layers();gpu_layers=exact_layers_on_device(layers,data.device)
    for lo in range(0,len(features),batch_size):
        pause();x=features[lo:lo+batch_size]
        a=model(x);b=exact_forward(x,gpu_layers);require_finite([a,b],'evaluation logits')
        arrays['float'].append(a.cpu().numpy());arrays['int8'].append(b.cpu().numpy())
    raw={k:np.concatenate(v) if v else np.empty((0,4),np.float32) for k,v in arrays.items()}
    probabilities={k:_refiner_probability(v,2) for k,v in raw.items()}
    result={k:score(data,v) for k,v in probabilities.items()}
    differences=np.abs(probabilities['float']-probabilities['int8'])
    result['quantization_difference']=dict(mean_abs_probability=float(differences.mean()),
        max_abs_probability=float(differences.max()),decision_flips=int(np.count_nonzero((probabilities['float']>=.35)!=(probabilities['int8']>=.35))))
    result['evaluation_seconds']=time.perf_counter()-started
    return result,probabilities,raw if keep_raw else None


def export(path,model,arm,epoch):
    arrays=dict(dimensions=np.array([36,64,32,4]),variant=np.array('C1-proposal-motion'),
                experiment=np.array('h2pro_dynamic_ann_refiner_train_v1'),arm=np.array(arm),epoch=np.array(epoch),event_threshold=np.array(.35))
    for i,(w,s) in enumerate(model.export_layers()):arrays[f'layer{i}_weight']=w;arrays[f'layer{i}_scale']=s
    atomic_npz(path,**arrays)


def differences(result,reference):
    return {kind:{g:{k:100*(result[kind][g][k]-reference[kind][g][k]) for k in ('iou','precision','recall')}
                  for g in ('overall','high_density')} for kind in ('float','int8')}
