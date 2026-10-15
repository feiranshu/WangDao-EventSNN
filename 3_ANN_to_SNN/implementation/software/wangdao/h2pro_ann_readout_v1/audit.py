"""Final-weight causal stress checks and complete Train-main health census."""
import time

import numpy as np
import torch

from ..h2pro_snn.stream_data import prefetch, stream_batches
from ..h2pro_snn.training import cpu_weights, memory_stats
from ..h2pro_snn.diagnostics import summarize_counts
from ..h2pro_ann_state_v1.audit import prefix_batch
from .monitor import HealthMonitor
from .training import forward_batch, logit_counts

HARD_PREFIXES = {"098":5, "042":5, "043":41, "097":6, "088":32, "096":76}


@torch.no_grad()
def hard_prefixes(model, store, device, config, threshold):
    before = cpu_weights(model)
    rows = []
    for number, last in HARD_PREFIXES.items():
        record = store.get('train_main', 'train/train_'+number+'.npz')
        state = None
        monitor = HealthMonitor(config['stability_limits'], config['warning_limits'])
        failures = []
        all_candidates = all_large = all_bad_large = 0
        low, high = np.inf, -np.inf
        for window in range(last+1):
            batch = prefix_batch(record, window)
            logits, state, _, _ = forward_batch(model, batch, state, device)
            values = logits.cpu().numpy()
            candidate = batch['candidate']; labels = batch['labels']
            cv = values[candidate]
            if len(cv): low, high = min(low,float(cv.min())),max(high,float(cv.max()))
            all_candidates += len(cv); all_large += int((np.abs(cv)>80).sum())
            all_bad_large += int(((cv>80)&(labels[candidate]==0)).sum())
            try:
                monitor.observe(model,logits,candidate,batch['active'])
            except FloatingPointError as error:
                if ' exceeds ' not in str(error): raise
                failures.append({'window':window,'message':str(error)})
        rows.append({'sequence':record.metadata['sequence'],'windows':last+1,'reported_window':last,
            'stability':monitor.summary(),'outliers':failures,
            'all_prefix_candidate_logits':{'count':all_candidates,'min':low,'max':high,
                                          'abs_above80':all_large,'background_above80':all_bad_large},
            'last_window':{**summarize_counts(logit_counts(labels,values,candidate,threshold)),
                           'background_above80':int(np.count_nonzero((values>80)&candidate&(labels==0)))}})
    assert all(torch.equal(v,cpu_weights(model)[k]) for k,v in before.items())
    return {'role':'train_main','scope':'All six previously identified outlier slots, each from sequence start',
            'prefixes':rows,'all_guards_passed':all(not r['outliers'] for r in rows),'parameter_updates':0}


@torch.no_grad()
def frozen_train_sweep(model,store,device,config):
    before = cpu_weights(model)
    monitor = HealthMonitor(config['stability_limits'],config['warning_limits'])
    state = None
    failures = []
    events = windows = candidates = above80 = background_above80 = 0
    minimum,maximum = np.inf,-np.inf
    started = time.perf_counter()
    for index,(batch,_) in enumerate(prefetch(iter(stream_batches(store,'train_main',config['batch_size'],
                                                   seed=config['seed'],epoch=1))),1):
        logits,state,_,_ = forward_batch(model,batch,state,device)
        values = logits.cpu().numpy()[batch['candidate']]; labels = batch['labels'][batch['candidate']]
        # Count before guard enforcement: finite outlier batches are included.
        candidates += len(values)
        if len(values):
            minimum,maximum = min(minimum,float(values.min())),max(maximum,float(values.max()))
            above80 += int((np.abs(values)>80).sum())
            background_above80 += int(((values>80)&(labels==0)).sum())
        try:
            monitor.observe(model,logits,batch['candidate'],batch['active'])
        except FloatingPointError as error:
            if ' exceeds ' not in str(error): raise
            failures.append({'batch':index,'message':str(error),
                             'refs':[{k:v for k,v in ref.items() if k!='ids'} for ref in batch['refs']]})
        events += len(batch['labels']); windows += int(batch['active'].sum())
        if index%500==0:
            print({'stage':'frozen_readout_train_main','batches':index,'events':events,
                   'outlier_batches':len(failures)},flush=True)
    expected = store.manifest['roles']['train_main']
    assert events == sum(r['published_events'] for r in expected)
    assert windows == sum(r['closed_windows'] for r in expected)
    assert candidates == sum(r['candidate_events'] for r in expected)
    assert all(torch.equal(v,cpu_weights(model)[k]) for k,v in before.items())
    return {'status':'passed' if not failures else 'finite_amplitude_budget_outliers',
            'role':'train_main','events':events,'windows':windows,'batches':index,
            'outlier_batches':failures,'stability':monitor.summary(),
            'all_candidate_logits_including_outliers':{'count':candidates,'min':minimum,'max':maximum,
                                                     'abs_above80':above80,'background_above80':background_above80},
            'monitor_note':'Signed head extrema and layer maxima include outliers; candidate aggregate above includes every batch',
            'parameter_updates':0,'parameters_unchanged':True,'seconds':time.perf_counter()-started,
            **memory_stats(device)}
