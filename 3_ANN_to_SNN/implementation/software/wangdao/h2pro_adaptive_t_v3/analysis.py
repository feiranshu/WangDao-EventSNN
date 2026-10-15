"""Paired error attribution, strictly after parameters have been sealed."""
import numpy as np
from ..h2pro_adaptive_t_v2.core import Policy, choose, THRESHOLD
from ..h2pro_adaptive_t_v2.scoring import error_changes
from .common import ANCHORS


def predict(data, policy):
    ix=choose(data['p'],policy)
    return data['p'][ix,np.arange(len(ix))]>=THRESHOLD


def paired_analysis(data, policies):
    anchor={k:predict(data,p) for k,p in ANCHORS.items()}
    y=data['labels']==1
    hd=np.zeros(len(y),bool)
    for seq in data['sequences']:
        if seq['row']['high_density']:hd[seq['lo']:seq['hi']]=True
    # The old independent candidate's additional misses are frozen reference events.
    missed=y&anchor['shared']&~anchor['combined']
    removed_fp=~y&anchor['shared']&~anchor['combined']
    groups=dict(overall=np.ones(len(y),bool),ordinary=~hd,high_density=hd)
    result={}
    for name,spec in policies.items():
        predicted=predict(data,Policy(**spec));parts={}
        for group,use in groups.items():
            parts[group]=dict(vs_shared=error_changes(data['labels'][use],predicted[use],anchor['shared'][use]),
                vs_combined=error_changes(data['labels'][use],predicted[use],anchor['combined'][use]),
                old_combined_extra_misses=int(np.count_nonzero(use&missed)),
                recovered_from_old_combined_misses=int(np.count_nonzero(use&missed&predicted)),
                old_combined_removed_fp=int(np.count_nonzero(use&removed_fp)),
                reintroduced_old_removed_fp=int(np.count_nonzero(use&removed_fp&predicted)))
        result[name]=parts
    delta=(anchor['combined'].astype(np.int8)-anchor['negative_only'].astype(np.int8)-
           anchor['positive_only'].astype(np.int8)+anchor['shared'].astype(np.int8))
    return dict(role=data['role'],reference='fixed .10/.95 combined versus shared .01/.99 at T8',
                policies=result,factorial_nonadditive_decisions=int(np.count_nonzero(delta)),
                factorial_interaction_sum=int(delta.sum()),labels_used_only_for_offline_analysis=True)
