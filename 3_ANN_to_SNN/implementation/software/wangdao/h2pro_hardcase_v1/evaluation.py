"""Milestone Main calibration is committed before its Dev validation."""
from datetime import datetime, timezone

import numpy as np
import torch

from ..h2pro_direct_v1.core import probabilities
from ..h2pro_distill_v1.evaluation import score, delta
from ..h2pro_fixed_diagnosis_v1.thresholds import GROUPS, POLICIES, CONTROL, choose, curve, measure
from .common import atomic_json, read_json, sha256, digest, save_arrays


@torch.no_grad()
def infer_main(model, data, batch_size, pause=lambda: None):
    model.eval()
    n = data.size('train_main')
    raw, spikes = np.empty((n,4),np.float32), np.empty((n,2),np.int64)
    for lo in range(0,n,batch_size):
        pause()
        r,s = model(data.roles['train_main']['features'][lo:lo+batch_size])
        raw[lo:lo+len(r)], spikes[lo:lo+len(r)] = r.cpu().numpy(), s.cpu().numpy()
    return raw,spikes


def fit_main(model, data, batch_size, directory, added_epoch, pause=lambda: None):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/f'added_{added_epoch:03}_main.json'
    path = directory/f'added_{added_epoch:03}_main_predictions.npz'
    model_id = digest(model.state_dict())
    if marker.exists():
        selected = read_json(marker)
        if (selected['model_digest'] != model_id or selected['predictions_sha256'] != sha256(path)
                or selected['selection_role'] != 'train_main'):
            raise ValueError('Milestone Main evidence no longer matches frozen weights')
        return selected
    raw,spikes = infer_main(model,data,batch_size,pause)
    p = probabilities(raw)
    seqs = data.sequences['train_main']
    hd = np.concatenate([np.full(s['hi']-s['lo'],s['row']['high_density'],bool) for s in seqs])
    y = data.roles['train_main']['labels'].cpu().numpy()
    ann = measure(seqs,probabilities(data.roles['train_main']['teacher'].cpu().numpy()),CONTROL)
    totals = {g: ann[g]['target_events'] for g in GROUPS}
    curves = curve(p,y,hd,totals)
    choices = {policy: choose(curves,ann,policy,role='train_main') for policy in POLICIES}
    points = {'control_035': measure(seqs,p,CONTROL)}
    for policy,choice in choices.items():
        if choice['status'] == 'selected':
            points[policy] = measure(seqs,p,choice['threshold'])
            for group in GROUPS:
                for metric in ('iou','precision','recall'):
                    if points[policy][group][metric] != choice['main'][group][metric]:
                        raise AssertionError('Main search differs from direct rescore')
    save_arrays(path,raw=raw,spikes=spikes)
    selected = {'selection_role': 'train_main', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'model_digest': model_id, 'predictions_sha256': sha256(path), 'choices': choices,
                'operating_points': points, 'cost': score(data,raw,spikes,'train_main')['cost'],
                'added_epoch': added_epoch, 'thresholds_reselected_on_dev': False}
    atomic_json(marker,selected)
    return selected


def finish_milestone(data, raw, spikes, fixed_result, selected, directory, added_epoch, ann, validation_started):
    marker = directory/f'added_{added_epoch:03}_main.json'
    if read_json(marker) != selected or selected['created_at_utc'] > validation_started:
        raise AssertionError('Main selection must be sealed before Dev inference')
    points = {'control_035': fixed_result}
    p = probabilities(raw)
    for policy,choice in selected['choices'].items():
        if choice['status'] == 'selected':
            points[policy] = measure(data.sequences['train_dev'],p,choice['threshold'])
    for point in points.values():
        point['delta_to_ann_pp'] = delta(point,ann)
    path = directory/f'added_{added_epoch:03}_dev_predictions.npz'
    save_arrays(path,raw=raw,spikes=spikes)
    result = {'role': 'train_dev', 'added_epoch': added_epoch,
              'main_selection_sha256': sha256(marker), 'validation_started_at_utc': validation_started,
              'predictions_sha256': sha256(path), 'thresholds_reselected_on_dev': False,
              'operating_points': points, 'cost': fixed_result['cost']}
    atomic_json(directory/f'added_{added_epoch:03}_dev.json',result)
    return result
