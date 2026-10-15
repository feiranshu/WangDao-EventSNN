"""Finite, label-free encoder calibration on fixed Train-main feature samples."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import numpy as np

from ..h2pro_snn.foundation import sha256, read_json
from .common import protocol, calibration_manifest, identity, lock_directory, atomic_json
from .core import Encoder, forward, load_layers, margin


def sample_features(cache,manifest):
    """Never load labels, teacher probabilities or development samples for selection."""
    settings=protocol(); parts=[]; groups=[]; sequences=[]; receipts=[]
    rng=np.random.default_rng(settings['sample_seed'])
    for index,row in enumerate(manifest['sequences']):
        with np.load(Path(cache)/row['cache_file'],allow_pickle=False) as z:
            features=z['features']
            n=len(features); k=min(settings['candidate_rows_per_sequence'],n)
            ids=np.sort(rng.choice(n,k,replace=False))
            parts.append(features[ids].copy())
        groups.extend([bool(row['high_density'])]*k); sequences.extend([index]*k)
        receipts.append({'sequence':row['sequence'],'high_density':row['high_density'],
            'available_candidates':n,'sampled_candidates':k,
            'selected_row_indices_sha256':hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()})
    x=np.concatenate(parts)
    if x.dtype!=np.uint8 or x.ndim!=2 or x.shape[1]!=36 or not len(x):
        raise ValueError('Invalid original-feature calibration sample')
    return x,np.asarray(groups,bool),np.asarray(sequences,np.int32),receipts


def parameter_key(encoders):
    return tuple((e.threshold,e.initial_membrane) for e in encoders)


def select_candidates(candidates,baseline):
    def error_key(c):return (c['balanced_margin_mse'],c['equivalent_ac'],c['key'])
    accuracy=min(candidates,key=error_key)
    eligible=[c for c in candidates if all(c['margin_mse'][g]<=baseline['margin_mse'][g]
                                         for g in ('ordinary','high_density'))]
    # Baseline is explicitly included; no thresholds silently stop poor results.
    if not eligible:raise AssertionError('The uncalibrated candidate must be eligible')
    low=min(eligible,key=lambda c:(c['equivalent_ac'],c['balanced_margin_mse'],c['key']))
    return accuracy,low


def refinement_grid(center,settings):
    choices=set()
    for delta in settings['threshold_refinement_offsets']:
        theta=center.threshold+delta
        if theta<1 or theta>np.iinfo(np.int32).max:continue
        # Keep the center's initial phase when changing threshold, then vary it.
        initial=theta*center.initial_membrane//center.threshold
        for offset in settings['membrane_refinement_offsets']:
            value=initial+offset
            if 0<=value<theta:choices.add((theta,value))
    return [Encoder(theta,initial) for theta,initial in sorted(choices)]


def search_point(features,groups,sequences,layers,stage,steps,initial=(Encoder(),Encoder())):
    settings=protocol()
    teacher,_=forward(features,layers,'ann',steps)
    expected=margin(teacher)
    candidates={}
    members=[(np.flatnonzero(sequences==s),bool(groups[sequences==s][0])) for s in np.unique(sequences)]
    if not all(any(dense==g for _,dense in members) for g in (False,True)):
        raise ValueError('Calibration requires ordinary and high-density sequences')
    def assess(encoders):
        key=parameter_key(encoders)
        if key not in candidates:
            raw,cost=forward(features,layers,stage,steps,encoders,'count_reference')
            delta=margin(raw)-expected
            values={'ordinary':[],'high_density':[]}
            for ids,dense in members:
                values['high_density' if dense else 'ordinary'].append(float(np.mean(np.square(delta[ids]))))
            errors={g:float(np.mean(v)) for g,v in values.items()}
            candidates[key]={'key':str(key),'encoders':[asdict(e) for e in encoders],
                'margin_mse':errors,'balanced_margin_mse':sum(errors.values())/2,
                'logit_mse':float(np.mean(np.square(raw.astype(np.float64)-teacher))),
                'unit_spikes':cost.unit_spikes,'equivalent_ac':cost.rate_network_ac,
                'reference_executed_mac':cost.executed_synaptic_mac,
                'encoder_saturated_cells':cost.encoder_saturated_cells,'encoder_cells':cost.encoder_cells}
        return candidates[key]
    default=(Encoder(),Encoder())
    baseline=assess(default); current=initial; assess(current)
    grid=[Encoder(theta,theta*q//4) for theta in settings['threshold_grid']
          for q in settings['initial_phase_quarters']]
    active=1 if stage=='r1' else 2
    for phase,passes in (('coarse',settings['coordinate_passes']),('refine',settings['refinement_passes'])):
        for iteration in range(passes):
            for layer in range(active):
                local=[]
                choices=grid if phase=='coarse' else refinement_grid(current[layer],settings)
                for encoder in choices:
                    proposal=list(current); proposal[layer]=encoder
                    local.append(assess(tuple(proposal)))
                winner=min([assess(current)]+local,
                           key=lambda c:(c['balanced_margin_mse'],c['equivalent_ac'],c['key']))
                current=tuple(Encoder(**e) for e in winner['encoders'])
            print(json.dumps({'stage':'calibrate','conversion':stage,'T':steps,'phase':phase,'pass':iteration+1,
                              'candidates':len(candidates),'margin_mse':assess(current)['balanced_margin_mse']}),flush=True)
    all_candidates=list(candidates.values())
    accuracy,low=select_candidates(all_candidates,baseline)
    return {'stage':stage,'steps':steps,'uncalibrated':baseline,'accuracy':accuracy,'low_spike':low,
            'candidates':all_candidates,'labels_used':False,'weights_updated':False,
            'search_exhaustive_joint_grid':False}


def calibrate(cache,output):
    manifest=calibration_manifest(cache)
    spec={**identity('encoder_calibration'),'cache_manifest_sha256':sha256(Path(cache)/'manifest.json')}
    with lock_directory(output,spec) as directory:
        final=directory/'calibration.json'
        if final.exists():
            result=read_json(final)
            if result['identity']!=spec:raise ValueError('Saved calibration identity mismatch')
            return result
        features,groups,sequences,sample=sample_features(cache,manifest)
        layers=load_layers(); points=[]
        for steps in protocol()['steps']:
            initial=(Encoder(),Encoder())
            for stage in protocol()['stages']:
                receipt=directory/f'{stage}_t{steps}.json'
                if receipt.exists():
                    point=read_json(receipt)
                    if point['stage']!=stage or point['steps']!=steps:raise ValueError('Calibration receipt mismatch')
                else:
                    point=search_point(features,groups,sequences,layers,stage,steps,initial)
                    atomic_json(receipt,point)
                initial=tuple(Encoder(**e) for e in point['accuracy']['encoders'])
                points.append(point)
                atomic_json(directory/'progress.json',{'status':'calibrating','completed_points':len(points),
                    'total_points':6,'training_started':False})
        result={'version':spec['version'],'status':'complete','identity':spec,'protocol':protocol(),
            'data_role':'train_main','labels_used':False,'training_started':False,'weights_updated':False,
            'dev_used_for_selection':False,'reserved_sequences_read':0,'sample':sample,
            'sampled_candidates':len(features),'points':points}
        atomic_json(final,result)
        atomic_json(directory/'progress.json',{'status':'complete','training_started':False})
        return result
