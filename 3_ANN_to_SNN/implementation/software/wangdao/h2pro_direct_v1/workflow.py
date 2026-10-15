"""No-training development comparisons, with sequence-level evaluation resume."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from ..h2pro_snn.diagnostics import aggregate_counts, error_counts
from ..h2pro_snn.foundation import read_json, role_records, sha256, verify_baseline
from .common import PACKAGE, atomic_json, base_identity, lock_directory, sources
from .core import BACKENDS, Cost, forward, load_layers, probabilities, spike_accumulate, validate_steps


def cache_manifest(directory):
    directory=Path(directory)
    m=read_json(directory/'manifest.json')
    identity=base_identity('original_feature_cache')
    if (any(m.get(k)!=v for k,v in identity.items()) or m.get('role')!='train_dev'
            or m.get('status')!='complete' or not m.get('complete_train_dev')):
        raise ValueError('Require the complete original-feature Train-dev cache')
    records=role_records('train_dev')
    if [r['sequence'] for r in m['sequences']] != [r['path'] for r in records]:
        raise ValueError('Wrong or incomplete development sequences')
    for row,record in zip(m['sequences'],records):
        filename=row['cache_file']
        if (Path(filename).name!=filename or row['source_sha256']!=record['sha256']
                or sha256(directory/filename)!=row['cache_sha256']):
            raise ValueError('Feature cache integrity mismatch')
    return m


def load_sequence(directory,row):
    with np.load(Path(directory)/row['cache_file'],allow_pickle=False) as z:
        a={k:z[k].copy() for k in ('features','label','candidate','teacher','logits')}
        meta=json.loads(str(z['metadata'].item()))
    n=len(a['label']); c=a['candidate']
    if (meta['sequence']!=row['sequence'] or meta['source_sha256']!=row['source_sha256']
            or n!=row['events'] or c.dtype!=np.bool_ or c.shape!=(n,)
            or a['teacher'].shape!=(n,) or a['teacher'].dtype!=np.float32
            or a['features'].dtype!=np.uint8 or a['features'].shape!=(int(c.sum()),36)
            or a['logits'].shape!=(int(c.sum()),4) or a['logits'].dtype!=np.float32
            or not np.isin(a['label'],[0,1]).all() or np.any(a['teacher'][~c]!=0)):
        raise ValueError('Original-feature cache payload mismatch')
    return a


def checks():
    baseline=verify_baseline()
    layers=load_layers()
    # Exhaust the full uint8 activation domain, signed weights and both IF backends.
    activations=np.tile(np.arange(256,dtype=np.uint8)[:,None],(1,64))
    weights=layers[1][0]
    for t in (1,4,8,16,32,64,255):
        ref,cr=spike_accumulate(activations,weights,t,'count_reference')
        actual,ca=spike_accumulate(activations,weights,t,'spike_add')
        if not np.array_equal(ref,actual) or cr.unit_spikes!=ca.unit_spikes:
            raise AssertionError('Binary IF event backend differs from count reference')
        if ca.executed_synaptic_mac!=0 or ca.executed_synaptic_ac!=ca.unit_spikes*32:
            raise AssertionError('Incorrect AC accounting')
    features=np.random.default_rng(42).integers(0,256,(257,36),dtype=np.uint8)
    ann,_=forward(features,layers)
    for stage in ('r1','r2'):
        for backend in BACKENDS:
            converted,_=forward(features,layers,stage,255,backend)
            if not np.array_equal(ann,converted):
                raise AssertionError('T255 must recover frozen exact-int8 logits')
    return {'status':'passed','baseline':baseline,'training_started':False,
            'uint8_encoder_domain_exhausted':True,'signed_spike_addition_equals_count_reference':True,
            't255_weight_preserving_alignment':True,'sources_utf8_lf':sources(),
            'not_a_development_accuracy_result':True}


def point(directory, manifest, layers, stage, steps, backend, output, chunk_rows):
    target=output/f'{stage}_t{steps}_{backend}'
    target.mkdir(exist_ok=True)
    rows=[]
    for row in manifest['sequences']:
        receipt=target/(Path(row['sequence']).stem+'.json')
        if receipt.exists():
            saved=read_json(receipt)
            if saved.get('cache_sha256')!=row['cache_sha256']:
                raise ValueError('Resumed sequence belongs to a different feature cache')
            rows.append(saved)
            continue
        a=load_sequence(directory,row)
        cost=Cost(); parts=[]; peak_chunk=0
        started=time.perf_counter()
        for lo in range(0,len(a['features']),chunk_rows):
            block=a['features'][lo:lo+chunk_rows]
            logits,c=forward(block,layers,stage,steps,backend)
            cost.merge(c); parts.append(logits); peak_chunk=max(peak_chunk,len(block))
        raw=np.concatenate(parts) if parts else np.empty((0,4),np.float32)
        p=np.zeros(len(a['label']),np.float32)
        p[a['candidate']]=probabilities(raw)
        elapsed=time.perf_counter()-started
        delta=raw.astype(np.float64)-a['logits'].astype(np.float64)
        exact=np.array_equal(raw,a['logits']) and np.array_equal(p,a['teacher'])
        # Gates check numerical equivalence, never amplitude or a low-accuracy run.
        if (stage=='ann' or steps==255) and not exact:
            raise AssertionError('Original ANN/T255 alignment failed; stop before another conversion stage')
        saved={'sequence':row['sequence'],'cache_sha256':row['cache_sha256'],
            'high_density':row['high_density'],'counts':error_counts(a['label'],p,a['candidate'],.35),
            'cost':asdict(cost),'proposal_synaptic_mac':row['proposal_synaptic_mac'],
            'teacher_exact':exact,'prediction_disagreements':int(np.count_nonzero((p>=.35)!=(a['teacher']>=.35))),
            'raw_logit_abs_error_max':float(np.abs(delta).max()) if delta.size else 0.,
            'raw_logit_squared_error_sum':float(np.square(delta).sum()),'raw_logit_count':int(delta.size),
            'inference_seconds_including_counting':elapsed,'max_chunk_rows':peak_chunk}
        atomic_json(receipt,saved)
        rows.append(saved)
        print(json.dumps({'stage':stage,'T':steps,'backend':backend,'sequence':row['sequence'],
                          'seconds':elapsed,'exact':exact}),flush=True)
    cost={k:sum(r['cost'][k] for r in rows) for k in asdict(Cost())}
    proposal=sum(r['proposal_synaptic_mac'] for r in rows)
    n=sum(r['counts']['candidate_events'] for r in rows)
    ann_refiner=n*4480
    result={'stage':stage,'steps':steps,'backend':backend,'threshold':.35,
        'overall':aggregate_counts([r['counts'] for r in rows]),
        'high_density':aggregate_counts([r['counts'] for r in rows if r['high_density']]),
        'ordinary':aggregate_counts([r['counts'] for r in rows if not r['high_density']]),
        'refiner_cost':cost,'proposal_synaptic_mac':proposal,
        'frozen_refiner_synaptic_mac':ann_refiner,
        'executed_neural_mac':proposal+cost['executed_synaptic_mac'],
        'executed_neural_ac':cost['executed_synaptic_ac'],
        'rate_network_neural_mac':proposal+cost['rate_network_mac'],
        'rate_network_neural_ac':cost['rate_network_ac'],
        'executed_neural_mac_reduction_fraction':1-(proposal+cost['executed_synaptic_mac'])/(proposal+ann_refiner),
        'rate_network_neural_mac_reduction_fraction':1-(proposal+cost['rate_network_mac'])/(proposal+ann_refiner),
        'executed_synaptic_mac_plus_ac_ratio':(proposal+cost['executed_synaptic_mac']+cost['executed_synaptic_ac'])/(proposal+ann_refiner),
        'teacher_exact':all(r['teacher_exact'] for r in rows),
        'prediction_disagreements':sum(r['prediction_disagreements'] for r in rows),
        'inference_seconds_including_counting':sum(r['inference_seconds_including_counting'] for r in rows),
        'sequences':rows,'end_to_end_complete':False,'energy_measured':False,
        'unaccounted':['original feature construction','heatmaps/routing/components','prototype probability readout','memory traffic'],
        'cost_note':'Arithmetic counts for executed software algorithm, not CPU/GPU instruction or energy measurements. Count-reference MACs remain MACs; rate_network is the equivalent explicit-spike contract.'}
    return result


def evaluate(cache,output,steps=(4,8,16,32,64,255),backend='count_reference',chunk_rows=8192):
    if backend not in BACKENDS or type(chunk_rows) is not int or chunk_rows<1:
        raise ValueError('Invalid backend or chunk size')
    steps=list(dict.fromkeys(steps))
    if not steps:
        raise ValueError('At least one T is required')
    for t in steps:
        validate_steps(t)
    manifest=cache_manifest(cache)
    layers=load_layers()
    identity={**base_identity('layerwise_development_evaluation'),
        'cache_manifest_sha256':sha256(Path(cache)/'manifest.json'),'sources_utf8_lf':sources(),
        'steps':steps,'backend':backend,'chunk_rows':chunk_rows}
    with lock_directory(output,identity) as directory:
        # Always establish exact endpoints first using the independently checked
        # count reference. Low-T results are measured, never gated on accuracy.
        plan=[('ann',255,'count_reference'),('r1',255,'count_reference'),('r2',255,'count_reference')]
        plan += [(s,t,backend) for t in steps for s in ('r1','r2') if (s,t,backend) not in plan]
        results=[]
        audit=checks()
        if not (directory/'checks.json').exists():
            atomic_json(directory/'checks.json',audit)
        for stage,t,implementation in plan:
            atomic_json(directory/'progress.json',{'status':'evaluating','stage':stage,'T':t,
                'backend':implementation,'completed_points':len(results),'total_points':len(plan),'training_started':False})
            result=point(cache,manifest,layers,stage,t,implementation,directory,chunk_rows)
            base=results[0] if results else result
            result['delta_to_frozen_ann_pp']={g:{m:100*(result[g][m]-base[g][m]) for m in ('iou','precision','recall')}
                                              for g in ('overall','ordinary','high_density')}
            atomic_json(directory/f'{stage}_t{t}_{implementation}.json',result)
            results.append(result)
        report={'version':identity['version'],'status':'complete','identity':identity,'training_started':False,
            'weights_updated':False,'reserved_sequences_read':0,'data_role':'train_dev',
            'protocol':read_json(PACKAGE/'protocol.json'),'results':results,
            'complete_train_dev':True,'threshold_selected_on_this_evaluation':False,
            'conclusion':'Frozen-weight layerwise conversion baseline; low-T accuracy and MAC/AC tradeoffs are measurements, not claims of novelty, full-SNN completion or hardware energy savings.'}
        atomic_json(directory/'comparison.json',report)
        atomic_json(directory/'progress.json',{'status':'complete','completed_points':len(results),
            'training_started':False,'comparison':str(directory/'comparison.json')})
        return report
