"""Post-selection error transitions and scalar-cost disclosure; no tuning/inference."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse
from pathlib import Path
import numpy as np
from src.workflow import read,save,sha,manifests,RAW,validate_seal,EXPERIMENT


def transitions(y,bp,bc,p,c):
    positive=y==1;before=bp>=.35;after=p>=.35
    return dict(targets_recovered=int(np.count_nonzero(positive&~before&after)),
        targets_lost=int(np.count_nonzero(positive&before&~after)),
        false_positives_removed=int(np.count_nonzero(~positive&before&~after)),
        false_positives_introduced=int(np.count_nonzero(~positive&~before&after)),
        proposal_targets_added=int(np.count_nonzero(positive&~bc&c)),
        proposal_targets_lost=int(np.count_nonzero(positive&bc&~c)),
        proposal_backgrounds_added=int(np.count_nonzero(~positive&~bc&c)),
        proposal_backgrounds_removed=int(np.count_nonzero(~positive&bc&~c)))


def diagnose(out):
    seal=validate_seal(out)
    if read(out/'audit.json')['status']!='passed':raise ValueError('Complete prediction audit first')
    manifest,_=manifests();report={}
    for role in ('train_main','train_dev'):
        summary=read(out/(role+'_summary.json'));names=list(summary['policies'])
        per_sequence={name:[] for name in names};sums={name:{} for name in names};hd={name:{} for name in names}
        normalization={name:0 for name in names}
        for row in manifest['roles'][role]:
            r=read(out/role/(Path(row['sequence']).stem+'.json'))
            with np.load(RAW/row['cache_file'],allow_pickle=False) as z:y=z['label']
            with np.load(out/role/r['payload'],allow_pickle=False) as z:
                bp,bc=z['fixed100__probabilities'],z['fixed100__candidate']
                for name in names:
                    delta=transitions(y,bp,bc,z[name+'__probabilities'],z[name+'__candidate'])
                    for k,v in delta.items():
                        sums[name][k]=sums[name].get(k,0)+v
                        if r['metadata']['high_density']:hd[name][k]=hd[name].get(k,0)+v
                    m=r['policies'][name]['metrics'];base=r['policies']['fixed100']['metrics']
                    per_sequence[name].append(dict(sequence=row['sequence'],high_density=r['metadata']['high_density'],
                        iou=m['iou'],precision=m['precision'],recall=m['recall'],iou_difference_pp=100*(m['iou']-base['iou']),**delta))
                    w=z[name+'__windows']
                    if name.endswith('_rate100'):
                        normalization[name]+=int(np.count_nonzero(((w[:,1]-w[:,0])<4)&(w[:,3]>w[:,2])))*2*68*88
        policies={}
        for name,v in summary['policies'].items():
            b=summary['policies']['fixed100'];c=v['cost']
            # Exact arithmetic identity linking transitions to aggregate counts.
            if (sums[name]['targets_recovered']-sums[name]['targets_lost']!=v['metrics']['tp']-b['metrics']['tp']
                    or sums[name]['false_positives_introduced']-sums[name]['false_positives_removed']!=v['metrics']['fp']-b['metrics']['fp']
                    or hd[name]['proposal_targets_added']-hd[name]['proposal_targets_lost']!=v['high_density']['candidate_targets']-b['high_density']['candidate_targets']):
                raise AssertionError('Transition decomposition inconsistent')
            proposal_calls=(c['proposal_current_dense_mac']+c['proposal_history_extra_dense_mac'])//2580480
            scale_points=c['proposal_output_scalars']+c['refiner_output_scalars']
            policies[name]=dict(all_events=sums[name],high_density=hd[name],per_sequence=per_sequence[name],
                additional_scalar_arithmetic=dict(current_count_normalization_multiplications=normalization[name],
                    weight_scale_multiplications=scale_points,dequantization_divisions=scale_points,
                    hidden_requantization_multiplications=int(proposal_calls)*336*(48+24)+v['metrics']['candidate_events']*(64+32)),
                neurons_not_measured_as_hardware=True)
        report[role]=policies
    result=dict(source_sha256=sha(Path(__file__)),selection_sha256=sha(out/'selection.json'),
        interpretation='Descriptive error transitions only. No threshold/model/window policy chosen using Dev.',
        scalar_cost_scope='Listed scalar arithmetic follows reference operations; compiler fusion may differ. Feature counting/log compression/heatmap merging/routing/state/memory remain outside the neural MAC/AC ledger.',
        roles=report)
    save(out/'diagnostics.json',result)
    print('Saved diagnostics.json (prediction-only analysis, no inference or training).')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=EXPERIMENT/'results/runs/frozen_ann_v1')
    diagnose(p.parse_args().output)
