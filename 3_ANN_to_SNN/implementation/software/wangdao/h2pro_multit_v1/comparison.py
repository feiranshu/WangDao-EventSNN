"""Compare paired milestones at all three prefixes and the fixed exit policy."""
from .common import ARMS, SOURCE_EPOCH, atomic_json, read_json, sha256
from .evaluation import delta


def modes(evaluation):
    return {**{'T'+t:v for t,v in evaluation['fixed'].items()},'adaptive':evaluation['adaptive']}


def compact(value):
    return dict(iou=value['overall']['iou'],precision=value['overall']['precision'],
                recall=value['overall']['recall'],hd_recall=value['high_density']['recall'],
                mac=value['cost']['inference_equivalent_mac_sparse_proposal'],
                ac=value['cost']['inference_equivalent_ac'],average_steps=value['cost']['average_steps'])


def compare(suite,added,ready):
    result=dict(added_epochs=added,effective_epoch=SOURCE_EPOCH+added,role='train_dev',
                initial_student=ready['initial_student'],frozen_ann=ready['frozen_ann'],arms={})
    endpoints={}
    for arm in ARMS:
        directory=suite/arm;training=read_json(directory/'training.json')
        if len(training['epochs'])<added:raise ValueError('Both arms must finish the milestone')
        history=training['epochs'][:added]
        entry=read_json(directory/'epochs'/f'added_{added:03}.json')
        if entry!=history[-1] or entry['added_epoch']!=added:raise ValueError('Epoch/history mismatch')
        if sha256(directory/'epochs'/f'added_{added:03}_predictions.npz')!=entry['predictions_sha256']:
            raise ValueError('Saved Dev predictions changed')
        endpoint=entry['evaluation'];endpoints[arm]=endpoint
        candidates=[(0,ready['initial_student'])]+[(e['added_epoch'],e['evaluation']) for e in history]
        rank=lambda e:(e['overall']['iou'],e['overall']['recall'])
        best_t16=max(candidates,key=lambda x:rank(x[1]))
        best_adaptive=max(candidates,key=lambda x:rank(x[1]['adaptive']))
        result['arms'][arm]=dict(
            endpoint=endpoint,memory=entry['memory'],best_added_epoch=best_t16[0],best=best_t16[1],
            best_adaptive_added_epoch=best_adaptive[0],best_adaptive=best_adaptive[1]['adaptive'],
            learning_trend=[dict(added_epoch=e['added_epoch'],
                modes={k:compact(v) for k,v in modes(e['evaluation']).items()},
                mean_losses=e['training']['mean_losses'],prefix_mean_losses=e['training']['prefix_mean_losses'],
                memory=e['memory']) for e in history],
            delta_to_source159_pp={k:delta(v,modes(ready['initial_student'])[k]) for k,v in modes(endpoint).items()},
            delta_to_ann_pp={k:delta(v,ready['frozen_ann']) for k,v in modes(endpoint).items()},
            checkpoint=str(directory/'epochs'/f'added_{added:03}.pt'),
            checkpoint_sha256=sha256(directory/'epochs'/f'added_{added:03}.pt'))
    result['multit_minus_control_pp']={k:delta(v,modes(endpoints['control'])[k])
                                     for k,v in modes(endpoints['multit']).items()}
    result['multit_minus_control_cost']={k:{name:v['cost'][name]-modes(endpoints['control'])[k]['cost'][name]
        for name in ('inference_equivalent_mac_sparse_proposal','inference_equivalent_ac')}
        for k,v in modes(endpoints['multit']).items()}
    path=suite/f'comparison_added_{added:03}.json'
    if path.exists():
        if read_json(path)!=result:raise ValueError('Existing milestone changed')
    else:atomic_json(path,result)
    return result
