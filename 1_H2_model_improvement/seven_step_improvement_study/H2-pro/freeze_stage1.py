"""Freeze one Train-dev-selected causal ANN candidate without touching A4-2."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import numpy as np

from run_stage1 import ROOT, INDEX, PROPOSAL, SOURCE_FLOAT, SOURCE_INT8, sha256, VARIANTS


def freeze(run: Path, selection: Path, variant: str, output: Path):
    if variant not in VARIANTS:
        raise ValueError('unregistered feature variant')
    try:
        output.absolute().relative_to(ROOT.absolute())
    except ValueError:
        raise ValueError('all new model files must stay under H2-pro')
    if output.exists():
        raise FileExistsError(output)
    if not (run/'training_complete.json').is_file():
        raise ValueError('all three matched training runs must complete first')
    decision = json.loads(selection.read_text(encoding='utf-8'))
    if decision['protocol_sha256'] != sha256(ROOT/'protocol.json'):
        raise ValueError('selection used another feature protocol')
    candidate = decision['candidates'][variant]
    winner = candidate['winner']
    if winner is None:
        raise ValueError('candidate has no eligible Train-dev working point')
    if 'delta_from_C0_same_point' in candidate:
        delta = candidate['delta_from_C0_same_point']
    else:
        control_progress = json.loads((run/'C0-zero'/'progress.json').read_text(encoding='utf-8'))
        control_epoch = next(row for row in control_progress if row['epoch'] == winner['epoch'])
        control = control_epoch['int8']['curve'][winner['threshold_key']]
        metric = winner['metrics']
        delta = {
            'iou': metric['event']['iou'] - control['event']['iou'],
            'ap': winner['ap'] - control_epoch['int8']['ap'],
            'high_density_recall': metric['high_density_recall'] - control['high_density_recall'],
        }
    if variant != 'C0-zero' and (delta['iou'] < 0 or delta['ap'] < 0 or
                                 delta['high_density_recall'] is None or
                                 delta['high_density_recall'] < -0.003):
        raise ValueError('candidate fails IoU/AP/high-density-Recall protection')
    epoch = int(winner['epoch'])
    threshold = float(winner['threshold'])
    model_id = 'H2-pro-Causal-ANN-v1'
    raw_float = run/variant/f'epoch{epoch:03}.pt'
    raw_int8 = run/variant/f'epoch{epoch:03}_int8.npz'
    if (sha256(raw_float) != winner['checkpoint_sha256'] or
            sha256(raw_int8) != winner['int8_sha256']):
        raise ValueError('selected artifact hash mismatch')
    output.mkdir(parents=True)
    target_float = output/'teacher_float.pt'
    target_int8 = output/'teacher_exact_int8.npz'
    shutil.copyfile(raw_float,target_float)
    with np.load(raw_int8,allow_pickle=False) as archive:
        arrays = {key:archive[key].copy() for key in archive.files}
    metadata = {'model_id':model_id,'variant':variant,'epoch':epoch,'threshold':threshold,
                'feature_protocol':'H2-pro-stage1-v1','window_ms':100,
                'input_dtype':'uint8','dimensions':[36,64,32,4],
                'numeric_path':'uint8-int8-int32 per-output-channel exact replay',
                'source_model':'A4-2 seed42 epoch8',
                'selection_role':'Train-dev','selection_amendment':'v2 post-v1 exploratory',
                'acceptance_status':'not_yet_evaluated'}
    arrays['model_id'] = np.asarray(model_id)
    arrays['epoch'] = np.asarray(epoch,dtype=np.int64)
    arrays['threshold'] = np.asarray(threshold,dtype=np.float64)
    arrays['metadata_json'] = np.asarray(json.dumps(metadata,sort_keys=True))
    np.savez_compressed(target_int8,**arrays)
    manifest = {'model_id':model_id,'metadata':metadata,
                'selection_sha256':sha256(selection),
                'selection_protocol_sha256':decision.get('selection_protocol_sha256'),
                'historical_reference_sha256':decision.get('historical_reference_sha256'),
                'protocol_sha256':sha256(ROOT/'protocol.json'),
                'index_sha256':sha256(INDEX),
                'source_weights_sha256':{p.name:sha256(p) for p in
                                         (PROPOSAL,SOURCE_FLOAT,SOURCE_INT8)},
                'teacher_float_sha256':sha256(target_float),
                'teacher_int8_sha256':sha256(target_int8),
                'candidate_delta_from_C0':delta,
                'paired_bootstrap_vs_C0':candidate.get('paired_bootstrap_vs_C0',candidate.get('paired_bootstrap')),
                'paired_bootstrap_vs_historical':candidate.get('paired_bootstrap_vs_historical'),
                'status':'frozen development candidate; reserved-data acceptance pending'}
    (output/'model_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(freeze(a.run,a.selection,a.variant,a.output),indent=2))
