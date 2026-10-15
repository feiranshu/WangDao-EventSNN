"""Main-only source loading, frozen provenance and independent output ownership."""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import numpy as np

from ..h2pro_snn.foundation import IMPLEMENTATION, verify_baseline, role_records, read_json, sha256
from ..h2pro_direct_v1.common import lock_directory, output_path, atomic_json

PACKAGE = Path(__file__).resolve().parent
DEFAULT_CACHE = IMPLEMENTATION/'results/runs/h2pro_distill_v1_features'
DEFAULT_RUN = IMPLEMENTATION/'results/runs/h2pro_channel_calibration_v1'
PRIOR = IMPLEMENTATION/'results/runs/h2pro_task_calibration_v1_fit/calibration.json'
PROTOCOL = {
    'version': 'h2pro_channel_calibration_v1', 'direction': 2,
    'points': [['r1',16],['r2',8]],
    'branches': ['shared','group_m0','group_joint','channel_m0','channel_joint'],
    'weights': 'unchanged frozen H2-pro original int8 matrices and FP32 output scales',
    'unchanged': ['ANN Proposal','36-64-32-4','original features','uint8 clip/round','two prototypes per class','decision threshold 0.35'],
    'training_enabled': False, 'optimizer_steps': 0, 'backward_calls': 0,
    'sample': {'seed':42, 'max_per_class_per_main_sequence':512,
               'weights':'inverse class-stratum inclusion probability; exact class population totals including Proposal rejects'},
    'grouping': 'sort original ANN hidden channels by sampled Main mean activation; tie by channel index; consecutive blocks of eight; no Dev data',
    'threshold_grid': [128,160,192,224,255,288,320,384,448,510],
    'phase_quarters': [0,1,2,3], 'coordinate_passes': 2,
    'local_threshold_offsets': [-16,-8,-4,-2,-1,0,1,2,4,8,16],
    'local_membrane_offsets': [-16,-8,-4,-2,-1,0,1,2,4,8,16],
    'search_objective': 'maximize mean of sampled full-population ordinary and HD IoU; tie by lower teacher-correct harm, margin MSE, estimated AC, parameter tuple',
    'initialization': 'shared starts from best prior Main-derived shared candidate; group starts from shared; channels start from corresponding group result',
    'full_main_selection': ['iou','recall_preserving'],
    'iou': 'highest full Main overall IoU, tie by HD recall, overall recall, lower AC, parameter key',
    'recall_preserving': 'same ranking, requiring full Main overall and HD TP >= frozen ANN at 0.35; report infeasible rather than relax',
    'candidate_pool': 'initial controls, branch initial state, accepted coordinate states; full Main scores all distinct accepted states after a completed layer sweep plus each branch final',
    'freeze': 'complete both points and every branch on Main and seal selection before loading Dev arrays',
    'decoding': 'single theta preserves historical decode order; mixed theta uses sum_g(theta_g*sum_i_in_g(count_i*W_i)), then original scale/T/255',
    'cost': 'original int8 synaptic MAC/AC separated from threshold bucket decode multiplies/adds, state, scales and reference execution',
    'backend': 'GPU FP64 exact integer-count reference for search/evaluation; explicit original-weight binary spike-add verification on fixed rows from every sequence',
    'roles': ['train_main','train_dev'], 'reserved_splits_read': False,
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    paths = list(PACKAGE.glob('*.py'))+[IMPLEMENTATION/'software/scripts/h2pro_channel_calibrate.py', IMPLEMENTATION/'calibrate_h2pro_channel.ps1']
    for folder, names in {'h2pro_direct_v1':['core.py','common.py'], 'h2pro_calibration_v1':['core.py'],
        'h2pro_distill_v1':['evaluation.py'], 'h2pro_snn':['foundation.py','stream_data.py','diagnostics.py']}.items():
        paths += [PACKAGE.parent/folder/name for name in names]
    return {p.relative_to(IMPLEMENTATION).as_posix(): hashlib.sha256(p.read_text('utf-8').encode()).hexdigest() for p in sorted(paths)}


def identity(cache, engine):
    return {'protocol':PROTOCOL,'baseline':verify_baseline(),'sources':source_hashes(),
            'cache_manifest_sha256':sha256(Path(cache)/'manifest.json'),'prior_main_calibration_sha256':sha256(PRIOR),
            'engine':engine}


def load_role(cache, role):
    if role not in ('train_main','train_dev'):
        raise ValueError('Only Main/Dev roles supported')
    cache = Path(cache); m = read_json(cache/'manifest.json')
    if m['status'] != 'complete' or m['identity'] != read_json(cache/'identity.json'):
        raise ValueError('Feature cache identity mismatch')
    rows, declared = m['roles'][role], role_records(role)
    if len(rows) != len(declared):
        raise ValueError('Incomplete role')
    seqs=[]; offset=0
    for row, original in zip(rows, declared):
        if row['role'] != role or row['sequence'] != original['path'] or row['source_sha256'] != original['sha256']:
            raise ValueError('Cache sequence/role mismatch')
        path=cache/row['cache_file']
        if path.parent != cache or sha256(path) != row['cache_sha256']:
            raise ValueError('Cache path/hash mismatch')
        with np.load(path,allow_pickle=False) as z:
            a={k:z[k] for k in z.files}
        x,y,c,teacher=(a[k] for k in ('features','labels','candidate','teacher'))
        if (x.dtype != np.uint8 or x.shape != (row['candidate_events'],36) or c.dtype != np.bool_
                or y.shape != c.shape or y.shape != (row['events'],) or not np.isin(y,[0,1]).all()
                or int(y.sum()) != row['target_events'] or int(c.sum()) != len(x)
                or int(y[c].sum()) != row['candidate_targets'] or teacher.shape != (len(x),4)
                or teacher.dtype != np.float32 or not np.isfinite(teacher).all()):
            raise ValueError('Invalid original feature/teacher arrays')
        seqs.append({'row':row,'features':x,'labels':y,'candidate':c,'teacher':teacher,'lo':offset,'hi':offset+len(x)})
        offset+=len(x)
    return seqs


def save_arrays(path, **a):
    path=Path(path); temp=path.with_suffix('.tmp')
    with temp.open('wb') as f:
        np.savez_compressed(f,**a); f.flush(); os.fsync(f.fileno())
    os.replace(temp,path)


def tree_hashes(path):
    path=Path(path)
    return {p.relative_to(IMPLEMENTATION).as_posix():sha256(p) for p in sorted(path.rglob('*')) if p.is_file()}


def parameter_key(encoders):
    return tuple(v for e in encoders for v in e.theta+e.membrane)


def parameter_id(encoders):
    return hashlib.sha256(np.asarray(parameter_key(encoders),'<i8').tobytes()).hexdigest()[:20]
