"""An independent continuation identity; only the training objective differs."""
from dataclasses import asdict
from pathlib import Path
import hashlib

from ..h2pro_r1_lr_v1.common import (
    Config, DEFAULT_CACHE, SOURCE_RUN, SOURCE_EPOCH, IMPLEMENTATION, digest,
    atomic_json, lock_directory, output_path, read_json, sha256, independent_output,
    verify_prepared, inputs as source_inputs, initialized_payload as source_payload)
from ..h2pro_r1_train_v1.runtime import fresh_totals as base_totals, add_totals as base_add
from ..h2pro_adaptive_t_v2.core import Policy

PACKAGE = Path(__file__).resolve().parent
VERSION = 'h2pro_multit_train_v1'
DEFAULT_RUN = IMPLEMENTATION/'results/runs/h2pro_multit_v1_s42_ep159'
ARMS = ('control', 'multit')
STEPS = (4, 8, 16)
MILESTONES = (1, 5, 10, 20, 50, 100, 200)
EXIT_POLICY = Policy(.01, .99, .10, .95)
PREFIX_REFERENCE = IMPLEMENTATION/'results/runs/h2pro_adaptive_t_v1_s42_ep159'
GATE_REFERENCE = IMPLEMENTATION/'results/runs/h2pro_adaptive_t_v2_alternatives_s42_ep159'
PROTOCOL = dict(
    version=VERSION, source_epoch=159, source_checkpoint='epochs/epoch_159.pt',
    arms={'control': [16], 'multit': [4, 8, 16]}, initial_additional_epochs_per_arm=5,
    initialization='identical source FP32 masters, Adam moments/step43089, hyperparameters and all RNG',
    loss_control='0.5 CE16 + 0.5 temperature^2 KL(teacher || student16), temperature=2',
    loss_multit='arithmetic mean of the original CE+KD loss at T4, T8 and T16; equal 1/3 weights',
    prefix_training='shared 36->64 projection; each prefix count floor((T*h1+191)/255); derivative T/255',
    structure=[36,64,32,4], window_ms=100, theta=255, m0=191,
    readout='original class-wise two-prototype maximum; fixed decision threshold 0.35',
    unchanged=['all original features and candidate masks', 'ANN Proposal', 'H2-pro teacher',
               'int8 weight QAT', 'original fixed scales and uint8 quantizers', 'Adam lr1e-4',
               'batch8192', 'absolute epoch159+added_epoch seed42 shuffle', '100 ms closed windows'],
    train_coverage='all 2219864 Main candidates once per epoch; same batches and Adam update count in both arms',
    training_backend='dense FP64 exact count reference, FP32 masters/logits/Adam; no sparse GPU claim',
    evaluation='full Dev at fixed4/8/16 and the pre-existing fixed adaptive exit policy; no gate search',
    adaptive_policy=EXIT_POLICY.as_dict(), adaptive_policy_source='Main/Dev v2 candidate predating Test',
    best_pt='Dev fixedT16 IoU, tie by recall; source159 eligible',
    best_adaptive_pt='Dev fixed-policy adaptive IoU, tie by recall; source159 eligible',
    retention='all epochs: full weights/Adam/RNG, int8 export, full Dev prefix predictions and metrics',
    ac_regularization=False, two_step_exit=False, hard_mining=False, gradient_clipping=False,
    amplitude_stop=False, early_stopping=False, memory_cap=False, scheduler=False,
    roles=['train_main','train_dev'], test_parameter_selection=False, reserved_splits_read=False,
)


def sources():
    paths = list(PACKAGE.glob('*.py')) + [
        IMPLEMENTATION/'software/scripts/train_h2pro_multit.py', IMPLEMENTATION/'train_local_4060_multit.ps1']
    for folder, names in {
        'h2pro_r1_lr_v1': ('common.py','runtime.py'),
        'h2pro_adaptive_t_v1': ('core.py','selection.py','common.py'),
        'h2pro_adaptive_t_v2': ('core.py','scoring.py'),
    }.items():
        paths += [PACKAGE.parent/folder/name for name in names]
    return {p.relative_to(IMPLEMENTATION).as_posix():
            hashlib.sha256(p.read_text('utf-8').encode()).hexdigest() for p in sorted(paths)}


def inputs(cache, source, device):
    parent, origin, config, ready = source_inputs(cache, source, device)
    if parent['source_files_sha256']['epochs/epoch_159.pt'] != '3e3ef613c71d7eb8dfd55b1e3640db2fb09126d13b0e6a10e9bc3c3db36c98d9':
        raise ValueError('Require the exact previously frozen epoch159')
    selected = read_json(GATE_REFERENCE/'selection.json')
    if selected['selected']['accuracy']['policy'] != EXIT_POLICY.as_dict():
        raise ValueError('The fixed pre-Test Main gate selection changed')
    identity = {k:v for k,v in parent.items() if k not in ('protocol','sources')}
    identity.update(protocol=PROTOCOL, sources=sources(), development_references={
        str(p.relative_to(IMPLEMENTATION)):sha256(p) for p in (
            PREFIX_REFERENCE/'comparison.json', GATE_REFERENCE/'selection.json',
            GATE_REFERENCE/'main_seal.json', GATE_REFERENCE/'comparison.json')})
    return identity, origin, config, ready


def arm_config(config, arm):
    if arm not in ARMS: raise ValueError('Unknown training arm')
    return config.validate()


def arm_identity(identity, arm):
    config=arm_config(Config(**identity['config']),arm)
    return dict(suite=identity,arm=arm,config=asdict(config),trained_steps=[16] if arm=='control' else list(STEPS))


def fresh_totals():
    return dict(base_totals(), prefix_loss_sums={}, prefix_spikes={})


def add_totals(totals, result, seconds):
    base_add(totals,result,seconds)
    n=result['events']
    for t,values in result['prefix_losses'].items():
        old=totals['prefix_loss_sums'].setdefault(t,[0.,0.,0.])
        totals['prefix_loss_sums'][t]=[a+b*n for a,b in zip(old,values)]
        totals['prefix_spikes'][t]=totals['prefix_spikes'].get(t,0)+result['prefix_spikes'][t]


def initialized_payload(origin, identity, baseline):
    payload=source_payload(origin,identity,baseline)
    # Both arms retain the COMPLETE optimizer, including lr and step counters.
    if digest(payload['optimizer']) != digest(origin['optimizer']):
        raise AssertionError('Initialization changed Adam')
    state=payload['state'];state['totals']=fresh_totals()
    a=baseline['adaptive']['overall']
    state.update(best_adaptive_epoch=0,best_adaptive_score=[a['iou'],a['recall']])
    return payload
