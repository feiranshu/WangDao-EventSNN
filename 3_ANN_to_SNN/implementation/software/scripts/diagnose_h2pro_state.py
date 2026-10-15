#!/usr/bin/env python3
"""Frozen readout/state study. No optimizer, checkpoint writes or reserved splits."""
import argparse
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT/'software'))

import torch
from wangdao.h2pro_snn.stream_data import SequenceStore, atomic_json, run_path
from wangdao.h2pro_snn.training import checkpoint_load, environment, memory_stats, setup_device
from wangdao.h2pro_snn_v2_2.foundation import research_sources as snn_sources, sha256
from wangdao.h2pro_snn_v2_2.refiner import RefinerConfig as SNNConfig
from wangdao.h2pro_ann_state_v1.foundation import research_sources as ann_sources
from wangdao.h2pro_ann_state_v1.refiner import RefinerConfig as ANNConfig
from wangdao.h2pro_ann_state_v1.probe import ReadoutProbe
from wangdao.h2pro_ann_state_v1.audit import evaluate_probe, hard_prefixes
from wangdao.h2pro_ann_state_v1.stress import frozen_train_sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--cache', type=Path, default=ROOT/'results/runs/h2pro_teacher_v1')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--analog-transplant', action='store_true')
    parser.add_argument('--full-train-health', action='store_true')
    args = parser.parse_args()
    output = run_path(args.output_directory)
    if output.exists():
        parser.error('Use a fresh output directory')
    saved = checkpoint_load(args.checkpoint)
    if 'evaluation' not in saved or saved.get('epoch') != 1:
        parser.error('This registered diagnostic uses a completed epoch-1 best checkpoint')
    profile = saved['identity']['profile']
    is_ann = profile['diagnostic_revision'] == 'ann_state_v1'
    sources = ann_sources() if is_ann else snn_sources()
    if saved['identity']['sources'] != sources:
        raise ValueError('Checkpoint numerical sources changed')
    if args.analog_transplant and is_ann:
        parser.error('Analog transplant applies only to the preserved SNN checkpoint')
    if args.full_train_health and not is_ann:
        parser.error('Full ANN health sweep requires the matched trained ANN checkpoint')
    config = profile['training']
    device = setup_device(args.device, config)
    store = SequenceStore(args.cache, config['cache_mib'])
    if sha256(store.directory/'manifest.json') != saved['identity']['teacher_cache_sha256']:
        raise ValueError('Teacher cache differs from checkpoint')
    cls = ANNConfig if is_ann else SNNConfig
    model = ReadoutProbe(cls(**profile['model']), saved['foreground_prior'], analog=is_ann or args.analog_transplant).to(device)
    model.load_state_dict(saved['model']); model.eval()
    output.mkdir(parents=True)
    if device.type == 'cuda': torch.cuda.reset_peak_memory_stats(device)
    hard = hard_prefixes(model, store, device, saved['threshold'])
    atomic_json(output/'hard_prefixes.json', hard)
    evaluation = evaluate_probe(model, store, device, config, saved['threshold'], output/'dev_scores.npz')
    exact = evaluation['readouts']['original']['selected'] == saved['evaluation']['selected']
    if not args.analog_transplant and not exact:
        raise AssertionError('Original checkpoint evaluation failed exact reproduction')
    sweep = frozen_train_sweep(model, store, device, config) if args.full_train_health else None
    if sweep is not None:
        atomic_json(output/'frozen_train_sweep.json', sweep)
    result = {'status': 'frozen_diagnosis_complete', 'checkpoint': str(args.checkpoint.absolute()),
              'checkpoint_sha256': sha256(args.checkpoint), 'source_checkpoint_reproduced': exact,
              'analog_transplant': args.analog_transplant, 'evaluation': evaluation, 'hard_prefixes': hard,
              'frozen_train_sweep': sweep,
              'environment': environment(device), 'checkpoint_sources': sources, 'diagnostic_sources': ann_sources(),
              'entrypoint_sha256': sha256(Path(__file__)), 'parameter_updates': 0, **memory_stats(device)}
    atomic_json(output/'audit.json', result)
    print({k: evaluation['readouts'][k]['selected']['overall']['iou'] for k in evaluation['readouts']}, flush=True)


if __name__ == '__main__':
    main()
