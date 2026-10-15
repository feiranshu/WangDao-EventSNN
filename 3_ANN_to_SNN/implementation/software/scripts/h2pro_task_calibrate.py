#!/usr/bin/env python3
"""Supervised short-time encoder calibration. No training or weight updates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT / 'software'))
from wangdao.h2pro_task_calibration_v1.common import inputs
from wangdao.h2pro_task_calibration_v1.calibration import calibrate
from wangdao.h2pro_task_calibration_v1.workflow import checks, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'prepare', 'calibrate', 'evaluate', 'all', 'status'))
    parser.add_argument('--cache', type=Path, default=ROOT / 'results/runs/h2pro_calibration_v1_features_fine')
    parser.add_argument('--prior-calibration', type=Path, default=ROOT / 'results/runs/h2pro_calibration_v1_fit_fine')
    parser.add_argument('--calibration-directory', type=Path, default=ROOT / 'results/runs/h2pro_task_calibration_v1_fit')
    parser.add_argument('--run-directory', type=Path, default=ROOT / 'results/runs/h2pro_task_calibration_v1_dev')
    parser.add_argument('--dev-cache', type=Path, default=ROOT / 'results/runs/h2pro_direct_v1_features')
    parser.add_argument('--chunk-rows', type=int, default=8192)
    args = parser.parse_args()
    if args.action in ('check', 'all'):
        print(json.dumps(checks(), ensure_ascii=False, indent=2), flush=True)
    if args.action in ('prepare', 'all'):
        manifest, previous = inputs(args.cache, args.prior_calibration)
        print(json.dumps({'status': 'ready', 'reused_train_main_sequences': len(manifest['sequences']),
            'prior_points': len(previous['points']), 'training_started': False}), flush=True)
    if args.action in ('calibrate', 'all'):
        result = calibrate(args.cache, args.prior_calibration, args.calibration_directory)
        print(json.dumps({'status': result['status'], 'calibration': str(args.calibration_directory / 'calibration.json'),
                          'training_started': False}), flush=True)
    if args.action in ('evaluate', 'all'):
        result = evaluate(args.dev_cache, args.calibration_directory, args.run_directory, args.chunk_rows)
        print(json.dumps({'status': result['status'], 'comparison': str(args.run_directory / 'comparison.json'),
                          'training_started': False}), flush=True)
    if args.action == 'status':
        result = {}
        for name, directory in (('calibration', args.calibration_directory), ('evaluation', args.run_directory)):
            path = directory / 'progress.json'
            result[name] = json.loads(path.read_text('utf-8')) if path.exists() else {'status': 'not_started'}
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
