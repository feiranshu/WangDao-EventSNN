"""Isolated frozen-ANN namespace entrypoint; inference only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import read_json, teacher_manifest, lineage


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('check', 'prepare', 'evaluate', 'all', 'status'))
    parser.add_argument('--teacher-cache', type=Path, required=True)
    parser.add_argument('--run-directory', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action in ('check', 'all'):
        from .validation import checks
        print(json.dumps(checks(), ensure_ascii=False, indent=2), flush=True)
    if args.action == 'prepare':
        m = teacher_manifest(args.teacher_cache)
        print(json.dumps({'status': 'ready', 'sequences': len(m['roles']['train_dev']),
                          'lineage': lineage(), 'training_started': False}), flush=True)
    if args.action in ('evaluate', 'all'):
        from .workflow import evaluate
        result = evaluate(args.teacher_cache, args.run_directory)
        print(json.dumps({'status': result['status'], 'training_started': False,
                          'comparison': str(args.run_directory / 'comparison.json'),
                          'results': [{k: r[k] for k in ('mode', 'overall', 'high_density',
                           'executed_neural_mac', 'executed_neural_ac')} for r in result['results']]}, indent=2), flush=True)
    if args.action == 'status':
        p = args.run_directory / 'progress.json'
        print(json.dumps(read_json(p) if p.exists() else {'status': 'not_started'}, indent=2))


if __name__ == '__main__':
    main()
