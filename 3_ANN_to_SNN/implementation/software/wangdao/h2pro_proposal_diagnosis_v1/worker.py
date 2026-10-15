from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .common import read_json


def main():
    parser = argparse.ArgumentParser(description='Direction 7 frozen-weight diagnosis; no training')
    parser.add_argument('action', choices=('check', 'all', 'status', 'verify'))
    parser.add_argument('--teacher-cache', type=Path, required=True)
    parser.add_argument('--run-directory', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.action in ('check', 'all'):
        from .validation import checks
        print(json.dumps(checks(), indent=2), flush=True)
    if args.action == 'all':
        from .workflow import evaluate
        report = evaluate(args.teacher_cache, args.run_directory)
        print(json.dumps({'status': report['status'], 'verdict': report['verdict'],
                          'comparison': str(args.run_directory/'comparison.json'),
                          'training_started': False}, indent=2), flush=True)
    if args.action == 'verify':
        from .verify import verify
        print(json.dumps(verify(args.teacher_cache, args.run_directory), indent=2), flush=True)
    if args.action == 'status':
        p = args.run_directory/'comparison.json'
        if p.exists():
            report = read_json(p)
            print(json.dumps({'status': report['status'], 'verdict': report['verdict']}, indent=2))
        else:
            print(json.dumps({'status': 'not_complete',
                              'completed_sequences': len(list((args.run_directory/'cache').rglob('*.json')))}))


if __name__ == '__main__':
    main()
