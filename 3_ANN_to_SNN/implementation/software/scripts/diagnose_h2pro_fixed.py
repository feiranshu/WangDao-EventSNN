"""Fixed r2/T8 inference only: seal Main thresholds, then validate on Dev."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from wangdao.h2pro_fixed_diagnosis_v1.workflow import DEFAULT_CACHE, DEFAULT_RUN, DEFAULT_OUTPUT, run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('main', 'dev', 'all'), default='all', nargs='?')
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--source', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--device', choices=('cuda','cpu'), default='cuda')
    parser.add_argument('--batch-size', type=int, default=8192)
    args = parser.parse_args()
    result = run(args.cache, args.source, args.output, args.action, args.device, args.batch_size)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
