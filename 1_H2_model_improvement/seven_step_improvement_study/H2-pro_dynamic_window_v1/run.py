"""Independent ANN experiment. No training, weight updates or reserved data access."""
import os

# Small exact BLAS matrices: avoid many competing BLAS pools across sequences.
# This limits host worker oversubscription, not data coverage or model activity.
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'

import argparse
from pathlib import Path
from src.workflow import run_stage, audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['main','dev','audit','all'])
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'results/runs/frozen_ann_v1')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    if args.workers<1:raise ValueError('workers must be positive')
    if args.stage in ('main','all'):run_stage(args.output,'train_main',args.workers)
    if args.stage in ('dev','all'):run_stage(args.output,'train_dev',args.workers)
    if args.stage in ('audit','all'):audit(args.output)


if __name__=='__main__':
    main()
