"""Prepare/check paired ANN adaptation; only the explicit train action updates real data."""
import os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse
import json
from pathlib import Path
from dynamic_ann.common import Config,DEFAULT_CACHE,DEFAULT_SUITE,local_output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','check','train','pause','status'))
    parser.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    parser.add_argument('--run-directory',type=Path,default=DEFAULT_SUITE)
    parser.add_argument('--epochs',type=int,default=5,help='Total completed epochs per arm, including existing progress')
    parser.add_argument('--checkpoint-batches',type=int,default=10)
    parser.add_argument('--source-batch-size',type=int,default=32768)
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args()
    config=Config(learning_rate=args.learning_rate,source_batch_size=args.source_batch_size).validate()
    if args.action=='prepare':
        from dynamic_ann.data import prepare
        result=prepare(args.cache,args.workers)['stats']
    elif args.action=='check':
        from dynamic_ann.checks import check
        ready=check(args.cache,args.run_directory,config,args.device)
        result={k:ready[k] for k in ('status','real_data_optimizer_steps','steps_per_epoch','arms')}
    elif args.action=='train':
        from dynamic_ann.runtime import train
        result=train(args.cache,args.run_directory,config,args.epochs,args.checkpoint_batches,args.device)
    elif args.action=='pause':
        suite=local_output(args.run_directory)
        if not suite.is_dir():raise ValueError('Run directory not prepared')
        (suite/'PAUSE').write_text('Pause requested by user; checkpoint at next safe boundary.\n',encoding='utf-8')
        result={'pause_requested':True,'suite':str(suite)}
    else:
        from dynamic_ann.runtime import suite_status
        result=suite_status(args.run_directory)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),flush=True)


if __name__=='__main__':main()
