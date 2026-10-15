"""Direction5 paired continuation. Only the explicit train action updates weights."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from wangdao.h2pro_hardcase_v1.common import DEFAULT_CACHE, DEFAULT_RUN, SOURCE_RUN, DEFAULT_DIAGNOSIS, output_path
    from wangdao.h2pro_hardcase_v1.runtime import status,train_suite
    from wangdao.h2pro_hardcase_v1.checks import check
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('check','train','pause','status'))
    parser.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    parser.add_argument('--suite',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--source',type=Path,default=SOURCE_RUN)
    parser.add_argument('--diagnosis',type=Path,default=DEFAULT_DIAGNOSIS)
    parser.add_argument('--additional-epochs',type=int,default=20,help='TOTAL additional epochs per arm after source epoch88, not a further increment')
    parser.add_argument('--checkpoint-batches',type=int,default=10)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    args=parser.parse_args()
    args.suite=output_path(args.suite)
    if args.action=='pause':
        if not (args.suite/'ready.json').exists():
            raise ValueError('Suite is not prepared')
        (args.suite/'PAUSE').write_text('User requested pause of entire paired suite\n',encoding='utf-8')
        result={'status':'pause_requested','suite':str(args.suite)}
    elif args.action=='status':
        result=status(args.suite)
    elif args.action=='check':
        ready=check(args.cache,args.suite,args.source,args.diagnosis,args.device)
        result={k:ready[k] for k in ('status','formal_training_started','real_data_optimizer_steps',
            'initial_adam_step','full_main_predictions_exact','full_dev_metrics_and_cost_exact','memory')}
        result['hard_category_counts']=ready['hard_mining']['counts']
    else:
        result=train_suite(args.cache,args.suite,args.source,args.diagnosis,args.additional_epochs,
                           args.checkpoint_batches,args.device)
    print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
