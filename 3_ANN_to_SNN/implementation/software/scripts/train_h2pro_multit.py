"""Paired fixed16/multi4-8-16 continuation; real training only via train action."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from wangdao.h2pro_multit_v1.common import DEFAULT_CACHE,DEFAULT_RUN,SOURCE_RUN,output_path
    from wangdao.h2pro_multit_v1.runtime import status,train_suite
    from wangdao.h2pro_multit_v1.checks import check
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('check','train','pause','status'))
    p.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--source',type=Path,default=SOURCE_RUN)
    p.add_argument('--suite',type=Path,default=DEFAULT_RUN)
    p.add_argument('--additional-epochs',type=int,default=5,
                   help='Total added epochs per arm after159; repeated calls resume, not add again')
    p.add_argument('--checkpoint-batches',type=int,default=10)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    a=p.parse_args();a.suite=output_path(a.suite)
    if a.action=='status':result=status(a.suite)
    elif a.action=='pause':
        if not (a.suite/'ready.json').exists():raise ValueError('Suite not prepared')
        (a.suite/'PAUSE').write_text('User requested whole-suite pause\n',encoding='utf-8')
        result=dict(status='pause_requested',suite=str(a.suite))
    elif a.action=='check':
        ready=check(a.cache,a.suite,a.source,a.device)
        result={k:ready[k] for k in ('status','formal_training_started','real_data_optimizer_steps',
            'initial_adam_step','full_dev_candidates_exact_per_prefix','memory_by_arm')}
        result.update(preparation_record_only=True,current_status=status(a.suite))
    else:result=train_suite(a.cache,a.suite,a.source,a.additional_epochs,a.checkpoint_batches,a.device)
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
