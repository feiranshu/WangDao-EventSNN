"""Frozen-weight 4/8/16 exit experiment. This script has no training action."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wangdao.h2pro_adaptive_t_v1.common import DEFAULT_CACHE,DEFAULT_RUN
from wangdao.h2pro_adaptive_t_v1.experiment import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('run','main','dev','status'),default='run')
    p.add_argument('--run-directory',type=Path,default=DEFAULT_RUN)
    p.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=8192)
    a=p.parse_args()
    print(json.dumps(run(a.action,a.run_directory,a.cache,a.device,a.batch_size),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
