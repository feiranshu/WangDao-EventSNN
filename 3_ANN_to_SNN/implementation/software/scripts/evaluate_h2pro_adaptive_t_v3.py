"""Frozen epoch159 T8 single-factor calibration and verification; no training."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from wangdao.h2pro_adaptive_t_v3.common import DEFAULT_RUN,DEFAULT_CACHE
from wangdao.h2pro_adaptive_t_v3.experiment import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('run','main','dev','status'),default='run')
    p.add_argument('--run-directory',type=Path,default=DEFAULT_RUN)
    p.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--batch-size',type=int,default=8192)
    a=p.parse_args()
    print(json.dumps(run(a.action,a.run_directory,a.cache,a.batch_size),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
