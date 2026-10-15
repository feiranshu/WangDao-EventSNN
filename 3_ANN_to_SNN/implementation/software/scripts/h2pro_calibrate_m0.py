"""Frozen r1/T16 m0-only calibration with sequence-disjoint Main validation."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'software'))
from wangdao.h2pro_m0_calibration_v2.common import DEFAULT_CACHE,DEFAULT_RUN,read_json
from wangdao.h2pro_m0_calibration_v2.workflow import run


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('all','calibrate','evaluate','status'))
    p.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    p.add_argument('--run-directory',type=Path,default=DEFAULT_RUN)
    p.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    p.add_argument('--chunk-rows',type=int,default=32768)
    a=p.parse_args()
    if a.chunk_rows<1:p.error('chunk-rows must be positive')
    if a.action=='status':
        path=a.run_directory/'progress.json'
        r=read_json(path) if path.exists() else {'status':'not_started'}
    else:
        r=run(a.action,a.cache,a.run_directory,a.device,a.chunk_rows)
        r={k:r[k] for k in ('status','training_started','weights_updated') if k in r}
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
