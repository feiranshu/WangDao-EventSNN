#!/usr/bin/env python3
"""Frozen H2-pro layerwise conversion. No optimizer, training or new weights."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).absolute().parents[2]
sys.path.insert(0,str(ROOT/'software'))
from wangdao.h2pro_snn.foundation import ANN_ROOT
from wangdao.h2pro_direct_v1.workflow import checks,evaluate

BOOTSTRAP="""import sys,runpy
ann,snn=sys.argv[1:3]
sys.argv=['h2pro_direct_v1.cache_worker']+sys.argv[3:]
sys.path.insert(0,ann)
import wangdao
wangdao.__path__.append(snn)
runpy.run_module('wangdao.h2pro_direct_v1.cache_worker',run_name='__main__')
"""


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('check','prepare','evaluate','status'))
    p.add_argument('--cache',type=Path,default=ROOT/'results/runs/h2pro_direct_v1_features')
    p.add_argument('--run-directory',type=Path,default=ROOT/'results/runs/h2pro_direct_v1_dev')
    p.add_argument('--teacher-cache',type=Path,default=ROOT/'results/runs/h2pro_teacher_v1')
    p.add_argument('--dataset-root',type=Path,default=ROOT/'data/EV-UAV-dataset')
    p.add_argument('--steps',type=int,nargs='+',default=[4,8,16,32,64,255])
    p.add_argument('--backend',choices=('count_reference','spike_add'),default='count_reference')
    p.add_argument('--chunk-rows',type=int,default=8192)
    a=p.parse_args()
    if a.action=='prepare':
        command=[sys.executable,'-B','-c',BOOTSTRAP,str(ANN_ROOT/'software'),str(ROOT/'software/wangdao'),
                 '--dataset-root',str(a.dataset_root.absolute()),'--cache',str(a.cache.absolute()),
                 '--teacher-cache',str(a.teacher_cache.absolute())]
        done=subprocess.run(command,cwd=str(ROOT),env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1'))
        raise SystemExit(done.returncode)
    if a.action=='check':
        result=checks()
    elif a.action=='evaluate':
        result=evaluate(a.cache,a.run_directory,a.steps,a.backend,a.chunk_rows)
        result={'status':result['status'],'training_started':False,'comparison':str(a.run_directory/'comparison.json')}
    else:
        path=a.run_directory/'progress.json'
        result=json.loads(path.read_text('utf-8')) if path.exists() else {'status':'not_started'}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
