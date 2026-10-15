#!/usr/bin/env python3
"""Train-main short IF calibration and full Dev evaluation; no training action."""
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
from wangdao.h2pro_calibration_v1.calibration import calibrate
from wangdao.h2pro_calibration_v1.workflow import checks,evaluate

BOOTSTRAP="""import sys,runpy
ann,snn=sys.argv[1:3]
sys.argv=['h2pro_calibration_v1.cache_worker']+sys.argv[3:]
sys.path.insert(0,ann)
import wangdao
wangdao.__path__.append(snn)
runpy.run_module('wangdao.h2pro_calibration_v1.cache_worker',run_name='__main__')
"""


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('check','prepare','calibrate','evaluate','all','status'))
    p.add_argument('--cache',type=Path,default=ROOT/'results/runs/h2pro_calibration_v1_features_fine')
    p.add_argument('--calibration-directory',type=Path,default=ROOT/'results/runs/h2pro_calibration_v1_fit_fine')
    p.add_argument('--run-directory',type=Path,default=ROOT/'results/runs/h2pro_calibration_v1_dev')
    p.add_argument('--dev-cache',type=Path,default=ROOT/'results/runs/h2pro_direct_v1_features')
    p.add_argument('--teacher-cache',type=Path,default=ROOT/'results/runs/h2pro_teacher_v1')
    p.add_argument('--dataset-root',type=Path,default=ROOT/'data/EV-UAV-dataset')
    p.add_argument('--chunk-rows',type=int,default=8192)
    a=p.parse_args()
    if a.action in ('check','all'):
        print(json.dumps(checks(),ensure_ascii=False,indent=2),flush=True)
    if a.action in ('prepare','all'):
        cmd=[sys.executable,'-B','-c',BOOTSTRAP,str(ANN_ROOT/'software'),str(ROOT/'software/wangdao'),
             '--dataset-root',str(a.dataset_root.absolute()),'--cache',str(a.cache.absolute()),
             '--teacher-cache',str(a.teacher_cache.absolute())]
        done=subprocess.run(cmd,cwd=str(ROOT),env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1'))
        if done.returncode:raise SystemExit(done.returncode)
    if a.action in ('calibrate','all'):
        result=calibrate(a.cache,a.calibration_directory)
        print(json.dumps({'status':result['status'],'calibration':str(a.calibration_directory/'calibration.json'),
                          'training_started':False}),flush=True)
    if a.action in ('evaluate','all'):
        result=evaluate(a.dev_cache,a.calibration_directory,a.run_directory,a.chunk_rows)
        print(json.dumps({'status':result['status'],'comparison':str(a.run_directory/'comparison.json'),
                          'training_started':False}),flush=True)
    if a.action=='status':
        result={}
        for name,directory in (('calibration',a.calibration_directory),('evaluation',a.run_directory)):
            path=directory/'progress.json'
            result[name]=json.loads(path.read_text('utf-8')) if path.exists() else {'status':'not_started'}
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
