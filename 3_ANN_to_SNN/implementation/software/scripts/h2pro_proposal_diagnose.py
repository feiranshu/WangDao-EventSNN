"""Isolated frozen-ANN namespace for direction 7 routing/context diagnosis."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[2]
ANN = ROOT.parents[1]/'1_H2_model_improvement/implementation'
BOOTSTRAP = '''import sys,runpy
ann,snn=sys.argv[1:3]
sys.argv=['h2pro_proposal_diagnosis_v1.worker']+sys.argv[3:]
sys.path.insert(0,ann)
import wangdao
wangdao.__path__.append(snn)
runpy.run_module('wangdao.h2pro_proposal_diagnosis_v1.worker',run_name='__main__')
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('check', 'all', 'status', 'verify'))
    p.add_argument('--teacher-cache', type=Path, default=ROOT/'results/runs/h2pro_teacher_v1')
    p.add_argument('--run-directory', type=Path, default=ROOT/'results/runs/h2pro_proposal_diagnosis_v1')
    a = p.parse_args()
    command = [sys.executable, '-B', '-c', BOOTSTRAP, str(ANN/'software'), str(ROOT/'software/wangdao'),
               a.action, '--teacher-cache', str(a.teacher_cache.absolute()),
               '--run-directory', str(a.run_directory.absolute())]
    result = subprocess.run(command, cwd=ROOT,
                            env=dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1'))
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
