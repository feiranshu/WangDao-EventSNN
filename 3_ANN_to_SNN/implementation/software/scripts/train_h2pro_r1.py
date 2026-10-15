"""Independent r1/T16 supervised/distillation experiment; train is user-invoked."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
ROOT = Path(__file__).absolute().parents[2]
sys.path.insert(0, str(ROOT/'software'))

PREPARE_BOOTSTRAP = '''import sys
ann,snn,teacher,evidence,output=sys.argv[1:]
sys.path.insert(0,ann)
import wangdao
wangdao.__path__.append(snn)
from wangdao.h2pro_distill_v1.prepare import prepare
r=prepare(teacher,evidence,output)
print({'status':r['status'],'formal_training_started':False})
'''


def main():
    from wangdao.h2pro_r1_train_v1.common import (Config, DEFAULT_CACHE, DEFAULT_EVIDENCE,
                                               DEFAULT_RUN, DEFAULT_TEACHER, output_path, read_json)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'check', 'train', 'pause', 'status'))
    p.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    p.add_argument('--run-directory', type=Path, default=DEFAULT_RUN)
    p.add_argument('--teacher', type=Path, default=DEFAULT_TEACHER)
    p.add_argument('--evidence', type=Path, default=DEFAULT_EVIDENCE)
    p.add_argument('--epochs', type=int, default=5, help='TOTAL target epochs, including completed epochs')
    p.add_argument('--batch-size', type=int, default=8192)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--kd-alpha', type=float, default=.5)
    p.add_argument('--temperature', type=float, default=2.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--checkpoint-batches', type=int, default=10)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    a = p.parse_args()
    config = Config(a.seed, a.batch_size, a.learning_rate, a.kd_alpha, a.temperature).validate()
    directory = output_path(a.run_directory)
    if a.action == 'prepare':
        ann = ROOT.parents[1]/'1_H2_model_improvement/implementation/software'
        command = [sys.executable, '-B', '-c', PREPARE_BOOTSTRAP, str(ann),
                   str(ROOT/'software/wangdao'), str(a.teacher.resolve()),
                   str(a.evidence.resolve()), str(output_path(a.cache))]
        raise SystemExit(subprocess.run(command, cwd=ROOT).returncode)
    if a.action == 'status':
        file = directory/'progress.json'
        print(json.dumps(read_json(file) if file.exists() else {'status': 'not_prepared'}, indent=2))
        return
    if a.action == 'pause':
        if not (directory/'identity.json').exists():
            p.error('This run has not been prepared')
        (directory/'PAUSE').write_text('User pause request\n', encoding='utf-8')
        print('Pause requested. Wait for status=paused before closing the training terminal.')
        return
    if a.action == 'check':
        from wangdao.h2pro_r1_train_v1.checks import check
        result = check(a.cache, directory, config, a.device)
        print(json.dumps({k: result[k] for k in ('status', 'formal_training_started', 'data', 'memory',
                         'all_dev_candidates_exact_original_r1t16', 'synthetic_resume')}, indent=2))
    else:
        from wangdao.h2pro_r1_train_v1.runtime import train
        train(a.cache, directory, config, a.epochs, a.checkpoint_batches, a.device)


if __name__ == '__main__':
    main()
