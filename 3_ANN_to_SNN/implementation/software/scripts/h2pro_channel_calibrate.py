"""Direction 2: frozen original weights; Main calibration then Dev validation."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'software'))
from wangdao.h2pro_channel_calibration_v1.common import DEFAULT_CACHE,DEFAULT_RUN,read_json
from wangdao.h2pro_channel_calibration_v1.workflow import run


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('calibrate','evaluate','all','status'))
    parser.add_argument('--cache',type=Path,default=DEFAULT_CACHE)
    parser.add_argument('--run-directory',type=Path,default=DEFAULT_RUN)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--chunk-rows',type=int,default=32768)
    a=parser.parse_args()
    if a.chunk_rows<1:parser.error('chunk-rows must be positive')
    if a.action=='status':
        p=a.run_directory/'progress.json'
        result=read_json(p) if p.exists() else {'status':'not_started'}
        if 'cursor_file' in result and (a.run_directory/result['cursor_file']).exists():
            cursor=read_json(a.run_directory/result['cursor_file'])
            result.update(completed_coordinates=cursor['next_step'],total_coordinates=cursor['total_coordinates'],
                          evaluated_candidates=cursor['evaluated'])
    else:
        result=run(a.action,a.cache,a.run_directory,a.device,a.chunk_rows)
        result={k:result[k] for k in ('status','training_started','weights_updated') if k in result}
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
