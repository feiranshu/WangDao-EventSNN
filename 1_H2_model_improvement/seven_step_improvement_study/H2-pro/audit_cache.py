"""Verify every development cache against the frozen index and write a compact manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from run_stage1 import ROOT, INDEX, VARIANTS, _records, sha256


def audit(cache: Path, output: Path):
    if output.exists():
        raise FileExistsError(output)
    summary = {'index_sha256':sha256(INDEX),'protocol_sha256':sha256(ROOT/'protocol.json'),
               'roles':{}}
    for role in ('train_main','train_dev'):
        records = {variant:_records(cache,role,variant) for variant in VARIANTS}
        rows = []
        for triplet in zip(*(records[variant] for variant in VARIANTS)):
            first = triplet[0]
            for other in triplet[1:]:
                np.testing.assert_array_equal(first['indices'],other['indices'])
                np.testing.assert_array_equal(first['features'][:,[0,1,*range(3,36)]],
                                              other['features'][:,[0,1,*range(3,36)]])
                np.testing.assert_array_equal(first['labels'],other['labels'])
            rows.append({'name':first['name'],'source_sha256':str(first['source_sha256'].item()),
                         'closed_events':int(len(first['labels'])),
                         'candidates':int(len(first['indices'])),
                         'feature_sha256':{variant:sha256(cache/role/
                              (Path(first['name']).stem+'_'+variant+'.npz')) for variant in VARIANTS}})
        summary['roles'][role] = {'sequences':len(rows),
                                  'closed_events':sum(row['closed_events'] for row in rows),
                                  'candidates':sum(row['candidates'] for row in rows),
                                  'files':rows}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,default=ROOT/'cache')
    parser.add_argument('--output',type=Path,default=ROOT/'reports'/'cache_audit.json')
    args=parser.parse_args()
    result=audit(args.cache,args.output)
    print(json.dumps({role:{key:value for key,value in item.items() if key!='files'}
                      for role,item in result['roles'].items()}))
