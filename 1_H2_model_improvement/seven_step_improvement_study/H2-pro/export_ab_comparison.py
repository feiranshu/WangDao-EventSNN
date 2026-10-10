"""Render A01-A22 and B01-B04 comparison CSV from immutable post-freeze JSON."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT=Path(__file__).absolute().parent
ROLES=('validation','test')
MODELS=('H2','H2-plus','H2-pro')


def metrics(a,b):
    event=a['event']
    track=a['target_track']
    averaged=track['averaged']
    return {
        'A01':a['ap'], 'A02':event['iou'], 'A03':event['precision'],
        'A04':event['recall'], 'A05':event['f1'],
        'A06':a['localization']['pd'], 'A07':a['localization']['fa'],
        'A08':a['macro_sequence_recall'],
        'A09':a['density_strata']['high']['recall'],
        'A10':a['density_strata']['ordinary']['recall'],
        'A11':event['true_positive'], 'A12':event['false_negative'],
        'A13':event['true_positive']+event['false_negative'],
        'A14':averaged['HOTA'], 'A15':averaged['DetA'],
        'A16':averaged['DetRe'], 'A17':averaged['DetPr'],
        'A18':averaged['AssA'], 'A19':averaged['AssRe'],
        'A20':averaged['AssPr'], 'A21':averaged['LocA'],
        'A22':track['longest_continuous_miss_p95_ms'],
        'B01':b['B01_parameters'], 'B02':b['B02_tensor_payload_bytes'],
        'B03':b['B03_dense_MAC_per_sequence'],
        'B04a':b['B04a_effective_MAC_per_sequence'],
        'B04b':b['B04b_effective_AC_per_sequence'],
        'B04c':b['B04c_effective_operations_per_sequence'],
    }


def export(output):
    if output.exists():
        raise FileExistsError(output)
    data={role:json.loads((ROOT/'reports'/('ab_same_prefix_'+role+'_v2.json')).read_text(encoding='utf-8'))
          for role in ROLES}
    names={
        'A01':'Event AP','A02':'Event IoU','A03':'Precision','A04':'Target-event Recall / ACC',
        'A05':'F1','A06':'Pd','A07':'Fa','A08':'Macro sequence Recall',
        'A09':'High-density Recall','A10':'Ordinary-density Recall',
        'A11':'Detected target events','A12':'Missed target events','A13':'Target events',
        'A14':'HOTA','A15':'DetA','A16':'DetRe','A17':'DetPr','A18':'AssA',
        'A19':'AssRe','A20':'AssPr','A21':'LocA','A22':'Longest continuous miss P95',
        'B01':'Parameter count','B02':'Static tensor payload','B03':'Dense MAC per sequence',
        'B04a':'Effective MAC per sequence','B04b':'Effective AC per sequence',
        'B04c':'Effective operations per sequence'}
    unit={key:('rate' if key in ('A01','A02','A03','A04','A05','A06','A07','A08','A09','A10',
                                'A14','A15','A16','A17','A18','A19','A20','A21') else
               'ms' if key=='A22' else 'byte' if key=='B02' else
               'MAC/sequence' if key in ('B03','B04a') else
               'AC/sequence' if key=='B04b' else
               'op/sequence' if key=='B04c' else 'count') for key in names}
    fields=['metric_id','metric_name','unit']+[role+'_'+model for role in ROLES for model in MODELS]
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for key,name in names.items():
            row={'metric_id':key,'metric_name':name,'unit':unit[key]}
            for role in ROLES:
                for model in MODELS:
                    row[role+'_'+model]=metrics(data[role]['A'][model],data[role]['B'][model])[key]
            writer.writerow(row)
    return len(names)


if __name__=='__main__':
    output=ROOT/'reports'/'AB_ALL_METRICS_SAME_PREFIX.csv'
    print({'output':str(output),'metrics':export(output)})
