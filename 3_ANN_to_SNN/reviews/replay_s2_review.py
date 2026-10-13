"""Read-only post-S2 review: signed per-update states and Train-dev score replay.

Prints diagnostics only. Uses the archived revision1 epoch5 checkpoint;
never trains, mutates model sources, or opens held-out data.
"""
import sys,json,hashlib,pathlib,numpy as np,torch
p=pathlib.Path(__file__).absolute().parents[1]/'s2';sys.path.insert(0,str(p))
from core import ResearchSNN,SNNRunner,THRESHOLD,class_logits
from train import read_index,role_records,windows,average_precision
root=p/'runs/GEN2-S2-001-revision1';prov=json.loads((root/'provenance.json').read_text())
print('provenance',json.dumps({k:hashlib.sha256((p/f).read_bytes()).hexdigest()==prov[k] for k,f in [('core_sha256','core.py'),('train_sha256','train.py'),('training_protocol_sha256','PROTOCOL_at_revision1_start.md')]}))
torch.set_num_threads(1);m=ResearchSNN().eval();m.load_state_dict(torch.load(root/'snn_epoch05.pt',map_location='cpu')['model_state_dict'])
idx,manifest=read_index();records=role_records(idx,manifest,'train_dev')
def blank():return [{'n':0,'min':float('inf'),'max':float('-inf'),'post_ge_theta':0,'pre_sg_zero_high':0,'pre_sg_zero_low':0,'spikes':0} for _ in range(3)]
stats={g:blank() for g in ['ordinary','high','target','background']};rows=[];allp=[];ally=[]
with torch.no_grad():
 for rec in records:
  run=SNNRunner(m,rec['t0']);prob=np.empty(len(rec['features']),np.float32);grade='high' if rec['event_rate']>=20000 else 'ordinary'
  for seg in windows(rec):
   f=rec['features'][seg];tar=rec['target'][seg].astype(bool);pack=run.pack(f,rec['x'][seg],rec['y'][seg],rec['t'][seg]);state=tuple(torch.from_numpy(a[pack.cells].copy()) for a in run.state)
   for k in range(len(pack.event_slots)):
    valid=pack.event_slots[k]>=0;safe=np.maximum(pack.event_slots[k],0);x=torch.from_numpy(f[safe].astype(np.float32)/255)
    logit,up,sp=m.step(x,torch.from_numpy(pack.delta_ms[k]),state);mask=torch.from_numpy(valid)[:,None];state=tuple(torch.where(mask,a,b) for a,b in zip(up,state))
    ids=pack.event_slots[k,valid];prob[seg.start+ids]=torch.softmax(class_logits(logit[valid]),1)[:,1].numpy()
    for layer in range(3):
     v=up[layer][valid].numpy();s=sp[layer][valid].numpy();pre=v+THRESHOLD*s
     for group,sel in [(grade,np.ones(len(v),bool)),('target',tar[ids]),('background',~tar[ids])]:
      a=v[sel];b=pre[sel];c=s[sel]
      if not a.size:continue
      st=stats[group][layer];st['n']+=a.size;st['min']=min(st['min'],float(a.min()));st['max']=max(st['max'],float(a.max()))
      st['post_ge_theta']+=int((a>=THRESHOLD).sum());st['pre_sg_zero_high']+=int((b>=THRESHOLD+1).sum());st['pre_sg_zero_low']+=int((b<=THRESHOLD-1).sum());st['spikes']+=int(c.sum())
   for dest,value in zip(run.state,state):dest[pack.cells]=value.numpy()
  y=rec['target'].astype(bool);full=np.zeros(len(rec['labels']),np.float32);full[rec['indices']]=prob
  row={'name':rec['name'],'density':grade,'events':len(full),'candidates':len(prob),'ap':average_precision(rec['labels'],full)}
  for th in [.35,.6,.75]:
   tp=int(((prob>=th)&y).sum());fp=int(((prob>=th)&~y).sum());fn=int(rec['labels'].sum())-tp;row[str(th)]={'tp':tp,'fp':fp,'fn':fn,'iou':tp/(tp+fp+fn)}
  rows.append(row);allp.append(full);ally.append(rec['labels'])
for vals in stats.values():
 for st in vals:
  for k in ['post_ge_theta','pre_sg_zero_high','pre_sg_zero_low','spikes']:st[k+'_fraction']=st.pop(k)/st['n']
print('signed_state_diagnostics',json.dumps(stats));print('sequence_metrics',json.dumps(rows))
ap=average_precision(np.concatenate(ally),np.concatenate(allp));expected=json.loads((root/'snn_history.json').read_text())[-1]['train_dev']['average_precision'];print('AP_replay',ap,'recorded',expected,'delta',ap-expected)
