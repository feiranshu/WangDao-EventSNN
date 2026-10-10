"""H2-pro development-only preparation, fair training, and Train-dev scoring.

No command in this file opens Validation, Test or hardware-calibration data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).absolute().parent
SOFTWARE = ROOT.parents[1] / 'implementation' / 'software'
sys.path[:0] = [str(ROOT), str(SOFTWARE)]

import numpy as np
from src.stream_state import CausalPipeline, VARIANTS, EVENT_DTYPE
from wangdao.h2.numpy_model import _refiner_probability, quantized_mlp
from wangdao.metrics import event_metrics, localization_metrics

PROTOCOL = json.loads((ROOT/'protocol.json').read_text(encoding='utf-8'))
PROPOSAL = SOFTWARE/'weights'/'H2_D1_seed42.npz'
INTERMEDIATE = ROOT.parent/'step7'/'intermediate_H2-plus'/'weights'
SOURCE_FLOAT = INTERMEDIATE/'A4-2_seed42_epoch008.pt'
SOURCE_INT8 = INTERMEDIATE/'A4-2_seed42_epoch008_int8.npz'
INDEX = ROOT.parents[1]/'implementation'/'data'/'index.json'
DATA = ROOT.parents[1]/'implementation'/'data'/'EV-UAV-dataset'
ROLE_DIR = {'train_main':'train', 'train_dev':'train'}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def verify_inputs() -> dict:
    expected = {
        PROPOSAL:'21019377617e429a599497a0a6b0f541d2c05bf7bc8fee83005deae72bb578b2',
        SOURCE_FLOAT:'a9c71311dbc01c6142ed826697b466546f3404aeb76798e9ede1461a35d3119f',
        SOURCE_INT8:'d7d42a3876c37497bdcd5206e6f35a5f949b0230dde0675fd128f989b70d05c4',
    }
    actual = {path.name:sha256(path) for path in expected}
    for path, digest in expected.items():
        if actual[path.name] != digest:
            raise ValueError(f'frozen source hash mismatch: {path}')
    return actual


def _role_files(role: str, limit: int = 0):
    if role not in ROLE_DIR:
        raise ValueError('only train_main and train_dev are permitted')
    index = json.loads(INDEX.read_text(encoding='utf-8'))
    lookup = {Path(row['path']).name:row['sha256'] for row in index['files']}
    names = index['roles'][role]
    for name in names[:limit or None]:
        path = DATA/ROLE_DIR[role]/name
        if sha256(path) != lookup[name]:
            raise ValueError(f'data SHA mismatch: {name}')
        yield name, path


def prepare(role: str, output: Path, limit: int = 0, resume: bool = False) -> None:
    hashes = verify_inputs()
    directory = output/role
    directory.mkdir(parents=True, exist_ok=True)
    protocol_sha = sha256(ROOT/'protocol.json')
    for name, path in _role_files(role, limit):
        targets = [directory/(Path(name).stem+'_'+variant+'.npz') for variant in VARIANTS]
        if resume and all(target.exists() for target in targets):
            for target in targets:
                with np.load(target,allow_pickle=False) as archive:
                    if (str(archive['protocol_sha256'].item()) != protocol_sha or
                            str(archive['source_sha256'].item()) != sha256(path)):
                        raise ValueError(f'cannot resume mismatched cache: {target}')
            print(json.dumps({'role':role,'sequence':name,'status':'verified cached'}),flush=True)
            continue
        if any(target.exists() for target in targets):
            raise FileExistsError(f'partial or pre-existing cache for {name}; preserve and inspect it')
        with np.load(path, allow_pickle=False) as archive:
            events = archive['ev'].copy()
        t0 = float(events['t'][0])
        duration = (float(events['t'][-1])-t0)
        closed_count = int(np.searchsorted(events['t'], t0+int(duration//100)*100,
                                            side='left'))
        count = int(duration//100)
        if count == 0:
            raise ValueError(f'sequence has no complete 100 ms window: {name}')
        complete = events[:closed_count]
        common = None
        for variant in VARIANTS:
            target = directory/(Path(name).stem+'_'+variant+'.npz')
            pipe = CausalPipeline(PROPOSAL, SOURCE_INT8, variant)
            pipe.reset(t0)
            pipe.push(complete, t0+count*100)
            windows = pipe.flush_completed_windows()
            features = np.concatenate([w.features for w in windows], axis=0)
            indices = np.concatenate([w.event_indices[w.candidate_mask] for w in windows])
            if common is None:
                common = (indices, features[:, [*range(2), *range(3,36)]])
            else:
                np.testing.assert_array_equal(indices, common[0])
                np.testing.assert_array_equal(features[:, [*range(2), *range(3,36)]], common[1])
            np.savez_compressed(target, features=features, indices=indices,
                                labels=complete['label'].astype(np.uint8),
                                identities=complete['name'], x=complete['x'], y=complete['y'],
                                t=complete['t'], t0=np.float64(t0),
                                source_sha256=np.asarray(sha256(path)),
                                protocol_sha256=np.asarray(protocol_sha))
        print(json.dumps({'role':role,'sequence':name,'closed_events':len(complete),
                          'candidates':len(common[0]),'source_hashes':hashes}), flush=True)


def _records(cache: Path, role: str, variant: str):
    if role not in ROLE_DIR or variant not in VARIANTS:
        raise ValueError('only registered development roles and variants are allowed')
    index = json.loads(INDEX.read_text(encoding='utf-8'))
    names = index['roles'][role]
    expected_sources = {Path(item['path']).name:item['sha256'] for item in index['files']}
    records = []
    expected_protocol = sha256(ROOT/'protocol.json')
    for name in names:
        path = cache/role/(Path(name).stem+'_'+variant+'.npz')
        with np.load(path, allow_pickle=False) as archive:
            row = {key:archive[key].copy() for key in archive.files}
        if str(row['protocol_sha256'].item()) != expected_protocol:
            raise ValueError(f'cache protocol mismatch: {path}')
        if str(row['source_sha256'].item()) != expected_sources[name]:
            raise ValueError(f'cache source hash mismatch: {path}')
        if row['features'].shape != (len(row['indices']),36) or row['features'].dtype != np.uint8:
            raise ValueError(f'cache feature contract mismatch: {path}')
        row['name'] = name
        source_path = DATA/ROLE_DIR[role]/name
        if sha256(source_path) != expected_sources[name]:
            raise ValueError(f'underlying data changed after cache creation: {source_path}')
        with np.load(source_path,allow_pickle=False) as source:
            full_events = source['ev']
            row['full_event_rate'] = len(full_events)/max(
                (float(full_events['t'][-1])-float(full_events['t'][0]))/1000,1e-9)
        records.append(row)
    return records


def _model_from_checkpoint(source: Path):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class Refiner(nn.Module):
        def __init__(self):
            super().__init__()
            self.old = nn.Linear(35,64,bias=False)
            self.new = nn.Linear(1,64,bias=False)
            self.hidden = nn.Linear(64,32,bias=False)
            self.output = nn.Linear(32,4,bias=False)

        def forward(self, features):
            value = features.float()/255.0
            old = torch.cat((value[:,:2], value[:,3:]), dim=1)
            value = F.hardtanh(self.old(old)+self.new(value[:,2:3]),0,1)
            value = F.hardtanh(self.hidden(value),0,1)
            return self.output(value)

        def combined(self):
            return (torch.cat((self.old.weight[:,:2], self.new.weight,
                               self.old.weight[:,2:]), dim=1),
                    self.hidden.weight, self.output.weight)

    checkpoint = torch.load(source, map_location='cpu')
    if tuple(checkpoint['dimensions']) != (36,64,32,4) or checkpoint['old_width'] != 32 or checkpoint['new_width'] != 4:
        raise ValueError('A4-2 float checkpoint shape or partition mismatch')
    state = checkpoint['state_dict']
    first = torch.cat((state['old_input.weight'],state['new_input.weight']),dim=1)
    model = Refiner()
    with torch.no_grad():
        model.old.weight.copy_(torch.cat((first[:,:2],first[:,3:]),dim=1))
        model.new.weight.zero_()
        model.hidden.weight.copy_(state['hidden.weight'])
        model.output.weight.copy_(state['output.weight'])
    return model


def _quantized(model):
    from wangdao.h2.export import quantize_weight
    return tuple((np.ascontiguousarray(q),np.ascontiguousarray(s))
                 for q,s in (quantize_weight(weight) for weight in model.combined()))


def load_trained_float(path: Path, device: str = 'cpu'):
    """ANN→SNN teacher loader for the frozen 36-column float checkpoint."""
    import torch
    checkpoint = torch.load(path,map_location='cpu')
    if checkpoint.get('protocol_sha256') != sha256(ROOT/'protocol.json'):
        raise ValueError('float teacher feature protocol mismatch')
    model = _model_from_checkpoint(SOURCE_FLOAT)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model.to(device).eval()


def _predict(model, features: np.ndarray, device: str, numeric: str) -> np.ndarray:
    if numeric == 'int8':
        return _refiner_probability(quantized_mlp(features, _quantized(model)),2)
    import torch
    model.eval()
    parts = []
    with torch.no_grad():
        for start in range(0,len(features),4096):
            block = torch.from_numpy(features[start:start+4096].copy()).to(device)
            raw = model(block).reshape(-1,2,2).amax(2)
            parts.append(torch.softmax(raw,dim=1)[:,1].cpu().numpy())
    return np.concatenate(parts) if parts else np.empty(0,dtype=np.float32)


def _ap(labels, probabilities):
    order = np.argsort(-probabilities,kind='stable')
    p, y = probabilities[order], labels[order].astype(bool)
    if not len(p) or not np.any(y):
        return None
    ends = np.r_[np.flatnonzero(p[1:] != p[:-1]),len(p)-1]
    tp = np.cumsum(y,dtype=np.int64)[ends]
    recall = tp/int(y.sum())
    return float(np.sum(np.diff(np.r_[0.,recall])*tp/(ends+1)))


def object_window_continuity(row, probabilities, threshold: float) -> dict:
    """Evaluation-only ID metric: eligible object-windows contain >=3 target events.

    A window is detected when any event of that object crosses the threshold.
    Consecutive eligible windows with the same ID form a track segment; absent
    windows terminate a segment, rather than counting as detection failures.
    """
    labels = row['labels'].astype(bool)
    ids = row['identities']
    windows = np.floor((row['t'].astype(np.float64)-float(row['t0']))/100).astype(np.int64)
    found = probabilities >= threshold
    eligible = detected = breaks = longest = 0
    reacquisition = []
    for identity in np.unique(ids[labels]):
        member = labels & (ids == identity)
        object_windows = np.unique(windows[member])
        last_window = None
        last_hit = None
        run = miss = 0
        for window in object_windows:
            selected = member & (windows == window)
            if int(selected.sum()) < 3:
                continue
            hit = bool(np.any(found[selected]))
            eligible += 1
            detected += int(hit)
            if last_window is None or window != last_window+1:
                run = miss = 0
                last_hit = None
            if last_hit is True and not hit:
                breaks += 1
            if hit:
                run += 1
                longest = max(longest,run)
                if miss and last_hit is False:
                    reacquisition.append(miss)
                miss = 0
            else:
                run = 0
                miss += 1
            last_window, last_hit = int(window), hit
    return {'eligible_object_windows':eligible,'detected_object_windows':detected,
            'coverage':detected/eligible if eligible else None,
            'breaks':breaks,'longest_detected_run':longest,
            'reacquisition_gaps_windows':reacquisition}


def score(records, model, device, numeric):
    thresholds = PROTOCOL['threshold_grid']
    all_labels, all_prob = [], []
    curves = {str(t):[] for t in thresholds}
    per_sequence = []
    for row in records:
        features = row['features']
        cp = _predict(model, features, device, numeric)
        prob = np.zeros(len(row['labels']),dtype=np.float32)
        prob[row['indices']] = cp
        all_labels.append(row['labels'])
        all_prob.append(prob)
        rate = row['full_event_rate']
        item = {'name':row['name'],'events':len(prob),'event_rate':rate,
                'high_density':rate>=20000}
        for t in thresholds:
            event = event_metrics(row['labels'],prob,t)
            loc = localization_metrics(row['t'],row['x'],row['y'],row['labels'],
                                       row['identities'],prob,t)
            curves[str(t)].append((event,loc))
            item[str(t)] = {'event':event,'localization':loc,
                            'continuity':object_window_continuity(row,prob,t)}
        per_sequence.append(item)
    labels = np.concatenate(all_labels)
    prob = np.concatenate(all_prob)
    result = {'numeric':numeric,'ap':_ap(labels,prob),'sequences':per_sequence,'curve':{}}
    for t in thresholds:
        key = str(t)
        event = event_metrics(labels,prob,t)
        locs = [item[1] for item in curves[key]]
        targets = sum(v['targets'] for v in locs)
        false = sum(v['false_components'] for v in locs)
        frames = sum(v['frame_denominator'] for v in locs)
        loc = {'pd':sum(v['correct_targets'] for v in locs)/targets if targets else 1.0,
               'fa':false/(frames*346*260) if frames else 0.0,
               'targets':targets,'false_components':false,'frame_denominator':frames}
        high = [row for row in per_sequence if row['high_density']]
        high_tp = sum(row[key]['event']['true_positive'] for row in high)
        high_fn = sum(row[key]['event']['false_negative'] for row in high)
        density_views = {}
        for density_name,group in (('high',high),('ordinary',[
            row for row in per_sequence if not row['high_density']])):
            tp = sum(row[key]['event']['true_positive'] for row in group)
            fp = sum(row[key]['event']['false_positive'] for row in group)
            fn = sum(row[key]['event']['false_negative'] for row in group)
            target_windows = sum(row[key]['localization']['targets'] for row in group)
            frames_group = sum(row[key]['localization']['frame_denominator'] for row in group)
            density_views[density_name] = {
                'sequences':len(group),'iou':tp/(tp+fp+fn) if tp+fp+fn else 1.0,
                'precision':tp/(tp+fp) if tp+fp else 1.0,
                'recall':tp/(tp+fn) if tp+fn else 1.0,
                'pd':sum(row[key]['localization']['correct_targets'] for row in group)/target_windows
                     if target_windows else 1.0,
                'fa':sum(row[key]['localization']['false_components'] for row in group)/
                     (frames_group*346*260) if frames_group else 0.0}
        eligible = sum(row[key]['continuity']['eligible_object_windows'] for row in per_sequence)
        detected = sum(row[key]['continuity']['detected_object_windows'] for row in per_sequence)
        result['curve'][key] = {'event':event,'localization':loc,
                                'density_strata':density_views,
                                'high_density_recall':high_tp/(high_tp+high_fn) if high_tp+high_fn else None,
                                'object_window_coverage':detected/eligible if eligible else None,
                                'object_window_breaks':sum(row[key]['continuity']['breaks'] for row in per_sequence)}
    return result


def train(cache: Path, output: Path, device: str) -> None:
    import torch
    import torch.nn.functional as F
    verify_inputs()
    if output.exists():
        raise FileExistsError(f'append-only run directory exists: {output}')
    output.mkdir(parents=True)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    initial = _model_from_checkpoint(SOURCE_FLOAT)
    initial_weights = [item.detach().clone() for item in initial.combined()]
    dev_by_variant = {variant:_records(cache,'train_dev',variant) for variant in VARIANTS}
    reference = dev_by_variant['C0-zero']
    audit = {'source_float_sha256':sha256(SOURCE_FLOAT),
             'source_int8_sha256':sha256(SOURCE_INT8),
             'protocol_sha256':sha256(ROOT/'protocol.json'),
             'other_35_equal':True,'max_float_probability_error':0.0,
             'max_int8_probability_error':0.0}
    cols = [*range(2),*range(3,36)]
    for variant in VARIANTS[1:]:
        for base,other in zip(reference,dev_by_variant[variant]):
            np.testing.assert_array_equal(base['indices'],other['indices'])
            np.testing.assert_array_equal(base['features'][:,cols],other['features'][:,cols])
            for numeric in ('float','int8'):
                a = _predict(initial,base['features'][:256],'cpu',numeric)
                b = _predict(initial,other['features'][:256],'cpu',numeric)
                delta = float(np.max(np.abs(a-b))) if len(a) else 0.0
                audit['max_'+numeric+'_probability_error'] = max(
                    audit['max_'+numeric+'_probability_error'],delta)
                if delta != 0.0:
                    raise AssertionError('zero-column initial outputs differ')
    (output/'initial_equivalence.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    all_results = {}
    for variant in VARIANTS:
        records = _records(cache,'train_main',variant)
        dev = dev_by_variant[variant]
        features = np.concatenate([row['features'] for row in records])
        labels = np.concatenate([row['labels'][row['indices']] for row in records])
        model = _model_from_checkpoint(SOURCE_FLOAT).to(device)
        for a,b in zip(initial_weights,model.combined()):
            if not torch.equal(a,b.cpu()):
                raise AssertionError('candidate initializations differ')
        for module in (model.old,model.hidden,model.output):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        optimizer = torch.optim.AdamW([{'params':model.new.parameters(),'lr':0.001}],
                                      weight_decay=0.0001)
        variant_dir = output/variant
        variant_dir.mkdir()
        logs = []
        for epoch in range(11):
            started = time.perf_counter()
            if epoch == 3:
                for module in (model.old,model.hidden,model.output):
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
                optimizer.param_groups[0]['lr'] = 0.00001
                optimizer.add_param_group({'params':list(model.old.parameters())+
                    list(model.hidden.parameters())+list(model.output.parameters()),'lr':0.00001})
            if epoch:
                model.train()
                order = np.random.default_rng(42*1000003+epoch).permutation(len(features))
                losses = []
                for start in range(0,len(order),4096):
                    idx = order[start:start+4096]
                    x = torch.from_numpy(features[idx].copy()).to(device)
                    y = torch.from_numpy(labels[idx].astype(np.float32)).to(device)
                    optimizer.zero_grad()
                    raw = model(x).reshape(-1,2,2).amax(2)
                    p = torch.softmax(raw,dim=1)[:,1]
                    loss = F.binary_cross_entropy(p,y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
                    optimizer.step()
                    losses.append(float(loss.item()))
            else:
                losses = []
            row = {'epoch':epoch,'variant':variant,'train_loss':float(np.mean(losses)) if losses else None,
                   'seconds':time.perf_counter()-started}
            if epoch in PROTOCOL['training']['checkpoint_epochs']:
                checkpoint = variant_dir/f'epoch{epoch:03}.pt'
                torch.save({'model_id':'H2-pro-'+variant,'epoch':epoch,'state_dict':model.state_dict(),
                            'protocol_sha256':sha256(ROOT/'protocol.json'),
                            'source_sha256':sha256(SOURCE_FLOAT)},checkpoint)
                arrays = {'variant':np.asarray(variant),
                          'dimensions':np.asarray([36,64,32,4]),
                          'protocol_sha256':np.asarray(sha256(ROOT/'protocol.json'))}
                for i,(w,s) in enumerate(_quantized(model)):
                    arrays[f'layer{i}_weight'] = w
                    arrays[f'layer{i}_scale'] = s
                np.savez_compressed(variant_dir/f'epoch{epoch:03}_int8.npz',**arrays)
                row['float'] = score(dev,model,device,'float')
                row['int8'] = score(dev,model,device,'int8')
                differences = []
                for sequence in dev:
                    x = sequence['features']
                    differences.append(np.abs(_predict(model,x,device,'float')-
                                              _predict(model,x,device,'int8')))
                differences = np.concatenate(differences)
                row['quantization_error'] = {
                    'mean_abs_probability':float(np.mean(differences)) if len(differences) else 0.0,
                    'max_abs_probability':float(np.max(differences)) if len(differences) else 0.0,
                    'p95_abs_probability':float(np.quantile(differences,0.95)) if len(differences) else 0.0}
                row['checkpoint_sha256'] = sha256(checkpoint)
                row['int8_sha256'] = sha256(variant_dir/f'epoch{epoch:03}_int8.npz')
            logs.append(row)
            (variant_dir/'progress.json').write_text(json.dumps(logs,indent=2),encoding='utf-8')
            print(json.dumps({'variant':variant,'epoch':epoch,'loss':row['train_loss'],
                              'seconds':row['seconds']}),flush=True)
        all_results[variant] = logs
    (output/'training_complete.json').write_text(json.dumps({
        'protocol_sha256':sha256(ROOT/'protocol.json'), 'variants':list(all_results),
        'status':'Train-dev results available; selection and reserved-data acceptance pending'
    },indent=2),encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['preflight','prepare','train'])
    parser.add_argument('--role',choices=['train_main','train_dev'])
    parser.add_argument('--cache',type=Path,default=ROOT/'cache')
    parser.add_argument('--output',type=Path,default=ROOT/'results'/'run1')
    parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    if args.command == 'preflight':
        print(json.dumps({'weights':verify_inputs(),'index_sha256':sha256(INDEX),
                          'protocol_sha256':sha256(ROOT/'protocol.json')},indent=2))
    elif args.command == 'prepare':
        if not args.role:
            parser.error('--role is required for prepare')
        prepare(args.role,args.cache,args.limit,args.resume)
    else:
        train(args.cache,args.output,args.device)


if __name__ == '__main__':
    main()
