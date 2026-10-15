"""Immutable protocol, deterministic Main split, and restricted m0 parameters."""
from pathlib import Path
import hashlib
import numpy as np

from ..h2pro_channel_calibration_v1.common import (
    IMPLEMENTATION, DEFAULT_CACHE, read_json, sha256, atomic_json, timestamp,
    load_role, parameter_id, tree_hashes, verify_baseline, save_arrays,
    lock_directory, output_path, source_hashes as old_sources,
)
from ..h2pro_channel_calibration_v1.core import Channels, cost
from ..h2pro_channel_calibration_v1.workflow import historical_parameters

PACKAGE=Path(__file__).resolve().parent
DEFAULT_RUN=IMPLEMENTATION/'results/runs/h2pro_m0_calibration_v2'
OLD_RUN=IMPLEMENTATION/'results/runs/h2pro_channel_calibration_v1'
PROTOCOL={
    'version':'h2pro_m0_calibration_v2','direction':2,'stage':'r1','steps':16,
    'anchor':'original frozen H2-pro r1/T16/task_rescore; theta=255, m0=191',
    'unchanged':['ANN Proposal','original 36-64-32-4 structure','original int8 weights',
                 'original FP32 scales','all theta','T16','original features','decision threshold 0.35'],
    'training_enabled':False,'optimizer_steps':0,'backward_calls':0,
    'split':{'seed':42,'validation_fraction':.25,
             'rule':'within ordinary/HD strata, sort SHA256(seed:sequence), take ceil(n/4) for validation'},
    'sample':{'seed':42,'positive':'all fit-sequence candidate positives',
              'background_per_fit_sequence':8192,'weight':'inverse class-stratum inclusion probability'},
    'groups':'fit-only population-weighted mean ANN hidden1 activation, tie channel index, eight groups of eight',
    'max_changed_groups':2,'m0_offsets':list(range(-16,0))+list(range(1,17)),
    'pair_search':'for each group retain best two negative and two positive offsets by the common ranking; enumerate all pairs of distinct groups',
    'shortlist_per_group_count':8,
    'objective':'overall event IoU, never average ordinary and HD IoUs',
    'eligibility':'IoU, overall recall, HD recall >= anchor; AC <= anchor; numerical comparison tolerance only',
    'rank':'eligible first, highest overall IoU, lower AC, fewer changed groups, smaller total m0 change, parameter id',
    'main_selection':'eligibility must hold on complete fit, complete internal validation and complete Main; rank by internal-validation overall IoU with the same tie order',
    'dev_points':['anchor','best changed one-group diagnostic','best changed two-group diagnostic','primary eligible winner including anchor'],
    'diagnostics':'changed-group winners may be ineligible; they are reported as tradeoffs, not promoted',
    'freeze':'seal fit shortlist before scoring internal validation; seal all Main choices before loading Dev',
    'backend':'unchanged GPU exact-count reference, fixed CPU explicit binary spike-add cross-checks',
    'reserved_splits_read':False,
}


def anchor():
    e=historical_parameters('r1',16)['task_rescore']
    if e != (Channels.shared(64,255,191),Channels.shared(32,255,127)):
        raise ValueError('Unexpected historical best identity')
    return e


def validate_groups(groups):
    if (len(groups)!=8 or any(len(g)!=8 for g in groups)
            or sorted(x for g in groups for x in g)!=list(range(64))):
        raise ValueError('Require a nonoverlapping partition into eight groups')


def encoders(changes,groups):
    validate_groups(groups)
    if len(changes)>PROTOCOL['max_changed_groups']:
        raise ValueError('At most two groups may change')
    initial=anchor();m=list(initial[0].membrane)
    for key,value in changes.items():
        i=int(key)
        if str(i)!=str(key) or not 0<=i<8 or type(value) is not int or value-191 not in PROTOCOL['m0_offsets']:
            raise ValueError('Only declared nonzero local m0 offsets are allowed')
        for channel in groups[i]:m[channel]=value
    e=(Channels(initial[0].theta,tuple(m)),initial[1])
    e[0].validate(64)
    return e


def split_sequences(rows):
    if any(r['role']!='train_main' for r in rows):
        raise ValueError('Only Main sequences may be split')
    names=[r['sequence'] for r in rows]
    if len(names)!=len(set(names)):raise ValueError('Duplicate sequences')
    validation=[]
    for hd in (False,True):
        group=[r['sequence'] for r in rows if r['high_density']==hd]
        if len(group)<2:raise ValueError('Need both fit and validation sequences in each density stratum')
        ordered=sorted(group,key=lambda x:(hashlib.sha256(f"42:{x}".encode()).hexdigest(),x))
        count=min(len(group)-1,(len(group)+3)//4)
        validation.extend(ordered[:count])
    val=set(validation)
    return {'fit':[n for n in names if n not in val],'validation':[n for n in names if n in val],
            'role':'train_main','protocol':PROTOCOL['split']}


def sample_fit(seqs):
    xs=[];ys=[];ws=[];hs=[];ts=[];receipts=[]
    for s in seqs:
        r=s['row']
        if r['role']!='train_main':raise ValueError('Fit samples must be Main')
        y=s['labels'][s['candidate']];positive=np.flatnonzero(y==1);negative=np.flatnonzero(y==0)
        k=min(PROTOCOL['sample']['background_per_fit_sequence'],len(negative))
        seed=int.from_bytes(hashlib.sha256(f"42:{r['sequence']}:background".encode()).digest()[:8],'little')
        bg=np.sort(np.random.default_rng(seed).choice(negative,k,replace=False)) if k else np.empty(0,np.int64)
        ids=np.sort(np.r_[positive,bg]);labels=y[ids]
        weights=np.where(labels==1,1.,len(negative)/k if k else 0.)
        xs.append(s['features'][ids]);ys.append(labels);ws.append(weights);ts.append(s['teacher'][ids])
        hs.append(np.full(len(ids),r['high_density'],bool))
        receipts.append({'sequence':r['sequence'],'positive_population':len(positive),'positive_sampled':len(positive),
                         'background_population':len(negative),'background_sampled':k,
                         'indices_sha256':hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest()})
    population={g:{'events':sum(s['row']['events'] for s in seqs if g=='overall' or s['row']['high_density']==(g=='high_density')),
                   'targets':sum(s['row']['target_events'] for s in seqs if g=='overall' or s['row']['high_density']==(g=='high_density'))}
                for g in ('overall','ordinary','high_density')}
    return {'features':np.concatenate(xs),'labels':np.concatenate(ys),'weights':np.concatenate(ws),
            'hd':np.concatenate(hs),'teacher':np.concatenate(ts),'receipts':receipts,'population':population}


def constraints(record,reference):
    checks=[('overall_iou',record['overall']['iou'],reference['overall']['iou'],True),
            ('overall_recall',record['overall']['recall'],reference['overall']['recall'],True),
            ('hd_recall',record['high_density']['recall'],reference['high_density']['recall'],True),
            ('AC',record['cost']['equivalent_ac'],reference['cost']['equivalent_ac'],False)]
    return [name for name,a,b,higher in checks
            if not np.isclose(a,b,rtol=1e-12,atol=1e-12) and (a<b if higher else a>b)]


def rank(record,reference,changes,eligible=None):
    good=not constraints(record,reference) if eligible is None else eligible
    return (not good,-record['overall']['iou'],record['cost']['equivalent_ac'],
            len(changes),sum(abs(v-191) for v in changes.values()),record['parameter_id'])


def identity(cache,engine):
    paths=list(PACKAGE.glob('*.py'))+[IMPLEMENTATION/'software/scripts/h2pro_calibrate_m0.py',
                                    IMPLEMENTATION/'calibrate_h2pro_m0.ps1']
    sources=old_sources()
    sources.update({p.relative_to(IMPLEMENTATION).as_posix():hashlib.sha256(p.read_text('utf-8').encode()).hexdigest()
                    for p in paths})
    return {'protocol':PROTOCOL,'engine':engine,'baseline':verify_baseline(),'sources':sources,
            'cache_manifest_sha256':sha256(Path(cache)/'manifest.json'),
            'prior_main_calibration_sha256':sha256(IMPLEMENTATION/'results/runs/h2pro_task_calibration_v1_fit/calibration.json'),
            'anchor_encoders':[x.record() for x in anchor()]}


def preserve(directory):
    marker=directory/'preserved_inputs.json'
    if marker.exists():saved=read_json(marker)
    else:
        saved=read_json(OLD_RUN/'preserved_inputs.json')
        saved.update(tree_hashes(OLD_RUN))
        for name in ('h2pro_channel_calibration_v1_implementation_20261015.json','h2pro_channel_calibration_v1_comparison_20261015.png'):
            p=IMPLEMENTATION/'results'/name;saved[p.relative_to(IMPLEMENTATION).as_posix()]=sha256(p)
        atomic_json(marker,saved)
    for name,digest in saved.items():
        if sha256(IMPLEMENTATION/name)!=digest:raise ValueError('Old experiment changed: '+name)
    return len(saved)
