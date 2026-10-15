"""Report exact, separately recorded concurrent edits without hiding differences."""
from pathlib import Path
from .common import ROOT,SNN_ROOT,read,sha,preserve_check,atomic_json


def verified_concurrent_changes(changed,record,before):
    known=record['files']
    if set(changed)!=set(known):raise ValueError('Unreviewed original-file change')
    allowed=(SNN_ROOT/'software/wangdao/h2pro_adaptive_t_v2').resolve()
    # This independent SNN branch is not imported by the ANN experiment.
    # All consumed source/weight/cache identities are checked separately.
    for name,item in known.items():
        path=Path(name).resolve()
        if not path.is_relative_to(allowed) or before[name]!=item['before']:
            raise ValueError('Concurrent-edit record is outside the reviewed independent branch')
        if not path.is_file():raise ValueError('Concurrent file missing')
        stat=path.stat();now=dict(bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,sha256=sha(path))
        if now!=item['after']:raise ValueError('Concurrent file changed again; review required')
    return known


def preservation_report():
    try:return dict(status='unchanged',**preserve_check())
    except ValueError:
        report=read(ROOT/'results/checks/preservation_after.json')
        record_path=ROOT/'results/checks/concurrent_changes.json'
        if not record_path.exists():raise
        record=read(record_path)
        known=verified_concurrent_changes(report['changed_or_missing'],record,
                                         read(ROOT/'results/checks/preservation_before.json'))
        report.update(status='unchanged_except_documented_concurrent_edit',
                      unchanged_files=report['original_files']-len(known),concurrent_changes=known,
                      concurrent_record_sha256=sha(record_path),experiment_dependency_identities_unchanged=True)
        atomic_json(ROOT/'results/checks/preservation_review.json',report)
        return report
