"""Initialize a separate run's preservation snapshot; never overwrite an existing one."""
import argparse
from pathlib import Path

import prepare_fred_diagnosis as source


def prepare(output):
    output=Path(output)
    target=output/'preservation_before.json'
    if target.exists():
        record=source.read_json(target)
        print(f'Reusing the original preservation snapshot: {len(record)} files')
        return
    if output.exists() and any(output.iterdir()):
        raise ValueError('A new run must start in an empty directory; existing artifacts are preserved')
    output.mkdir(parents=True,exist_ok=True)
    prior=source.read_json(source.OUT/'preservation_before.json')
    paths={Path(p) for p in prior}
    ann=source.ROOT.parents[1]/'1_H2_model_improvement/implementation'
    for folder in (source.ROOT/'software',ann/'software',source.OUT):
        paths.update(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    paths.update(p for p in (source.ROOT/'results').glob('*') if p.is_file())
    record={str(p):{'sha256':source.sha256(p),'bytes':p.stat().st_size} for p in sorted(paths)}
    source.atomic_json(target,record)
    print(f'Preservation snapshot created: {len(record)} files; no inference or training')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=source.ROOT/'results/runs/fred_statistics_adaptation_v1')
    prepare(p.parse_args().output)
