"""Audit the original acceptance lock after research and A4-2 weights moved.

The frozen lock itself is unchanged. Former absolute paths are rebased to the
study and intermediate-model locations; every recorded SHA-256 must still match.
"""
from __future__ import annotations

import json
from pathlib import Path

from run_stage1 import ROOT, SOURCE_INT8, sha256


LOCK = ROOT / 'frozen' / 'H2-pro-Causal-ANN-v1' / 'acceptance_lock.json'


def verify_migrated_lock() -> dict:
    payload = json.loads(LOCK.read_text(encoding='utf-8'))
    old_research_root = ROOT.parents[1] / 'H2-pro'
    for name, digest in payload['digests'].items():
        original = Path(name)
        if original.name == 'A4-2_seed42_epoch008_int8.npz':
            path = SOURCE_INT8
        elif str(original).lower().startswith(str(old_research_root).lower() + '\\'):
            path = ROOT / original.relative_to(old_research_root)
        else:
            path = original
        if sha256(path) != digest:
            raise ValueError(f'locked file changed or missing: {name} -> {path}')
    return payload


if __name__ == '__main__':
    payload = verify_migrated_lock()
    print(f"verified {len(payload['digests'])} locked file hashes after relocation")
