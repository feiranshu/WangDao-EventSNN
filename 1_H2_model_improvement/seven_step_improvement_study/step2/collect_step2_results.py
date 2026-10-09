#!/usr/bin/env python3
"""Collect a compact, reviewable Step-2 result archive.

Candidate caches and transient logs are intentionally excluded.  The archive
contains decisions, metrics, audit reports, configuration snapshots and the
selected checkpoints required to reproduce the reported conclusion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


INCLUDE_SUFFIXES = {".json", ".csv", ".svg", ".png", ".npz", ".pt", ".md", ".txt", ".log"}
EXCLUDE_PARTS = {"cache", "candidate_cache", "__pycache__"}


def include(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    if any(part in EXCLUDE_PARTS for part in relative.parts):
        return False
    return path.is_file() and path.suffix.lower() in INCLUDE_SUFFIXES


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "step2_decision.json").exists():
        raise SystemExit(f"missing step2_decision.json: {run_dir}")
    output = (args.output or run_dir.with_suffix(".results.zip")).resolve()
    files = sorted(path for path in run_dir.rglob("*") if include(path, run_dir))
    manifest = {
        "source": str(run_dir),
        "files": [
            {"path": path.relative_to(run_dir).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path(run_dir.name) / path.relative_to(run_dir))
        archive.writestr(
            f"{run_dir.name}/RESULT_MANIFEST.json",
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
    print(json.dumps({"archive": str(output), "sha256": sha256(output), "files": len(files)}, indent=2))


if __name__ == "__main__":
    main()
