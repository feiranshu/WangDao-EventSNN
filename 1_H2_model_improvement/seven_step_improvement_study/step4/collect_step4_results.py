#!/usr/bin/env python3
"""Collect reviewable Step4 results while excluding regenerated caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


INCLUDE_SUFFIXES = {".json", ".csv", ".svg", ".png", ".npz", ".pt", ".md", ".txt", ".log"}
EXCLUDE_PARTS = {"cache", "preflight_cache", "__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(path: Path, root: Path) -> bool:
    relative = path.relative_to(root)
    return path.is_file() and path.suffix.lower() in INCLUDE_SUFFIXES and not any(part in EXCLUDE_PARTS for part in relative.parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    if not (root / "step4_decision.json").is_file():
        raise SystemExit(f"missing step4_decision.json: {root}")
    output = (args.output or root.with_suffix(".results.zip")).resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    files = sorted(path for path in root.rglob("*") if included(path, root))
    manifest = {
        "source": str(root),
        "validation_used": False,
        "test_used": False,
        "excluded_regenerable_caches": True,
        "files": [{"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size} for path in files],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path(root.name) / path.relative_to(root))
        archive.writestr(f"{root.name}/RESULT_MANIFEST.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"archive": str(output), "sha256": sha256(output), "files": len(files)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
