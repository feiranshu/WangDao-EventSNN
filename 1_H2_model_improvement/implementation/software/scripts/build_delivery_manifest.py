#!/usr/bin/env python3
"""Build the deterministic SHA-256 manifest for the final implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).absolute().parents[2]
DEFAULT_OUTPUT = IMPLEMENTATION_ROOT / "DELIVERY_MANIFEST.json"
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", "EV-UAV-dataset"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_files(root: Path, output: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    output_resolved = output.absolute()
    paths: list[Path] = []
    for directory, child_directories, names in os.walk(root):
        child_directories[:] = sorted(name for name in child_directories if name not in EXCLUDED_PARTS)
        base = Path(directory)
        for name in names:
            path = base / name
            if path.suffix not in EXCLUDED_SUFFIXES and path.absolute() != output_resolved:
                paths.append(path)
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root)
        files.append({"path": relative.as_posix(), "bytes": path.stat().st_size, "sha256": sha256(path)})
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.absolute()
    files = collect_files(IMPLEMENTATION_ROOT, output)
    payload = {
        "schema_version": 1,
        "artifact": "H2 ANN final H2-pro implementation",
        "model_id": "H2-pro-Causal-ANN-v1",
        "seed": 42,
        "epoch": 10,
        "threshold": 0.35,
        "root": "implementation",
        "excluded": ["data/EV-UAV-dataset", "__pycache__", ".pytest_cache", ".mypy_cache", "*.pyc", "DELIVERY_MANIFEST.json"],
        "file_count": len(files),
        "files": files,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    shown = output.name if output.parent == IMPLEMENTATION_ROOT else str(output)
    print(f"wrote {shown} with {len(files)} files")


if __name__ == "__main__":
    main()
