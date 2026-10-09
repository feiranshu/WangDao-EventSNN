#!/usr/bin/env python3
"""Collect a verified Step7 result directory without candidate caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile


FILE_MODE = stat.S_IFREG | 0o644
DIR_MODE = stat.S_IFDIR | 0o755


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def info(name: str, directory: bool = False) -> zipfile.ZipInfo:
    item = zipfile.ZipInfo(name + ("/" if directory and not name.endswith("/") else ""))
    item.create_system = 3
    item.external_attr = (DIR_MODE if directory else FILE_MODE) << 16
    if directory:
        item.external_attr |= 0x10
    item.compress_type = zipfile.ZIP_DEFLATED
    return item


def collect(results: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    decision = results / "step7_decision.json"
    manifest = results / "RESULT_MANIFEST.json"
    if not decision.is_file() or not manifest.is_file():
        raise FileNotFoundError("Step7 is incomplete: decision or result manifest missing")
    if json.loads(decision.read_text(encoding="utf-8")).get("status") != "complete":
        raise RuntimeError("Step7 decision is not complete")
    files = [
        path for path in sorted(results.rglob("*"))
        if path.is_file() and "work" not in path.relative_to(results).parts and "__pycache__" not in path.parts
    ]
    root = output.stem
    directories = {root}
    for path in files:
        relative = path.relative_to(results).as_posix()
        directories.update((Path(root) / relative).parents)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for directory in sorted(path.as_posix() if isinstance(path, Path) else path for path in directories):
            archive.writestr(info(directory, directory=True), b"")
        for path in files:
            archive.writestr(info(f"{root}/{path.relative_to(results).as_posix()}"), path.read_bytes())
    payload = output.read_bytes()
    sha = output.with_suffix(output.suffix + ".sha256")
    sha.write_text(f"{digest(payload)}  {output.name}\n", encoding="ascii", newline="\n")
    return {"zip": str(output), "sha256": digest(payload), "files": len(files), "cache_excluded": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.results, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

