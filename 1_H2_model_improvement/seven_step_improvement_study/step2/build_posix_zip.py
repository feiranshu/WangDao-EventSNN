#!/usr/bin/env python3
"""Create a deterministic ZIP with Linux-safe POSIX permission metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import stat
import zipfile


DIR_MODE = stat.S_IFDIR | 0o755
FILE_MODE = stat.S_IFREG | 0o644
EXECUTABLE_MODE = stat.S_IFREG | 0o755


def info(name: str, mode: int, is_directory: bool = False) -> zipfile.ZipInfo:
    value = zipfile.ZipInfo(name + ("/" if is_directory and not name.endswith("/") else ""))
    value.create_system = 3  # Unix
    value.external_attr = mode << 16
    if is_directory:
        value.external_attr |= 0x10
    value.compress_type = zipfile.ZIP_DEFLATED
    return value


def build(source: Path, output: Path) -> None:
    source = source.resolve(strict=True)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not source.is_dir():
        raise NotADirectoryError(source)
    root = source.parent
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        directories = [source, *sorted(path for path in source.rglob("*") if path.is_dir())]
        for path in directories:
            name = path.relative_to(root).as_posix()
            archive.writestr(info(name, DIR_MODE, is_directory=True), b"")
        for path in sorted(item for item in source.rglob("*") if item.is_file()):
            name = path.relative_to(root).as_posix()
            mode = EXECUTABLE_MODE if path.name == "run_server.sh" else FILE_MODE
            archive.writestr(info(name, mode), path.read_bytes())


def verify(output: Path) -> None:
    with zipfile.ZipFile(output, "r") as archive:
        for member in archive.infolist():
            if member.create_system != 3:
                raise RuntimeError(f"non-Unix ZIP member: {member.filename}")
            mode = (member.external_attr >> 16) & 0xFFFF
            expected = DIR_MODE if member.is_dir() else (
                EXECUTABLE_MODE if member.filename.endswith("/run_server.sh") else FILE_MODE
            )
            if mode != expected:
                raise RuntimeError(
                    f"permission mismatch for {member.filename}: {oct(mode)} != {oct(expected)}"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.source, args.output)
    verify(args.output)
    print(f"POSIX ZIP verified: {args.output}")


if __name__ == "__main__":
    main()
