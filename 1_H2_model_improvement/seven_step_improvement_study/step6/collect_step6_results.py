#!/usr/bin/env python3
"""Collect Step6 evidence while excluding regenerated candidate caches."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


INCLUDE_SUFFIXES = {".json", ".csv", ".svg", ".png", ".npz", ".pt", ".md", ".txt", ".log"}
EXCLUDE_PARTS = {"cache", "__pycache__"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    decision = root / "step6_decision.json"
    if not decision.is_file():
        raise SystemExit(f"missing step6_decision.json: {root}")
    state = json.loads(decision.read_text(encoding="utf-8"))
    if state.get("status") != "complete" or state.get("validation_used") is not False or state.get("test_used") is not False:
        raise SystemExit("Step6 formal decision is incomplete or data-role isolation failed")
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite {output}")
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in INCLUDE_SUFFIXES
        and not any(part in EXCLUDE_PARTS for part in path.relative_to(root).parts)
        and path.name != "RESULT_MANIFEST.json"
    )
    manifest = {
        "source": str(root), "protocol": "h2-ann-step6-three-seed-stability-v1",
        "validation_used": False, "test_used": False,
        "excluded_regenerable_caches": True,
        "files": [
            {"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in files
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path(root.name) / path.relative_to(root))
        archive.writestr(f"{root.name}/RESULT_MANIFEST.json", payload)
    (root / "RESULT_MANIFEST.json").write_text(payload, encoding="utf-8", newline="\n")
    print(json.dumps({"archive": str(output), "sha256": sha256(output), "files": len(files)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
