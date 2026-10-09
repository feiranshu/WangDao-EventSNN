#!/usr/bin/env python3
"""Build and verify the self-contained Linux-safe Step6 server ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
from typing import Any, Mapping
import zipfile


HERE = Path(__file__).resolve().parent
IMPLEMENTATION = HERE.parents[1] / "implementation"
STEP3 = HERE.parent / "step3"
STEP4 = HERE.parent / "step4"
STEP5 = HERE.parent / "step5"
STEP4_RESULTS = STEP4 / "results" / "step4_formal_20261004"
STEP5_RESULTS = STEP5 / "step5_results_server" / "step5_formal"
UPLOAD = HERE / "upload"
DIR_MODE = stat.S_IFDIR | 0o755
FILE_MODE = stat.S_IFREG | 0o644
EXEC_MODE = stat.S_IFREG | 0o755


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def info(name: str, directory: bool = False, executable: bool = False) -> zipfile.ZipInfo:
    value = zipfile.ZipInfo(name + ("/" if directory and not name.endswith("/") else ""))
    value.create_system = 3
    value.external_attr = (DIR_MODE if directory else (EXEC_MODE if executable else FILE_MODE)) << 16
    if directory:
        value.external_attr |= 0x10
    value.compress_type = zipfile.ZIP_DEFLATED
    return value


def source_files() -> dict[str, Path]:
    names = (
        "README_SERVER.md", "run_step6.py", "step6_common.py", "step6_training.py",
        "collect_step6_results.py", "step6_protocol.json", "requirements.txt", "run_server.sh",
    )
    result = {name: HERE / name for name in names}
    result["README.md"] = HERE / "README_SERVER.md"
    for name, path in (
        ("step3_common.py", STEP3 / "step3_common.py"),
        ("step4_common.py", STEP4 / "step4_common.py"),
        ("step4_training.py", STEP4 / "step4_training.py"),
        ("step5_common.py", STEP5 / "step5_common.py"),
    ):
        result[name] = path
    for path in sorted((HERE / "tests").rglob("*.py")):
        if "__pycache__" not in path.parts:
            result[(Path("tests") / path.relative_to(HERE / "tests")).as_posix()] = path
    runtime = (
        "__init__.py", "data.py", "h2_refiner.py", "h2_research.py", "phase1_common.py",
        "phase1_metrics.py", "runtime.py", "h2/__init__.py", "h2/constants.py",
        "h2/export.py", "h2/features.py", "h2/numpy_model.py", "h2/representation.py",
        "h2/torch_model.py",
    )
    for relative in runtime:
        result[f"software/wangdao/{relative}"] = IMPLEMENTATION / "software" / "wangdao" / relative
    result["software/weights/H2_D1_seed42.npz"] = IMPLEMENTATION / "software" / "weights" / "H2_D1_seed42.npz"
    result["data/index.json"] = IMPLEMENTATION / "data" / "index.json"
    step5_inputs = {
        "frozen_candidate.json": STEP5_RESULTS / "frozen_candidate.json",
        "artifacts/epoch_008.pt": STEP5_RESULTS / "artifacts" / "epoch_008.pt",
        "metrics/A4-2/int8/summary.json": STEP5_RESULTS / "metrics" / "A4-2" / "int8" / "summary.json",
        "metrics/A4-2/int8/bootstrap_event_scores.npz": STEP5_RESULTS / "metrics" / "A4-2" / "int8" / "bootstrap_event_scores.npz",
        "metrics/H2-R/int8/summary.json": STEP5_RESULTS / "metrics" / "H2-R" / "int8" / "summary.json",
        "metrics/H2-R/int8/bootstrap_event_scores.npz": STEP5_RESULTS / "metrics" / "H2-R" / "int8" / "bootstrap_event_scores.npz",
    }
    for relative, path in step5_inputs.items():
        result[f"input/step5_results/{relative}"] = path
    result["input/step4_results/work/density_transform.json"] = STEP4_RESULTS / "work" / "density_transform.json"
    result["input/step4_results/runs/A4-2-null-seed42/summary.json"] = STEP4_RESULTS / "runs" / "A4-2-null-seed42" / "summary.json"
    missing = [str(path) for path in result.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("package source missing: " + ", ".join(missing))
    return result


def verify(package_name: str, zip_path: Path, manifest: Mapping[str, Any]) -> None:
    expected = {row["path"]: row for row in manifest["files"]}
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.create_system != 3:
                raise RuntimeError(f"non-POSIX ZIP member: {member.filename}")
            mode = (member.external_attr >> 16) & 0xFFFF
            wanted = DIR_MODE if member.is_dir() else (EXEC_MODE if member.filename.endswith("/run_server.sh") else FILE_MODE)
            if mode != wanted:
                raise RuntimeError(f"permission mismatch: {member.filename}")
        for name, row in expected.items():
            value = archive.read(f"{package_name}/{name}")
            if len(value) != row["bytes"] or digest_bytes(value) != row["sha256"]:
                raise RuntimeError(f"archive content mismatch: {name}")
        if json.loads(archive.read(f"{package_name}/PACKAGE_MANIFEST.json")) != manifest:
            raise RuntimeError("embedded manifest mismatch")


def build(package_name: str) -> dict:
    UPLOAD.mkdir(parents=True, exist_ok=True)
    zip_path = UPLOAD / f"{package_name}.zip"
    sha_path = UPLOAD / f"{package_name}.zip.sha256"
    manifest_path = UPLOAD / f"{package_name}.manifest.json"
    existing = [str(path) for path in (zip_path, sha_path, manifest_path) if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite: " + ", ".join(existing))
    files = source_files()
    placeholder_name = "data/EV-UAV-dataset/PLACE_DATASET_HERE.txt"
    placeholder = b"Place or symlink the EV-UAV-dataset here. The dataset is not included.\n"
    entries = [
        {"path": name, "bytes": path.stat().st_size, "sha256": digest_file(path)}
        for name, path in sorted(files.items())
    ]
    entries.append({"path": placeholder_name, "bytes": len(placeholder), "sha256": digest_bytes(placeholder)})
    entries.sort(key=lambda row: row["path"])
    manifest = {
        "package": package_name, "dataset_included": False,
        "step5_server_results_included": True,
        "validation_or_test_results_included": False,
        "files": entries,
    }
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    directories = {Path(package_name)}
    for name in [*files, placeholder_name, "PACKAGE_MANIFEST.json"]:
        directories.update((Path(package_name) / name).parents)
    directories = {path.as_posix() for path in directories if path.as_posix() != "."}
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for directory in sorted(directories):
            archive.writestr(info(directory, directory=True), b"")
        for name, path in sorted(files.items()):
            archive.writestr(info(f"{package_name}/{name}", executable=name == "run_server.sh"), path.read_bytes())
        archive.writestr(info(f"{package_name}/{placeholder_name}"), placeholder)
        archive.writestr(info(f"{package_name}/PACKAGE_MANIFEST.json"), manifest_bytes)
    manifest_path.write_bytes(manifest_bytes)
    zip_hash = digest_file(zip_path)
    sha_path.write_text(f"{zip_hash}  {zip_path.name}\n", encoding="ascii", newline="\n")
    verify(package_name, zip_path, manifest)
    return {"zip": str(zip_path), "sha256": zip_hash, "manifest": str(manifest_path), "verified_files": len(entries)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-name", default="H2_step6_server_20261005")
    args = parser.parse_args()
    print(json.dumps(build(args.package_name), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
