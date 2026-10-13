"""Build a Linux-safe, hash-verified S3 server upload ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile

HERE = Path(__file__).absolute().parent
ROOT = HERE.parent.parent
H2 = ROOT / "1_H2_model_improvement"
STUDY = H2 / "seven_step_improvement_study" / "H2-pro"
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


def info(name: str, directory: bool = False, executable: bool = False):
    value = zipfile.ZipInfo(name + ("/" if directory and not name.endswith("/") else ""))
    value.create_system = 3
    value.external_attr = (DIR_MODE if directory else (EXEC_MODE if executable else FILE_MODE)) << 16
    if directory:
        value.external_attr |= 0x10
    value.compress_type = zipfile.ZIP_DEFLATED
    return value


def sources():
    files = {}
    for name in ("PROTOCOL.md", "REPORT_A.md", "README_SERVER.md", "runtime.py", "train.py", "preflight.py",
                 "test_runtime.py", "collect_results.py", "run_server.sh", "requirements.txt"):
        files[f"3_ANN_to_SNN/s3/{name}"] = HERE / name
    for root_name in ("runs/ann", "results/GEN2-S3-001-A-revision1"):
        root = HERE / root_name
        for path in sorted(root.glob("*")):
            if path.is_file() and not path.name.endswith(".tmp"):
                files[f"3_ANN_to_SNN/s3/{root_name}/{path.name}"] = path
    files["README.md"] = HERE / "README_SERVER.md"
    for name in ("core.py", "train.py"):
        files[f"3_ANN_to_SNN/s2/{name}"] = HERE.parent / "s2" / name
    files["3_ANN_to_SNN/PLAN.md"] = HERE.parent / "PLAN.md"
    files["3_ANN_to_SNN/reviews/S2_REVIEW.md"] = HERE.parent / "reviews" / "S2_REVIEW.md"
    files["1_H2_model_improvement/implementation/data/index.json"] = H2 / "implementation" / "data" / "index.json"
    files["1_H2_model_improvement/seven_step_improvement_study/H2-pro/protocol.json"] = STUDY / "protocol.json"
    frozen = STUDY / "frozen" / "H2-pro-Causal-ANN-v1"
    for name in ("model_manifest.json", "teacher_exact_int8.npz"):
        files[f"1_H2_model_improvement/seven_step_improvement_study/H2-pro/frozen/H2-pro-Causal-ANN-v1/{name}"] = frozen / name
    counts = {}
    for role, expected in (("train_main", 79), ("train_dev", 10)):
        cache_dir = STUDY / "cache" / role
        found = sorted(cache_dir.glob("*_C1-proposal-motion.npz"))
        if len(found) != expected:
            raise RuntimeError(f"{role} cache count {len(found)} != {expected}")
        counts[role] = len(found)
        for path in found:
            files[f"1_H2_model_improvement/seven_step_improvement_study/H2-pro/cache/{role}/{path.name}"] = path
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing package source: " + ", ".join(missing))
    ann_history = json.loads((HERE / "runs" / "ann" / "history.json").read_text(encoding="utf-8"))
    if len(ann_history) != 10 or ann_history[-1]["epoch"] != 10:
        raise RuntimeError("local S3-B ANN reference is incomplete")
    return files, counts


def verify(name: str, zip_path: Path, manifest: dict):
    expected = {row["path"]: row for row in manifest["files"]}
    with zipfile.ZipFile(zip_path) as archive:
        seen = set()
        for member in archive.infolist():
            if member.create_system != 3 or ".." in Path(member.filename).parts:
                raise RuntimeError(f"unsafe member: {member.filename}")
            mode = (member.external_attr >> 16) & 0xFFFF
            wanted = DIR_MODE if member.is_dir() else (EXEC_MODE if member.filename.endswith("/run_server.sh") else FILE_MODE)
            if mode != wanted:
                raise RuntimeError(f"permission mismatch: {member.filename}")
            if not member.is_dir():
                seen.add(member.filename)
        for relative, row in expected.items():
            value = archive.read(f"{name}/{relative}")
            if len(value) != row["bytes"] or digest_bytes(value) != row["sha256"]:
                raise RuntimeError(f"archive hash mismatch: {relative}")
        if json.loads(archive.read(f"{name}/PACKAGE_MANIFEST.json")) != manifest:
            raise RuntimeError("embedded manifest mismatch")
        if seen != {f"{name}/{relative}" for relative in expected} | {f"{name}/PACKAGE_MANIFEST.json"}:
            raise RuntimeError("unexpected or missing ZIP members")


def build(package_name: str):
    UPLOAD.mkdir(parents=True, exist_ok=True)
    zip_path = UPLOAD / f"{package_name}.zip"
    sha_path = UPLOAD / f"{package_name}.zip.sha256"
    manifest_path = UPLOAD / f"{package_name}.manifest.json"
    if any(path.exists() for path in (zip_path, sha_path, manifest_path)):
        raise FileExistsError("upload artifacts are append-only")
    files, counts = sources()
    placeholder = "1_H2_model_improvement/implementation/data/EV-UAV-dataset/PLACE_DATASET_HERE.txt"
    placeholder_bytes = b"Link the existing server EV-UAV-dataset here; no raw dataset is in this ZIP.\n"
    entries = [{"path": relative, "bytes": path.stat().st_size, "sha256": digest_file(path)}
               for relative, path in sorted(files.items())]
    entries.append({"path": placeholder, "bytes": len(placeholder_bytes),
                    "sha256": digest_bytes(placeholder_bytes)})
    entries.sort(key=lambda row: row["path"])
    manifest = {"package": package_name, "stage": "S3-B; S3-C conditional",
                "dataset_included": False, "allowed_roles": ["train_main", "train_dev"],
                "validation_test_hw_included": False, "frozen_c1_cache_counts": counts,
                "files": entries}
    manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    directories = {Path(package_name)}
    for relative in [*files, placeholder, "PACKAGE_MANIFEST.json"]:
        directories.update((Path(package_name) / relative).parents)
    directories = {path.as_posix() for path in directories if path.as_posix() != "."}
    with zipfile.ZipFile(zip_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for directory in sorted(directories):
            archive.writestr(info(directory, directory=True), b"")
        for relative, path in sorted(files.items()):
            archive.writestr(info(f"{package_name}/{relative}", executable=relative.endswith("/run_server.sh")), path.read_bytes())
        archive.writestr(info(f"{package_name}/{placeholder}"), placeholder_bytes)
        archive.writestr(info(f"{package_name}/PACKAGE_MANIFEST.json"), manifest_bytes)
    manifest_path.write_bytes(manifest_bytes)
    with sha_path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"{digest_file(zip_path)}  {zip_path.name}\n")
    verify(package_name, zip_path, manifest)
    return {"archive": str(zip_path), "bytes": zip_path.stat().st_size,
            "sha256": digest_file(zip_path), "files": len(entries)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-name", default="GEN2_S3_server_20261012")
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    if args.verify_existing:
        path = UPLOAD / f"{args.package_name}.zip"
        manifest = json.loads((UPLOAD / f"{args.package_name}.manifest.json").read_text(encoding="utf-8"))
        verify(args.package_name, path, manifest)
        sha_path = UPLOAD / f"{args.package_name}.zip.sha256"
        digest = digest_file(path)
        if sha_path.exists():
            if sha_path.read_text(encoding="ascii").strip() != f"{digest}  {path.name}":
                raise RuntimeError("external SHA-256 file mismatch")
        else:
            with sha_path.open("w", encoding="ascii", newline="\n") as stream:
                stream.write(f"{digest}  {path.name}\n")
        print(json.dumps({"archive": str(path), "bytes": path.stat().st_size,
                          "sha256": digest, "files": len(manifest["files"]),
                          "status": "verified_existing"}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(build(args.package_name), ensure_ascii=False, indent=2))
