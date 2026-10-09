#!/usr/bin/env python3
"""Verify and import an EV-SpSegNet A+B server return archive."""

from __future__ import print_function

import argparse
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


EVALUATION_ID = "EV-SpSegNet"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--replace", action="store_true", help="archive and replace an earlier imported measurement")
    return parser.parse_args()


def safe_extract(archive, destination):
    with zipfile.ZipFile(str(archive), "r") as stream:
        for item in stream.infolist():
            target = (destination / item.filename).resolve()
            try:
                target.relative_to(destination.resolve())
            except ValueError:
                raise RuntimeError("unsafe archive path: {}".format(item.filename))
        stream.extractall(str(destination))


def verify_manifest(root):
    manifest_path = root / "RETURN_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "verified":
        raise RuntimeError("return manifest is not verified")
    for item in manifest.get("files", []):
        path = root / item["path"]
        if not path.is_file():
            raise RuntimeError("missing return file: {}".format(item["path"]))
        if path.stat().st_size != int(item["bytes"]) or sha256(path) != item["sha256"]:
            raise RuntimeError("return file/hash mismatch: {}".format(item["path"]))
    report = json.loads((root / "verification_report.json").read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise RuntimeError("server verification report did not pass")


def archive_if_needed(path, archive_root, replace):
    if not path.exists():
        return
    if not replace:
        raise RuntimeError("destination already exists: {}; pass --replace to archive and replace it".format(path))
    destination = archive_root / path.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))


def update_registry(registry_path, weight_sha):
    with registry_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    found = False
    for row in rows:
        if row["evaluation_id"] == EVALUATION_ID:
            found = True
            row["numeric_path"] = "official_float_cuda"
            row["weight_sha256"] = weight_sha
            row["platform_id"] = "official_cuda_server"
            row["notes"] = "threshold=0.90; official EV-SpSegNet checkpoint; A+B server measurement imported"
    if not found:
        raise RuntimeError("model registry does not contain {}".format(EVALUATION_ID))
    with registry_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def main():
    args = parse_args()
    archive = args.archive.resolve()
    project = args.project_root.resolve() if args.project_root else Path(__file__).resolve().parents[4]
    evaluation_root = project / "5_model_evaluation"
    builder = project / "1_H2_model_improvement" / "implementation" / "software" / "scripts" / "rebuild_evaluation_tables.py"
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = evaluation_root / "archive" / ("ev_spsegnet_replaced_" + timestamp)

    with tempfile.TemporaryDirectory(prefix="evspsegnet_import_") as temporary:
        extracted = Path(temporary)
        safe_extract(archive, extracted)
        verify_manifest(extracted)
        source = extracted / "results" / "EV-SpSegNet"
        common = json.loads((source / "complexity" / "common" / "model_profile.json").read_text(encoding="utf-8"))
        weight_sha = common.get("source_weights_sha256", "")
        if len(weight_sha) != 64:
            raise RuntimeError("missing or invalid checkpoint SHA-256")

        a_root = evaluation_root / "A_model_accuracy" / "02_model_records" / "formal" / EVALUATION_ID
        b_root = evaluation_root / "B_model_complexity" / "02_model_records" / "formal" / EVALUATION_ID
        for split, server_name in (("validation", "Validation"), ("test", "Test")):
            a_destination = a_root / split
            b_destination = b_root / split
            archive_if_needed(a_destination, archive_root / "A", args.replace)
            archive_if_needed(b_destination, archive_root / "B", args.replace)
            shutil.copytree(str(source / server_name), str(a_destination))
            b_destination.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(source / "complexity" / server_name / "summary.json"), str(b_destination / "summary.json"))
        common_destination = b_root / "common" / "model_profile.json"
        archive_if_needed(common_destination, archive_root / "B" / "common", args.replace)
        common_destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source / "complexity" / "common" / "model_profile.json"), str(common_destination))

        return_root = a_root / "server_return"
        archive_if_needed(return_root, archive_root / "A", args.replace)
        return_root.mkdir(parents=True, exist_ok=True)
        for name in ("RETURN_MANIFEST.json", "verification_report.json", "provenance.json", "source_archive.json", "server_inspection.json"):
            path = extracted / name
            if path.is_file():
                shutil.copy2(str(path), str(return_root / name))
        shutil.copy2(str(archive), str(return_root / archive.name))

    update_registry(evaluation_root / "model_registry.csv", weight_sha)
    subprocess.check_call([sys.executable, str(builder), "--root", str(evaluation_root), "--category", "all"])
    print("Imported {} and rebuilt all comparison/coverage CSVs.".format(EVALUATION_ID))
    print("Checkpoint SHA-256: {}".format(weight_sha))


if __name__ == "__main__":
    main()
