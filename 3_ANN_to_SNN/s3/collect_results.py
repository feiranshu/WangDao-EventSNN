"""Collect S3 server results with hashes; exclude caches and raw dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).absolute().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(run: Path, model: str, mode: str):
    provenance = json.loads((run / "provenance.json").read_text(encoding="utf-8"))
    history = json.loads((run / "history.json").read_text(encoding="utf-8"))
    if (provenance["status"] != "formal_training" or provenance["model"] != model
            or provenance["mode"] != mode or provenance["train_sequences"] != 79
            or provenance["dev_sequences"] != 10 or len(history) != 10
            or history[-1]["epoch"] != 10 or not (run / "epoch_010.pt").is_file()):
        raise RuntimeError(f"incomplete or wrong task: {run}")
    return provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("b", "c"), required=True)
    parser.add_argument("--runs", type=Path, default=HERE / "runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".sha256").exists():
        raise FileExistsError(args.output)
    tasks = (("ann", "ann", "persistent"), ("persistent_snn", "snn", "persistent")) if args.task == "b" else (
        ("window_reset_snn", "snn", "window_reset"),)
    provenances = [validate(args.runs / folder, model, mode) for folder, model, mode in tasks]
    if len(provenances) == 2:
        for key in ("index_sha256", "feature_protocol_sha256", "s3_protocol_sha256", "seed", "epochs",
                    "train_candidate_events_per_epoch"):
            if provenances[0][key] != provenances[1][key]:
                raise RuntimeError(f"matched task provenance differs: {key}")
    preflight = args.runs / "preflight.json"
    if not preflight.is_file() or json.loads(preflight.read_text(encoding="utf-8"))["status"] != "pass":
        raise RuntimeError("passing preflight report missing")
    files = [preflight]
    for folder, _, _ in tasks:
        root = args.runs / folder
        files += sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in
                        (".json", ".md", ".pt", ".log", ".txt") and not path.name.endswith(".tmp"))
    root = args.runs.resolve()
    manifest = {"task": f"GEN2-S3-00{1 if args.task == 'b' else 2}",
                "validation_test_hw_read": False, "regenerable_caches_excluded": True,
                "files": [{"path": p.resolve().relative_to(root).as_posix(), "bytes": p.stat().st_size,
                           "sha256": sha256(p)} for p in files]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, f"s3_{args.task}_results/{path.resolve().relative_to(root).as_posix()}")
        archive.writestr(f"s3_{args.task}_results/RESULT_MANIFEST.json", payload)
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n", encoding="ascii")
    print(json.dumps({"archive": str(args.output), "sha256": sha256(args.output),
                      "files": len(files)}), flush=True)


if __name__ == "__main__":
    main()
