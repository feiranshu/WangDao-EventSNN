"""Download explicit FRED sequences with pinned revision, resume and SHA-256."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

REPOSITORY = "GabrieleMagrini/FRED"
HOST = "https://huggingface.co"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_url(url, headers=None):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={"User-Agent": "WangDao-FRED-adapter/1", **(headers or {})}), timeout=45)


def download_file(url, path, size, expected_sha256):
    """Resume only after validating Content-Range; never replace an existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.stat().st_size != size or sha256(path) != expected_sha256:
            raise ValueError(f"Existing download integrity mismatch: {path.name}")
        return {"file": path.name, "bytes": size, "sha256": expected_sha256, "reused": True}
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(6):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError("Partial download exceeds expected source size")
        if offset == size:
            break
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        try:
            with open_url(url, headers) as response:
                status = response.status
                content_range = response.headers.get("Content-Range", "")
                if status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range)
                    if not match or int(match[1]) != offset or int(match[3]) != size:
                        raise ValueError("Unexpected HTTP Content-Range; refusing to append")
                elif status == 200:
                    # A server may ignore Range. Restart the partial file, never append.
                    offset = 0
                else:
                    raise ValueError(f"Unexpected HTTP status {status}")
                last_print = time.monotonic()
                with partial.open("ab" if offset else "wb") as stream:
                    received = offset
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
                        received += len(chunk)
                        if received > size:
                            raise ValueError("Response exceeds expected source size")
                        if time.monotonic() - last_print >= 15:
                            print(json.dumps({"stage": "download", "file": path.name,
                                "bytes": received, "total": size,
                                "percent": round(received * 100 / size, 1)}), flush=True)
                            last_print = time.monotonic()
            if partial.stat().st_size == size:
                break
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            print(json.dumps({"stage": "download_retry", "file": path.name,
                "attempt": attempt + 1, "error_type": type(error).__name__}), flush=True)
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 15))
    if not partial.exists() or partial.stat().st_size != size:
        raise IOError("Incomplete download; .part retained for the next invocation")
    actual = sha256(partial)
    if actual != expected_sha256:
        raise ValueError("Source SHA-256 mismatch; .part retained for inspection")
    os.replace(partial, path)
    receipt = {"file": path.name, "bytes": size, "sha256": actual, "reused": False}
    atomic_json(path.with_suffix(".download.json"), receipt)
    print(json.dumps({"stage": "download_verified", **receipt}), flush=True)
    return receipt


def download_sequences(directory, sequences, split="train"):
    if split not in ("train", "test") or not sequences or any(int(i) < 0 for i in sequences):
        raise ValueError("Specify a split and nonnegative sequence IDs")
    if len(set(sequences)) != len(sequences):
        raise ValueError("Duplicate sequence IDs")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    selection_path = directory / "source_selection.json"
    requested = [f"{split}/{int(i)}.zip" for i in sequences]
    if selection_path.exists():
        selection = json.loads(selection_path.read_text("utf-8"))
        if selection["repository"] != REPOSITORY or [r["path"] for r in selection["files"]] != requested:
            raise ValueError("Different source selection; use a separate download directory")
    else:
        with open_url(f"{HOST}/api/datasets/{REPOSITORY}") as response:
            revision = json.load(response)["sha"]
        with open_url(f"{HOST}/api/datasets/{REPOSITORY}/tree/{revision}/{split}?limit=1000") as response:
            listing = json.load(response)
        records = {r["path"]: r for r in listing if r["type"] == "file"}
        selection = {"repository": REPOSITORY, "revision": revision,
            "selection_reason": "Explicit sequence IDs selected before model evaluation; adapter pilot only",
            "files": [{"path": p, "size": records[p]["size"],
                       "sha256": records[p]["lfs"]["oid"]} for p in requested]}
        atomic_json(selection_path, selection)
    receipts = []
    for record in selection["files"]:
        name = Path(record["path"]).stem
        url = f"{HOST}/datasets/{REPOSITORY}/resolve/{selection['revision']}/{record['path']}?download=true"
        receipts.append(download_file(url, directory / f"{split}_{int(name):03d}.zip",
                                      record["size"], record["sha256"]))
    return {**selection, "downloads": receipts}
