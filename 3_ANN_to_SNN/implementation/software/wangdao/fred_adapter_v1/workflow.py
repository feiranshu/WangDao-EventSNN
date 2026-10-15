"""Stream all original events into resumable, separately indexed validation shards."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time
import zipfile

import h5py
import numpy as np

from .core import PROTOCOL, convert_block, frame_members, letterbox, parse_annotations, validate_raw
from .download import atomic_json, sha256
from .ecf import ECFCodec, build_codec, read_chunk

PACKAGE = Path(__file__).absolute().parent


def extract_member(archive, member, target):
    """Use caller-controlled output paths and CRC-checked reads, not ZIP extractall."""
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        digest = hashlib.sha256()
        with archive.open(member) as source:
            for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if sha256(target) != digest.hexdigest():
            raise ValueError("Extracted source changed")
        return
    temporary = target.with_suffix(target.suffix + ".tmp")
    with archive.open(member) as source, temporary.open("wb") as output:
        shutil.copyfileobj(source, output, 8 * 1024 * 1024)
    os.replace(temporary, target)


def source_identity():
    return {p.name: sha256(p) for p in [PACKAGE / "core.py", PACKAGE / "workflow.py", PACKAGE / "ecf.py"]}


def save_shard(directory, sequence, number, blocks, masks):
    events = np.concatenate(blocks)
    mask = np.concatenate(masks)
    relative = f"events/{sequence}/part_{number:05d}.npz"
    target = directory / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    row = {"path": relative, "source_sequence": sequence, "part": number,
        "events": len(events), "target_events_weak": int(events["label"].sum()),
        "evaluated_events": int(mask.sum()), "ignored_label_events": int((~mask).sum()),
        "evaluated_target_events_weak": int(events["label"][mask].sum()),
        "first_timestamp_ms": float(events["t"][0]), "last_timestamp_ms": float(events["t"][-1]),
        "shard_begin_ms": number * PROTOCOL["shard_duration_us"] / 1000,
        "shard_end_ms": (number + 1) * PROTOCOL["shard_duration_us"] / 1000,
        "requires_eval_mask": True, "label_kind": PROTOCOL["label_kind"],
        "independent_sequence": False}
    if target.exists():
        with np.load(target, allow_pickle=False) as previous:
            if (not np.array_equal(previous["ev"], events) or not np.array_equal(previous["eval_mask"], mask)
                    or json.loads(str(previous["metadata"])) != row):
                raise ValueError("Existing shard differs from recomputed source; refusing overwrite")
    else:
        temporary = target.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, ev=events, eval_mask=mask,
                                metadata=np.array(json.dumps(row, sort_keys=True)))
        os.replace(temporary, target)
    row.update({"sha256": sha256(target), "bytes": target.stat().st_size})
    print(json.dumps({"stage": "adapted_shard", "sequence": sequence,
                      "part": number, "events": len(events)}), flush=True)
    return row


def adapt_sequence(download_directory, directory, record, codec):
    split, filename = record["path"].split("/")
    source_id = Path(filename).stem
    sequence = f"{split}_{int(source_id):03d}"
    archive_path = download_directory / f"{sequence}.zip"
    if sha256(archive_path) != record["sha256"]:
        raise ValueError("Download differs from pinned official hash")
    receipt = directory / "sequences" / f"{sequence}.json"
    if receipt.exists():
        prior = json.loads(receipt.read_text("utf-8"))
        if prior["source_archive_sha256"] != record["sha256"]:
            raise ValueError("Source identity changed")
        for row in prior["shards"]:
            if sha256(directory / row["path"]) != row["sha256"]:
                raise ValueError("Completed external shard changed")
        print(json.dumps({"stage": "reuse_verified", "sequence": sequence}), flush=True)
        return prior
    started = time.perf_counter()
    source_directory = directory / "source" / sequence
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        members = frame_members(names)
        annotation_member = f"{source_id}/coordinates.txt"
        annotation_text = archive.read(annotation_member).decode("utf-8")
        annotations = parse_annotations(annotation_text, list(members))
        extract_member(archive, f"{source_id}/Event/events.hdf5", source_directory / "events.hdf5")
        extract_member(archive, annotation_member, source_directory / "coordinates.txt")
        if f"{source_id}/coordinates_rgb.txt" in names:
            extract_member(archive, f"{source_id}/coordinates_rgb.txt", source_directory / "coordinates_rgb.txt")
        atomic_json(source_directory / "frame_timestamps.json", list(members))
    with h5py.File(source_directory / "events.hdf5", "r") as handle:
        attributes = {str(k): str(v) for k, v in handle.attrs.items()}
        match = re.fullmatch(r"(\d+)x(\d+)", attributes.get("geometry", ""))
        if not match:
            raise ValueError("Explicit HDF5 geometry is required")
        geometry = letterbox(int(match[1]), int(match[2]))
        dataset = handle["CD/events"]
        if dataset.dtype.names != ("x", "y", "p", "t") or dataset.dtype.itemsize != 16 or not dataset.chunks:
            raise ValueError("Unexpected Prophesee event dataset layout")
        rows, blocks, masks = [], [], []
        part = None
        previous = None
        first_timestamp = None
        decoded_count = 0
        for start in range(0, len(dataset), dataset.chunks[0]):
            raw = read_chunk(dataset, start, codec)
            validate_raw(raw, geometry["source_width"], geometry["source_height"], previous)
            if first_timestamp is None:
                first_timestamp = int(raw["t"][0])
            previous = int(raw["t"][-1])
            decoded_count += len(raw)
            events, valid = convert_block(raw, annotations, geometry)
            parts = raw["t"] // PROTOCOL["shard_duration_us"]
            boundaries = np.r_[0, np.flatnonzero(parts[1:] != parts[:-1]) + 1, len(parts)]
            for low, high in zip(boundaries[:-1], boundaries[1:]):
                next_part = int(parts[low])
                if part is not None and next_part != part:
                    rows.append(save_shard(directory, sequence, part, blocks, masks))
                    blocks, masks = [], []
                part = next_part
                blocks.append(events[low:high].copy())
                masks.append(valid[low:high].copy())
        if blocks:
            rows.append(save_shard(directory, sequence, part, blocks, masks))
        if decoded_count != len(dataset) or sum(r["events"] for r in rows) != decoded_count:
            raise AssertionError("Event coverage mismatch")
    box_array = np.array([b for frame in annotations["boxes"] for b in frame.values()])
    scaled_sizes = (box_array[:, 2:4] - box_array[:, 0:2]) * geometry["scale"]
    result = {"source_sequence": sequence, "official_split": split,
        "source_archive_sha256": record["sha256"], "source_hdf5_sha256": sha256(source_directory / "events.hdf5"),
        "annotation_sha256": sha256(source_directory / "coordinates.txt"),
        "source_attributes": attributes, "geometry": geometry,
        "raw_events": decoded_count, "output_events": decoded_count, "events_dropped": 0,
        "source_first_timestamp_us": first_timestamp, "source_last_timestamp_us": previous,
        "duration_s": (previous - first_timestamp) / 1000000.0,
        "mean_events_per_s": decoded_count * 1000000.0 / (previous - first_timestamp),
        "frame_count": len(annotations["times"]), "frame_step_us": annotations["frame_step_us"],
        "frames_with_boxes": sum(bool(b) for b in annotations["boxes"]),
        "classes": annotations["classes"], "identity_mapping": annotations["id_map"],
        "mapped_box_width_p10_p50_p90": np.percentile(scaled_sizes[:, 0], [10, 50, 90]).tolist(),
        "mapped_box_height_p10_p50_p90": np.percentile(scaled_sizes[:, 1], [10, 50, 90]).tolist(),
        "shards": rows, "adapt_seconds": time.perf_counter() - started}
    atomic_json(receipt, result)
    return result


def adapt(download_directory, directory, runtime):
    download_directory, directory = Path(download_directory), Path(directory)
    if download_directory.absolute() == directory.absolute():
        raise ValueError("Use separate download and adapted dataset directories")
    selection = json.loads((download_directory / "source_selection.json").read_text("utf-8"))
    identity = {"protocol": PROTOCOL, "source_selection": selection, "sources": source_identity()}
    marker = directory / "identity.json"
    if marker.exists():
        if json.loads(marker.read_text("utf-8")) != identity:
            raise ValueError("Adapter or source identity changed; choose a new output directory")
    else:
        if directory.exists() and any(directory.iterdir()):
            raise ValueError("Refusing to adopt a nonempty untracked output directory")
        atomic_json(marker, identity)
    codec = ECFCodec(build_codec(runtime))
    rows = [adapt_sequence(download_directory, directory, record, codec) for record in selection["files"]]
    files = [f for sequence in rows for f in sequence["shards"]]
    index = {"format_version": 1, "dataset": "FRED-real-event-boxweak-pilot-v1", "protocol": PROTOCOL,
        "source_repository": selection["repository"], "source_revision": selection["revision"],
        "sensor": {"width": 346, "height": 260}, "time_unit": "ms", "requires_eval_mask": True,
        "roles": {"external_adapter_pilot": [r["path"] for r in files]},
        "original_recordings": [r["source_sequence"] for r in rows], "sequence_count": len(rows),
        "file_count": len(files), "events": sum(r["events"] for r in files),
        "evaluated_events": sum(r["evaluated_events"] for r in files),
        "target_events_weak": sum(r["target_events_weak"] for r in files),
        "evaluated_target_events_weak": sum(r["evaluated_target_events_weak"] for r in files),
        "total_bytes": sum(r["bytes"] for r in files), "files": files, "sequences": rows}
    path = directory / "index.json"
    if path.exists():
        if json.loads(path.read_text("utf-8")) != index:
            raise ValueError("Completed external index changed")
    else:
        atomic_json(path, index)
    print(json.dumps({"stage": "adapt_complete", **{k: index[k] for k in
        ("sequence_count", "file_count", "events", "evaluated_events", "target_events_weak", "total_bytes")}}), flush=True)
    return index
