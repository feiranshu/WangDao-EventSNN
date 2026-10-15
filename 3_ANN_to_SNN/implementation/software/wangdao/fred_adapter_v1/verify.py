"""Full event-by-event source audit and visual alignment checks; no fitting."""
from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import zipfile

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .core import convert_block, frame_members, parse_annotations
from .download import atomic_json, sha256
from .ecf import ECFCodec, build_codec, read_chunk

IMPLEMENTATION = Path(__file__).absolute().parents[3]
ANN = IMPLEMENTATION.parents[1] / "1_H2_model_improvement/implementation"


def canonical_loader():
    specification = importlib.util.spec_from_file_location("fred_canonical_data", ANN / "software/wangdao/data.py")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.load_events


def verify_dataset(directory, runtime, output):
    directory = Path(directory)
    index = json.loads((directory / "index.json").read_text("utf-8"))
    codec = ECFCodec(build_codec(runtime))
    load_events = canonical_loader()
    reports = []
    for sequence in index["sequences"]:
        source = directory / "source" / sequence["source_sequence"]
        if sha256(source / "events.hdf5") != sequence["source_hdf5_sha256"]:
            raise ValueError("Original extracted HDF5 changed")
        count, valid_count = 0, 0
        per_second = np.zeros(int(sequence["source_last_timestamp_us"] // 1000000) + 1, dtype=np.int64)
        last_t = -np.inf
        with h5py.File(source / "events.hdf5", "r") as handle:
            dataset = handle["CD/events"]
            chunk = None
            chunk_offset, next_chunk = 0, 0
            geometry = sequence["geometry"]
            for row in sequence["shards"]:
                path = directory / row["path"]
                if sha256(path) != row["sha256"]:
                    raise ValueError("Output shard hash mismatch")
                adapted = load_events(path, require_labels=True)
                with np.load(path, allow_pickle=False) as archive:
                    mask = archive["eval_mask"]
                if mask.dtype != np.bool_ or mask.shape != (len(adapted),):
                    raise ValueError("Missing or invalid evaluation mask")
                if adapted["t"][0] < last_t:
                    raise ValueError("Cross-shard time order changed")
                last_t = adapted["t"][-1]
                cursor = 0
                while cursor < len(adapted):
                    if chunk is None or chunk_offset == len(chunk):
                        chunk = read_chunk(dataset, next_chunk, codec)
                        next_chunk += dataset.chunks[0]
                        chunk_offset = 0
                    length = min(len(adapted) - cursor, len(chunk) - chunk_offset)
                    raw = chunk[chunk_offset:chunk_offset + length]
                    part = adapted[cursor:cursor + length]
                    for axis in ("x", "y"):
                        expected = np.floor((raw[axis].astype(np.float64) + 0.5) * geometry["scale"]
                                            + geometry[f"offset_{axis}"]).astype(np.uint16)
                        np.testing.assert_array_equal(part[axis], expected)
                    np.testing.assert_array_equal(part["t"], raw["t"] / 1000.0)
                    np.testing.assert_array_equal(part["p"], raw["p"])
                    cursor += length
                    chunk_offset += length
                count += len(adapted)
                valid_count += int(mask.sum())
                per_second += np.bincount((adapted["t"] // 1000).astype(np.int64), minlength=len(per_second))
            if count != len(dataset) or count != sequence["raw_events"]:
                raise AssertionError("Source and adapted event totals differ")
        reports.append({"sequence": sequence["source_sequence"], "events_checked": count,
            "scorable_weak_labels": valid_count, "canonical_loader_passed": True,
            "every_timestamp_polarity_coordinate_and_order_checked": True,
            "events_removed_or_duplicated": 0, "events_per_source_second": per_second.tolist(),
            "events_in_first_5s": int(per_second[:5].sum()),
            "first_5s_event_fraction": float(per_second[:5].sum() / count)})
        print(json.dumps({"stage": "source_audit_passed", **reports[-1]}), flush=True)
    report = {"status": "passed", "index_sha256": sha256(directory / "index.json"), "sequences": reports,
              "label_kind": index["protocol"]["label_kind"], "requires_eval_mask": True,
              "training_started": False, "formal_external_benchmark": False}
    atomic_json(output, report)
    return report


def events_between(dataset, codec, begin, end):
    """Binary search chunk start timestamps, then read a complete short time range."""
    size = dataset.chunks[0]
    low, high = 0, (len(dataset) + size - 1) // size
    while low < high:
        middle = (low + high) // 2
        block = read_chunk(dataset, middle * size, codec)
        if block["t"][0] < begin:
            low = middle + 1
        else:
            high = middle
    blocks = []
    for start in range(max(0, low - 1) * size, len(dataset), size):
        block = read_chunk(dataset, start, codec)
        blocks.append(block[(block["t"] >= begin) & (block["t"] <= end)])
        if block["t"][-1] > end:
            break
    return np.concatenate(blocks)


def raster(events, width, height, labeled=False):
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    positive = events["p"] > 0
    canvas[events["y"][~positive], events["x"][~positive]] = [57, 104, 163]
    canvas[events["y"][positive], events["x"][positive]] = [220, 114, 65]
    if labeled:
        target = events["label"] > 0
        canvas[events["y"][target], events["x"][target]] = [20, 145, 60]
    return canvas


def make_previews(directory, downloads, runtime, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    directory, downloads, output = Path(directory), Path(downloads), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    index = json.loads((directory / "index.json").read_text("utf-8"))
    codec = ECFCodec(build_codec(runtime))
    rows = []
    for sequence in index["sequences"]:
        name = sequence["source_sequence"]
        source = directory / "source" / name
        g = sequence["geometry"]
        with zipfile.ZipFile(downloads / f"{name}.zip") as archive, h5py.File(source / "events.hdf5", "r") as handle:
            members = frame_members(archive.namelist())
            ann = parse_annotations((source / "coordinates.txt").read_text("utf-8"), list(members))
            figure, axes = plt.subplots(4, 3, figsize=(14, 11.5), dpi=130)
            for row_number, seconds in enumerate([0.1, 15, 45, 90]):
                frame_index = int(np.argmin(np.abs(ann["times"] - seconds * 1000000)))
                stamp = int(ann["times"][frame_index])
                with archive.open(members[stamp]) as stream:
                    original_frame = np.array(Image.open(io.BytesIO(stream.read())).convert("RGB"))
                raw = events_between(handle["CD/events"], codec, stamp - 33333, stamp)
                adapted, mask = convert_block(raw, ann, g)
                axes[row_number, 0].imshow(original_frame)
                axes[row_number, 1].imshow(raster(raw, g["source_width"], g["source_height"]))
                axes[row_number, 2].imshow(raster(adapted, 346, 260, labeled=True))
                for box in ann["boxes"][frame_index].values():
                    for column in (0, 1):
                        axes[row_number, column].add_patch(Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, edgecolor="limegreen", linewidth=1.3))
                    mapped = box * g["scale"] + np.array([g["offset_x"], g["offset_y"], g["offset_x"], g["offset_y"]])
                    axes[row_number, 2].add_patch(Rectangle((mapped[0], mapped[1]), mapped[2]-mapped[0], mapped[3]-mapped[1], fill=False, edgecolor="black", linewidth=1.0))
                for column, title in enumerate(["Official event frame + source box", "Decoded raw events (previous 33.333 ms)", "346 x 260; green = box-derived target"]):
                    axes[row_number, column].set_title(f"{stamp / 1e6:.3f}s | {title}", fontsize=9)
                    axes[row_number, column].axis("off")
                # Official event image background is its most frequent color. Exposure may differ.
                colors, counts = np.unique(original_frame.reshape(-1, 3), axis=0, return_counts=True)
                background = colors[counts.argmax()]
                official_active = np.any(original_frame != background, axis=2)
                decoded_active = np.zeros(official_active.shape, dtype=bool)
                decoded_active[raw["y"], raw["x"]] = True
                coverage = float((official_active & decoded_active).sum() / max(1, official_active.sum()))
                rows.append({"sequence": name, "timestamp_us": stamp, "events": len(raw),
                    "box_derived_target_events": int(adapted["label"].sum()), "eval_mask_events": int(mask.sum()),
                    "official_active_pixel_coverage_by_raw_33ms": coverage,
                    "note": "alignment diagnostic only; official visualization exposure may differ"})
            figure.tight_layout()
            figure.canvas.draw()
            canvas = Image.fromarray(np.asarray(figure.canvas.buffer_rgba())).convert("RGB")
            titled = Image.new("RGB", (canvas.width, canvas.height + 56), "white")
            titled.paste(canvas, (0, 56))
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except OSError:
                font = ImageFont.load_default()
            ImageDraw.Draw(titled).text((24, 14),
                f"FRED {name} | Real events, weak box labels | No model fitting", fill="black", font=font)
            titled.save(output / f"{name}_alignment.png")
            plt.close(figure)
    atomic_json(output / "alignment_checks.json", rows)
    return rows
