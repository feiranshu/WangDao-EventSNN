"""Read native ECF chunks with the official codec, without altering system HDF5."""
from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np

from .download import atomic_json, sha256

PACKAGE = Path(__file__).absolute().parent
RAW_DTYPE = np.dtype({"names": ["x", "y", "p", "t"],
    "formats": ["<u2", "<u2", "<i2", "<i8"], "offsets": [0, 2, 4, 8], "itemsize": 16})


def build_codec(directory):
    directory = Path(directory).absolute()
    directory.mkdir(parents=True, exist_ok=True)
    source_paths = [PACKAGE / "ecf_bridge.cpp", *sorted((PACKAGE / "vendor").glob("*.cpp")),
                    *sorted((PACKAGE / "vendor").glob("*.h"))]
    identity = {p.relative_to(PACKAGE).as_posix(): sha256(p) for p in source_paths}
    target = directory / ("fred_ecf.dll" if os.name == "nt" else "libfred_ecf.so")
    receipt_path = directory / "codec_build.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text("utf-8"))
        if receipt["sources"] != identity or sha256(target) != receipt["binary_sha256"]:
            raise ValueError("Existing ECF runtime identity changed; use another runtime directory")
        return target
    compiler = shutil.which("g++")
    if not compiler:
        raise RuntimeError("g++ is required to build the small standalone ECF codec")
    command = [compiler, "-std=c++17", "-O2", "-shared", "-static-libgcc", "-static-libstdc++"]
    if os.name != "nt":
        command.append("-fPIC")
    command += ["-I", str(PACKAGE / "vendor"), str(PACKAGE / "vendor/ecf_codec.cpp"),
                str(PACKAGE / "ecf_bridge.cpp"), "-o", str(target)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    atomic_json(receipt_path, {"sources": identity, "command": command,
        "binary_sha256": sha256(target), "upstream": json.loads((PACKAGE / "vendor/source.json").read_text("utf-8"))})
    return target


class ECFCodec:
    def __init__(self, binary):
        self.library = ctypes.CDLL(str(Path(binary).absolute()))
        for name in ("fred_ecf_decode", "fred_ecf_encode"):
            function = getattr(self.library, name)
            function.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t]
            function.restype = ctypes.c_longlong

    def decode(self, payload, capacity):
        if len(payload) < 12 or not 0 < capacity <= 65535:
            raise ValueError("Invalid ECF payload/capacity")
        buffer = ctypes.create_string_buffer(payload)
        result = np.zeros(capacity, dtype=RAW_DTYPE)
        count = self.library.fred_ecf_decode(buffer, len(payload), result.ctypes.data, capacity)
        if count <= 0 or count > capacity:
            raise ValueError(f"ECF decoder failed ({count})")
        return result[:count]

    def encode(self, events):
        events = np.ascontiguousarray(events, dtype=RAW_DTYPE)
        output = ctypes.create_string_buffer(12 + 65535 * 9)
        size = self.library.fred_ecf_encode(events.ctypes.data, len(events), output, len(output))
        if size <= 0:
            raise ValueError(f"ECF encoder failed ({size})")
        return output.raw[:size]


def read_chunk(dataset, start, codec):
    chunk_size = dataset.chunks[0]
    if start % chunk_size or start >= len(dataset):
        raise ValueError("Chunk start must be aligned and inside the dataset")
    filters = dataset.id.get_create_plist()
    descriptions = [filters.get_filter(i) for i in range(filters.get_nfilters())]
    if not descriptions:
        return np.asarray(dataset[start:start + chunk_size], dtype=RAW_DTYPE)
    if len(descriptions) != 1 or descriptions[0][0] != 0x8ECF:
        raise ValueError("Unrecognized event HDF5 filters")
    mask, payload = dataset.id.read_direct_chunk((start,))
    decoded = (np.frombuffer(payload, dtype=RAW_DTYPE) if mask & 1 else codec.decode(payload, chunk_size))
    valid = min(chunk_size, len(dataset) - start)
    if len(decoded) < valid:
        raise ValueError("Decoded chunk shorter than HDF5 extent")
    return decoded[:valid]
