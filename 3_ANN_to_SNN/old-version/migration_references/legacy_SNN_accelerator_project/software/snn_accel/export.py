from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .bittrue import approximate_add, exact_add, required_accumulator_bits
from .quantization import decode_signed_pot, quantize_signed_pot


@dataclass(frozen=True)
class LayerSpec:
    name: str
    input_channels: int
    output_channels: int
    kernel: int
    stride: int
    threshold: int
    leak_shift: int
    accumulator_bits: int
    approx_lsb_bits: int
    exponent_min: int = -8
    exponent_max: int = 7
    scale_pow2: int = 0
    weight_offset: int = 0
    weight_count: int = 0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def export_model(
    weights: Mapping[str, np.ndarray],
    layers: Sequence[LayerSpec],
    output_dir: str | Path,
    *,
    input_shape: tuple[int, int, int] = (2, 256, 320),
    timesteps: int = 4,
    model_hash: str = "untrained-reference",
    data_config_hash: str = "unset",
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weight_path = output / "weights.bin"
    exported_layers: list[LayerSpec] = []
    payload = bytearray()
    for layer in layers:
        if layer.name not in weights:
            raise KeyError(f"missing weights for layer {layer.name!r}")
        array = np.asarray(weights[layer.name], dtype=np.float32)
        expected = layer.output_channels * layer.input_channels * layer.kernel * layer.kernel
        if array.size != expected:
            raise ValueError(f"{layer.name}: expected {expected} weights, got {array.size}")
        codes = quantize_signed_pot(
            array,
            exponent_min=layer.exponent_min,
            exponent_max=layer.exponent_max,
            scale_pow2=layer.scale_pow2,
        )
        encoded = codes.packed_bytes()
        exported_layers.append(
            LayerSpec(
                **{
                    **asdict(layer),
                    "weight_offset": len(payload),
                    "weight_count": array.size,
                }
            )
        )
        payload.extend(encoded)
    weight_path.write_bytes(payload)

    vector_path = output / "golden_vectors.npz"
    rng = np.random.default_rng(24307140011)
    operands_a = rng.integers(-2048, 2048, size=1024, dtype=np.int64)
    operands_b = rng.integers(-2048, 2048, size=1024, dtype=np.int64)
    width = max(layer.accumulator_bits for layer in layers)
    approx_bits = max(layer.approx_lsb_bits for layer in layers)
    np.savez_compressed(
        vector_path,
        a=operands_a,
        b=operands_b,
        exact=exact_add(operands_a, operands_b, width),
        approximate=approximate_add(operands_a, operands_b, width, approx_bits),
        width=np.asarray(width),
        approx_lsb_bits=np.asarray(approx_bits),
    )

    manifest = {
        "format_version": "1.0",
        "model_hash": model_hash,
        "data_config_hash": data_config_hash,
        "input_shape": list(input_shape),
        "timesteps": timesteps,
        "outputs": {"P3": [32, 40], "P4": [16, 20], "P5": [8, 10]},
        "weight_encoding": "zero|sign|reserved2|biased_exponent4",
        "layers": [asdict(layer) for layer in exported_layers],
        "checksums": {
            "weights.bin": sha256_file(weight_path),
            "golden_vectors.npz": sha256_file(vector_path),
        },
    }
    manifest_path = output / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def calibrate_layer_width(values: np.ndarray, guard_bits: int = 2) -> int:
    maximum = int(np.max(np.abs(np.asarray(values, dtype=np.int64)), initial=0))
    return required_accumulator_bits(maximum, guard_bits)


def verify_manifest(directory: str | Path) -> None:
    root = Path(directory)
    manifest = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format_version") != "1.0":
        raise ValueError("unsupported manifest version")
    for filename, expected in manifest.get("checksums", {}).items():
        actual = sha256_file(root / filename)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {filename}")

