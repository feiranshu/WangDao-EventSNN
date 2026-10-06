"""生成并校验第一批板卡无关 RTL 黄金向量。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.fixed import (  # noqa: E402
    FixedProposalSNN,
    ApproximateProposalSNN,
    fixed_balanced_add_tree,
    load_approximate_proposal_snn,
    load_fixed_proposal_snn,
    loa_add_signed,
    saturate_signed,
    signed_round_shift,
)
from wangdao.s0p import build_s0p_sequence  # noqa: E402


WIDTH = 24
LOW_BITS = 2
THRESHOLD_Q = 1024
BETA_NUMERATOR = 7
BETA_SHIFT = 3
DEFAULT_OUTPUT = ROOT / "hardware" / "vectors"
INDEX_PATH = ROOT / "data" / "index.json"
S1_WEIGHT_PATH = ROOT / "software/weights/S1P_INT8_ACC24_MEM24_Q12.npz"
S3_WEIGHT_PATH = (
    ROOT / "software" / "weights" / "S3P_LOA_SYN13_MEMALL_K2_ACC24_MEM24_Q12.npz"
)
FIRST_LAYER_CHANNELS = ((4, 20), (5, 0), (6, 13))
HARDWARE_WINDOWS = (0, 1, 2, 3)
PATCH_CASES = 12


def encode_signed(value: int, bits: int = WIDTH) -> str:
    digits = (bits + 3) // 4
    return f"{value & ((1 << bits) - 1):0{digits}X}"


def decode_signed(value: str, bits: int = WIDTH) -> int:
    unsigned = int(value, 16)
    if unsigned & (1 << (bits - 1)):
        return unsigned - (1 << bits)
    return unsigned


def scalar_add(first: int, second: int, low_bits: int) -> tuple[int, int]:
    value, saturations = loa_add_signed(first, second, WIDTH, low_bits)
    return int(value.item()), saturations


def lif_step(
    membrane: int, current: int, fire: bool, low_bits: int
) -> tuple[int, int, int]:
    leaked = int(
        signed_round_shift(
            np.asarray(membrane * BETA_NUMERATOR, dtype=np.int64), BETA_SHIFT
        ).item()
    )
    if low_bits:
        updated, saturations = scalar_add(leaked, current, low_bits)
    else:
        value, saturations = saturate_signed(
            np.asarray(leaked + current, dtype=np.int64), WIDTH
        )
        updated = int(value.item())
    spike = int(fire and updated >= THRESHOLD_Q)
    return updated - spike * THRESHOLD_Q, spike, saturations


def atomic_text(path: Path, lines: Iterable[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(lines), encoding="ascii", newline="\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_cases(random_cases: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    minimum = -(1 << (WIDTH - 1))
    maximum = (1 << (WIDTH - 1)) - 1
    targeted = [
        (0, 0),
        (1, 1),
        (3, 3),
        (4, 3),
        (-1, 1),
        (-1, -1),
        (-4, 3),
        (-5, -3),
        (maximum, 0),
        (maximum, 1),
        (maximum, maximum),
        (minimum, 0),
        (minimum, -1),
        (minimum, minimum),
        (0x2AAAAA, 0x155555),
        (-0x2AAAAA, 0x155555),
    ]
    random_values = rng.integers(
        minimum, maximum + 1, size=(random_cases, 2), dtype=np.int64
    )
    return targeted + [(int(row[0]), int(row[1])) for row in random_values]


def tree_cases(random_cases: int, rng: np.random.Generator) -> list[tuple[int, ...]]:
    minimum = -(1 << (WIDTH - 1))
    maximum = (1 << (WIDTH - 1)) - 1
    targeted = [
        (0, 0, 0, 0, 0),
        (1, 2, 3, 4, 5),
        (-1, -2, -3, -4, -5),
        (maximum, 1, -1, 0, 0),
        (minimum, -1, 1, 0, 0),
        (maximum, maximum, minimum, minimum, 3),
        (3, 3, 3, 3, 3),
        (-3, -3, -3, -3, -3),
    ]
    random_values = rng.integers(
        minimum, maximum + 1, size=(random_cases, 5), dtype=np.int64
    )
    return targeted + [tuple(int(item) for item in row) for row in random_values]


def lif_cases(
    random_cases: int, rng: np.random.Generator
) -> list[tuple[int, int, bool]]:
    minimum = -(1 << (WIDTH - 1))
    maximum = (1 << (WIDTH - 1)) - 1
    targeted = [
        (0, 0, False),
        (0, 0, True),
        (0, THRESHOLD_Q - 1, True),
        (0, THRESHOLD_Q, True),
        (1, 0, False),
        (-1, 0, False),
        (4, 0, False),
        (-4, 0, False),
        (maximum, 0, True),
        (minimum, 0, True),
        (maximum, maximum, True),
        (minimum, minimum, False),
        (2048, -768, True),
        (-2048, 3072, True),
    ]
    membrane = rng.integers(
        minimum, maximum + 1, size=random_cases, dtype=np.int64
    )
    current = rng.integers(
        minimum, maximum + 1, size=random_cases, dtype=np.int64
    )
    fire = rng.integers(0, 2, size=random_cases, dtype=np.uint8)
    return targeted + [
        (int(mem), int(cur), bool(enable))
        for mem, cur, enable in zip(membrane, current, fire)
    ]


def write_add_vectors(path: Path, cases: Sequence[tuple[int, int]]) -> int:
    lines = []
    for first, second in cases:
        exact, exact_saturation = scalar_add(first, second, 0)
        approximate, approximate_saturation = scalar_add(first, second, LOW_BITS)
        lines.append(
            " ".join(
                (
                    encode_signed(first),
                    encode_signed(second),
                    encode_signed(exact),
                    str(exact_saturation),
                    encode_signed(approximate),
                    str(approximate_saturation),
                )
            )
            + "\n"
        )
    atomic_text(path, lines)
    return len(lines)


def write_tree_vectors(path: Path, cases: Sequence[tuple[int, ...]]) -> int:
    lines = []
    for inputs in cases:
        values = np.asarray(inputs, dtype=np.int64)[None, :]
        exact, exact_saturation, exact_additions = fixed_balanced_add_tree(
            values, WIDTH, 0
        )
        approximate, approximate_saturation, approximate_additions = (
            fixed_balanced_add_tree(values, WIDTH, LOW_BITS)
        )
        if exact_additions != 4 or approximate_additions != 4:
            raise AssertionError("五输入固定加法树必须包含四次加法")
        fields = [encode_signed(value) for value in inputs]
        fields.extend(
            (
                encode_signed(int(exact.item())),
                str(exact_saturation),
                encode_signed(int(approximate.item())),
                str(approximate_saturation),
            )
        )
        lines.append(" ".join(fields) + "\n")
    atomic_text(path, lines)
    return len(lines)


def write_lif_vectors(
    path: Path, cases: Sequence[tuple[int, int, bool]]
) -> int:
    lines = []
    for membrane, current, fire in cases:
        exact_membrane, exact_spike, exact_saturation = lif_step(
            membrane, current, fire, 0
        )
        approximate_membrane, approximate_spike, approximate_saturation = lif_step(
            membrane, current, fire, LOW_BITS
        )
        lines.append(
            " ".join(
                (
                    encode_signed(membrane),
                    encode_signed(current),
                    str(int(fire)),
                    encode_signed(exact_membrane),
                    str(exact_spike),
                    str(exact_saturation),
                    encode_signed(approximate_membrane),
                    str(approximate_spike),
                    str(approximate_saturation),
                )
            )
            + "\n"
        )
    atomic_text(path, lines)
    return len(lines)


def frozen_first_layer() -> tuple[FixedProposalSNN, ApproximateProposalSNN]:
    """Bind the hardware arithmetic contract to the already frozen exports."""
    calibration = json.loads((ROOT / "results/approx_combination.json").read_text(encoding="utf-8"))
    s1_path = S1_WEIGHT_PATH
    selection = calibration["selection"]
    if (calibration.get("decision") != "freeze_for_validation"
            or selection["frozen_export"] != S3_WEIGHT_PATH.relative_to(ROOT).as_posix()
            or selection["frozen_export_sha256"] != sha256_file(S3_WEIGHT_PATH)):
        raise ValueError("S3 frozen export identity mismatch")
    exact, _, exact_metadata = load_fixed_proposal_snn(s1_path)
    approximate, config, metadata = load_approximate_proposal_snn(S3_WEIGHT_PATH)
    if (metadata["source_s1_weight_sha256"] != sha256_file(s1_path)
            or metadata["dataset_index_sha256"] != sha256_file(INDEX_PATH)
            or exact_metadata["dataset_index_sha256"] != sha256_file(INDEX_PATH)):
        raise ValueError("Frozen model/index identity mismatch")
    required = {"accumulator_bits": 24, "membrane_bits": 24,
                "membrane_fraction_bits": 12, "input_fraction_bits": 8,
                "beta_numerator": 7, "beta_shift": 3, "threshold": 0.25,
                "time_steps": 4, "low_bits": 2, "tree": "balanced_input_index_order",
                "synaptic_layers": [True, False, True], "membrane_updates": [True, True, True]}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError("Frozen hardware arithmetic contract mismatch")
    if exact.configuration != approximate.configuration or exact.threshold_q != THRESHOLD_Q:
        raise ValueError("S1/S3 arithmetic mismatch")
    if not np.array_equal(exact.weights[0], approximate.weights[0]):
        raise ValueError("S1/S3 first-layer weight mismatch")
    if (not np.array_equal(exact.weight_fraction_bits[0], approximate.weight_fraction_bits[0])
            or set(exact.weight_fraction_bits[0].tolist()) != {4, 5, 6}):
        raise ValueError("Frozen first-layer channel scales changed")
    return exact, approximate


def hardware_calibration_path() -> Path:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    if index.get("status") != "ok":
        raise ValueError("数据索引状态无效")
    names = index.get("roles", {}).get("hw_calibration")
    records = index.get("files")
    location = index.get("location")
    if not isinstance(names, list) or not names:
        raise ValueError("HW-calibration 数据角色为空")
    if not isinstance(records, list) or not isinstance(location, str):
        raise ValueError("数据索引缺少文件记录或位置")
    by_name = {
        Path(str(record["path"])).name: str(record["path"])
        for record in records
        if isinstance(record, dict) and "path" in record
    }
    name = "train_022.npz"
    if name not in names:
        raise ValueError("train_022 must remain in HW-calibration")
    if name not in by_name:
        raise ValueError("HW-calibration 首条序列不在文件索引中")
    path = (ROOT / location / by_name[name]).resolve()
    expected_path = (ROOT / "data/EV-UAV-dataset/train/train_022.npz").resolve()
    if path != expected_path:
        raise ValueError("Hardware data must use the official repository location")
    if not path.is_file():
        raise FileNotFoundError(f"HW-calibration 数据文件缺失：{path}")
    record = next(record for record in records if record["path"] == by_name[name])
    if sha256_file(path) != record["sha256"]:
        raise ValueError("HW-calibration sequence hash mismatch")
    return path


def selected_patch_indices(temporal: np.ndarray, count: int) -> list[int]:
    if temporal.ndim != 3 or temporal.shape[0] != 4 or temporal.shape[2] != 128:
        raise ValueError("第一层真实向量必须具有 [4, patch, 128] 形状")
    if count < 1 or count > temporal.shape[1]:
        raise ValueError("第一层真实 patch 数量无效")
    boundary = [0, temporal.shape[1] // 4, temporal.shape[1] // 2, temporal.shape[1] - 1]
    activity = temporal.astype(np.int64).sum(axis=(0, 2))
    ranked = np.argsort(-activity, kind="stable").tolist()
    selected: list[int] = []
    for index in boundary + ranked:
        candidate = int(index)
        if candidate not in selected:
            selected.append(candidate)
        if len(selected) == count:
            break
    return selected


def write_weight_file(path: Path, weights: np.ndarray) -> int:
    source = np.asarray(weights, dtype=np.int8).reshape(-1)
    atomic_text(path, (encode_signed(int(value), 8) + "\n" for value in source))
    return int(source.size)


def first_layer_result(
    features: np.ndarray,
    weights: np.ndarray,
    weight_fraction_bits: int,
    membrane: int,
    low_bits: int,
) -> tuple[int, int, int, int]:
    products = (
        np.asarray(features, dtype=np.int64)
        * np.asarray(weights, dtype=np.int64)
    )[None, :]
    raw, tree_saturations, additions = fixed_balanced_add_tree(
        products, WIDTH, low_bits
    )
    if additions != 127:
        raise AssertionError("128 输入固定加法树必须包含 127 次加法")
    right_shift = 8 + int(weight_fraction_bits) - 12
    current = int(signed_round_shift(raw, right_shift).item())
    membrane_out, spike, membrane_saturations = lif_step(
        membrane, current, True, low_bits
    )
    return membrane_out, spike, tree_saturations, membrane_saturations


def write_first_layer_vectors(
    path: Path,
    temporal: np.ndarray,
    weights: Sequence[np.ndarray],
    fractions: Sequence[int],
    patch_indices: Sequence[int],
) -> int:
    lines: list[str] = []
    for patch_index in patch_indices:
        exact_membranes = [0] * len(weights)
        approximate_membranes = [0] * len(weights)
        for step in range(temporal.shape[0]):
            features = temporal[step, patch_index]
            fields = [encode_signed(int(value), 8) for value in features]
            for channel, (channel_weights, fraction) in enumerate(
                zip(weights, fractions)
            ):
                exact_in = exact_membranes[channel]
                approximate_in = approximate_membranes[channel]
                exact = first_layer_result(
                    features, channel_weights, fraction, exact_in, 0
                )
                approximate = first_layer_result(
                    features, channel_weights, fraction, approximate_in, LOW_BITS
                )
                exact_membranes[channel] = exact[0]
                approximate_membranes[channel] = approximate[0]
                fields.extend(
                    (
                        encode_signed(exact_in),
                        encode_signed(approximate_in),
                        encode_signed(exact[0]),
                        str(exact[1]),
                        str(exact[2]),
                        str(exact[3]),
                        encode_signed(approximate[0]),
                        str(approximate[1]),
                        str(approximate[2]),
                        str(approximate[3]),
                    )
                )
            lines.append(" ".join(fields) + "\n")
    atomic_text(path, lines)
    return len(lines)


def parse_add_vectors(path: Path) -> int:
    records = 0
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 6:
            raise ValueError("加法器黄金向量字段数错误")
        first, second = decode_signed(fields[0]), decode_signed(fields[1])
        exact = scalar_add(first, second, 0)
        approximate = scalar_add(first, second, LOW_BITS)
        expected = (
            decode_signed(fields[2]),
            int(fields[3]),
            decode_signed(fields[4]),
            int(fields[5]),
        )
        if expected != (exact[0], exact[1], approximate[0], approximate[1]):
            raise AssertionError("加法器黄金向量回读不一致")
        records += 1
    return records


def parse_tree_vectors(path: Path) -> int:
    records = 0
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 9:
            raise ValueError("加法树黄金向量字段数错误")
        inputs = np.asarray(
            [decode_signed(field) for field in fields[:5]], dtype=np.int64
        )[None, :]
        exact = fixed_balanced_add_tree(inputs, WIDTH, 0)
        approximate = fixed_balanced_add_tree(inputs, WIDTH, LOW_BITS)
        expected = (
            decode_signed(fields[5]),
            int(fields[6]),
            decode_signed(fields[7]),
            int(fields[8]),
        )
        actual = (
            int(exact[0].item()),
            exact[1],
            int(approximate[0].item()),
            approximate[1],
        )
        if expected != actual:
            raise AssertionError("加法树黄金向量回读不一致")
        records += 1
    return records


def parse_lif_vectors(path: Path) -> int:
    records = 0
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != 9:
            raise ValueError("神经元黄金向量字段数错误")
        membrane = decode_signed(fields[0])
        current = decode_signed(fields[1])
        fire = bool(int(fields[2]))
        exact = lif_step(membrane, current, fire, 0)
        approximate = lif_step(membrane, current, fire, LOW_BITS)
        expected = (
            decode_signed(fields[3]),
            int(fields[4]),
            int(fields[5]),
            decode_signed(fields[6]),
            int(fields[7]),
            int(fields[8]),
        )
        if expected != (*exact, *approximate):
            raise AssertionError("神经元黄金向量回读不一致")
        records += 1
    return records


def parse_first_layer_vectors(
    path: Path,
    weights: Sequence[np.ndarray],
    fractions: Sequence[int],
) -> int:
    records = 0
    expected_fields = 128 + len(weights) * 10
    for line in path.read_text(encoding="ascii").splitlines():
        fields = line.split()
        if len(fields) != expected_fields:
            raise ValueError("第一层黄金向量字段数错误")
        features = np.asarray(
            [int(field, 16) for field in fields[:128]], dtype=np.uint8
        )
        offset = 128
        for channel, (channel_weights, fraction) in enumerate(
            zip(weights, fractions)
        ):
            record = fields[offset + channel * 10 : offset + (channel + 1) * 10]
            exact_in = decode_signed(record[0])
            approximate_in = decode_signed(record[1])
            exact = first_layer_result(
                features, channel_weights, fraction, exact_in, 0
            )
            approximate = first_layer_result(
                features, channel_weights, fraction, approximate_in, LOW_BITS
            )
            stored = (
                decode_signed(record[2]),
                int(record[3]),
                int(record[4]),
                int(record[5]),
                decode_signed(record[6]),
                int(record[7]),
                int(record[8]),
                int(record[9]),
            )
            if stored != (*exact, *approximate):
                raise AssertionError("第一层黄金向量回读不一致")
        records += 1
    return records


def generate(output_directory: Path, seed: int, random_cases: int) -> dict[str, object]:
    if random_cases < 1:
        raise ValueError("随机向量数量必须为正数")
    output_directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = {
        "fixed_add": output_directory / "fixed_add.txt",
        "fixed_add_tree": output_directory / "fixed_add_tree.txt",
        "lif_neuron": output_directory / "lif_neuron.txt",
        "linear_lif_neuron": output_directory / "linear_lif_neuron.txt",
    }
    _, model = frozen_first_layer()
    channel_weights = [
        np.ascontiguousarray(model.weights[0][channel], dtype=np.int8)
        for _fraction, channel in FIRST_LAYER_CHANNELS
    ]
    fractions = [fraction for fraction, _channel in FIRST_LAYER_CHANNELS]
    for (fraction, channel), actual in zip(
        FIRST_LAYER_CHANNELS, fractions
    ):
        if int(model.weight_fraction_bits[0][channel]) != actual:
            raise AssertionError("冻结第一层代表通道的小数位发生变化")
    calibration_path = hardware_calibration_path()
    sequence = build_s0p_sequence(
        calibration_path, model.time_steps, windows=HARDWARE_WINDOWS
    )
    temporal = sequence.temporal_features.transpose(1, 0, 2, 3).reshape(
        model.time_steps, -1, model.weights[0].shape[1]
    )
    patch_indices = selected_patch_indices(temporal, PATCH_CASES)
    weight_paths = {
        f"layer1_weight_f{fraction}": output_directory / f"layer1_weight_f{fraction}.txt"
        for fraction in fractions
    }
    weight_counts = {
        name: write_weight_file(path, channel_weights[index])
        for index, (name, path) in enumerate(weight_paths.items())
    }
    counts = {
        "fixed_add": write_add_vectors(paths["fixed_add"], add_cases(random_cases, rng)),
        "fixed_add_tree": write_tree_vectors(
            paths["fixed_add_tree"], tree_cases(random_cases, rng)
        ),
        "lif_neuron": write_lif_vectors(
            paths["lif_neuron"], lif_cases(random_cases, rng)
        ),
        "linear_lif_neuron": write_first_layer_vectors(
            paths["linear_lif_neuron"],
            temporal,
            channel_weights,
            fractions,
            patch_indices,
        ),
    }
    verified = {
        "fixed_add": parse_add_vectors(paths["fixed_add"]),
        "fixed_add_tree": parse_tree_vectors(paths["fixed_add_tree"]),
        "lif_neuron": parse_lif_vectors(paths["lif_neuron"]),
        "linear_lif_neuron": parse_first_layer_vectors(
            paths["linear_lif_neuron"], channel_weights, fractions
        ),
    }
    if verified != counts:
        raise AssertionError("黄金向量生成数量与回读数量不一致")
    files = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "records": counts[name],
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    weight_files = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "values": weight_counts[name],
            "sha256": sha256_file(path),
        }
        for name, path in weight_paths.items()
    }
    result: dict[str, object] = {
        "schema_version": 1,
        "status": "ok",
        "experiment_id": "RTL-UNIT-VECTORS-v1",
        "stage": "P4-rtl-unit-vectors",
        "data_role": "synthetic_and_hw_calibration_golden_vectors",
        "primary_metrics": {
            "vector_files": len(files),
            "weight_files": len(weight_files),
            "records": int(sum(counts.values())),
            "verified_records": int(sum(verified.values())),
        },
        "decision": "ready_for_rtl_simulation",
        "conclusion": "第一批板卡无关算术与神经元黄金向量已生成并通过回读校验。",
        "configuration": {
            "width": WIDTH,
            "approximate_low_bits": LOW_BITS,
            "rounding": "nearest_ties_away_from_zero",
            "overflow": "saturate",
            "tree": "balanced_input_index_order",
            "membrane_fraction_bits": 12,
            "threshold_q": THRESHOLD_Q,
            "beta_numerator": BETA_NUMERATOR,
            "beta_shift": BETA_SHIFT,
            "reset": "soft_reset",
            "seed": seed,
            "random_cases_per_vector_type": random_cases,
            "first_layer": {
                "inputs": 128,
                "representative_channels": [
                    {
                        "channel": channel,
                        "weight_fraction_bits": fraction,
                        "right_shift": 8 + fraction - 12,
                    }
                    for fraction, channel in FIRST_LAYER_CHANNELS
                ],
                "patches": len(patch_indices),
                "time_steps_per_patch": model.time_steps,
                "windows": list(HARDWARE_WINDOWS),
            },
        },
        "files": files,
        "weight_files": weight_files,
        "source": {
            "dataset_sequence": calibration_path.relative_to(ROOT).as_posix(),
            "dataset_role": "hw_calibration",
            "selected_patch_indices": patch_indices,
            "frozen_weight": S3_WEIGHT_PATH.relative_to(ROOT).as_posix(),
            "frozen_weight_sha256": sha256_file(S3_WEIGHT_PATH),
        },
        "test_used": False,
        "validation_used": False,
    }
    manifest = output_directory / "manifest.json"
    temporary = manifest.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(manifest)
    return result


def engine_trace(features: np.ndarray, model: FixedProposalSNN, low_bits: int) -> np.ndarray:
    """Full 128-input golden tree, independent of the RTL's grouped schedule."""
    batch = len(features)
    membrane = np.zeros((batch, 48), dtype=np.int64)
    trace = np.zeros((batch, 4, 48, 3), dtype=np.int64)
    for step in range(4):
        values = features[:, step, None, :].astype(np.int64) * model.weights[0][None, :, :]
        tree_saturations = np.zeros((batch, 48), dtype=np.int64)
        while values.shape[-1] > 1:
            # One guard bit exposes overflow before the required 24-bit clamp.
            raw, _ = loa_add_signed(values[..., ::2], values[..., 1::2], 25, low_bits)
            tree_saturations += ((raw < -(1 << 23)) | (raw > (1 << 23) - 1)).sum(axis=-1)
            values = np.clip(raw, -(1 << 23), (1 << 23) - 1)
        current = np.empty_like(membrane)
        for channel in range(48):
            current[:, channel] = signed_round_shift(values[:, channel, 0],
                int(model.weight_fraction_bits[0][channel]) - 4)
        leaked = signed_round_shift(membrane * 7, 3)
        raw, _ = loa_add_signed(leaked, current, 25, low_bits)
        membrane_saturations = (raw < -(1 << 23)) | (raw > (1 << 23) - 1)
        updated = np.clip(raw, -(1 << 23), (1 << 23) - 1)
        spikes = updated >= THRESHOLD_Q
        updated -= spikes * THRESHOLD_Q
        reference_current, reference_tree_saturations = model._current(features[:, step], 0, 8)
        reference_spikes, reference_membrane, reference_membrane_saturations = model._update(
            reference_current, membrane, True, 0)
        if (not np.array_equal(reference_current, current)
                or not np.array_equal(reference_spikes, spikes)
                or not np.array_equal(reference_membrane, updated)
                or reference_tree_saturations != int(tree_saturations.sum())
                or reference_membrane_saturations != int(membrane_saturations.sum())):
            raise AssertionError("First-layer trace differs from frozen model")
        membrane = updated
        trace[:, step, :, 0] = (updated & 0xFFFFFF) | (spikes.astype(np.int64) << 24)
        trace[:, step, :, 1] = tree_saturations
        trace[:, step, :, 2] = membrane_saturations
    return trace


def zero_group_cases() -> tuple[list[np.ndarray], list[str]]:
    """Exercise each mask address at its first/last byte and each bank independently."""
    cases, names = [], []
    for group in range(64):
        for lane in (0, 7):
            case = np.zeros(512, dtype=np.uint8)
            case[group * 8 + lane] = 255
            cases.append(case.reshape(4, 128))
            names.append(f"group{group}_lane{lane}")
    for lane in range(8):
        case = np.zeros(512, dtype=np.uint8)
        case[lane::8] = 255
        cases.append(case.reshape(4, 128))
        names.append(f"all_groups_lane{lane}")
    for parity in (0, 1):
        case = np.zeros((64, 8), dtype=np.uint8)
        case[parity::2] = 255
        cases.append(case.reshape(4, 128))
        names.append(f"alternating_groups_{parity}")
    return cases, names


def generate_engine(output_directory: Path, seed: int, real_patches: int, directed_groups: bool = False) -> dict[str, Any]:
    if not 0 <= real_patches <= 336:
        raise ValueError("real-patches must be between 0 and 336")
    if (output_directory / "first_layer_manifest.json").exists():
        raise FileExistsError("Refusing to overwrite an existing engine vector manifest; use a new output directory")
    exact, approximate = frozen_first_layer()
    source = hardware_calibration_path()
    rng = np.random.default_rng(seed)
    cases = [np.zeros((4, 128), dtype=np.uint8), np.full((4, 128), 255, dtype=np.uint8)]
    for step in range(4):
        case = np.zeros((4, 128), dtype=np.uint8)
        case[step, (0, 7, 64, 127)[step]] = 255
        cases.append(case)
    cases.extend(rng.integers(0, 256, (4, 4, 128), dtype=np.uint8))
    case_names = ["zero", "maximum", "single_step0_index0", "single_step1_index7",
                  "single_step2_index64", "single_step3_index127", "random0", "random1", "random2", "random3"]
    if directed_groups:
        extra_cases, extra_names = zero_group_cases()
        cases.extend(extra_cases)
        case_names.extend(extra_names)
    synthetic_count = len(cases)
    if synthetic_count + real_patches > 346:
        raise ValueError("Engine bench supports at most 346 patches; run directed groups with --real-patches 0")
    if real_patches:
        sequence = build_s0p_sequence(source, 4, windows=(0,))
        cases.extend(sequence.temporal_features[0].transpose(1, 0, 2)[:real_patches])
    features = np.stack(cases)
    exact_trace = engine_trace(features, exact, 0)
    approximate_trace = engine_trace(features, approximate, 2)
    combined = np.concatenate((exact_trace, approximate_trace), axis=-1).reshape(-1, 6)
    output_directory.mkdir(parents=True, exist_ok=True)
    input_path = output_directory / "first_layer_inputs.txt"
    expected_path = output_directory / "first_layer_expected.txt"
    atomic_text(input_path, (" ".join(f"{int(value):02X}" for value in case.flat) + "\n" for case in features))
    atomic_text(expected_path, (f"{a:08X} {b} {c} {d:08X} {e} {f}\n" for a, b, c, d, e, f in combined))
    # Read back the public byte/word layout, not just the in-memory golden arrays.
    restored_input = np.asarray([[int(v, 16) for v in line.split()]
        for line in input_path.read_text(encoding="ascii").splitlines()], dtype=np.uint8)
    restored_output = np.asarray([[int(v, 16 if i in (0, 3) else 10)
        for i, v in enumerate(line.split())]
        for line in expected_path.read_text(encoding="ascii").splitlines()], dtype=np.int64)
    if not np.array_equal(restored_input, features.reshape(-1, 512)) or not np.array_equal(restored_output, combined):
        raise AssertionError("Engine vector readback mismatch")
    weight_files = {}
    for bank in range(8):
        path = output_directory / f"layer1_bank{bank}.txt"
        values = exact.weights[0][:, bank::8].reshape(-1)
        write_weight_file(path, values)
        restored = np.asarray([decode_signed(v, 8) for v in path.read_text().split()], dtype=np.int8)
        if not np.array_equal(restored, values):
            raise AssertionError("Weight bank readback mismatch")
        weight_files[path.stem] = {"path": path.name, "values": 768, "sha256": sha256_file(path)}
    shift_path = output_directory / "layer1_shifts.txt"
    shifts = exact.weight_fraction_bits[0].astype(np.int64) - 4
    atomic_text(shift_path, (f"{value:X}\n" for value in shifts))
    if [int(v, 16) for v in shift_path.read_text().split()] != shifts.tolist():
        raise AssertionError("Shift ROM readback mismatch")
    weight_files[shift_path.stem] = {"path": shift_path.name, "values": 48, "sha256": sha256_file(shift_path)}
    result: dict[str, Any] = {
        "schema_version": 1, "status": "ok", "experiment_id": "RTL-ZERO-GROUP-VECTORS-v1" if directed_groups else "RTL-FIRST-LAYER-VECTORS-v1",
        "stage": "P4-rtl-first-layer-vectors", "data_role": "synthetic_and_hw_calibration_golden_vectors",
        "primary_metrics": {"patches": len(features), "output_records": len(combined), "verified_records": len(combined)},
        "decision": "ready_for_rtl_simulation", "conclusion": "First-layer vectors match the frozen S1/S3 models and serialized layouts.",
        "configuration": {"patches": len(features), "synthetic_patches": synthetic_count,
            "real_patches": real_patches, "complete_window": real_patches == 336,
            "window": 0, "channels": 48, "time_steps": 4, "seed": seed,
            "synthetic_cases": case_names,
            "input_order": "patch,time,feature", "output_order": "patch,time,channel",
            "modes": ["exact", "loa_k2"], "lanes": 8,
            "right_shifts": shifts.tolist(), "reset_between_patches": True},
        "files": {"first_layer_inputs": {"path": input_path.name, "records": len(features), "sha256": sha256_file(input_path)},
                  "first_layer_expected": {"path": expected_path.name, "records": len(combined), "sha256": sha256_file(expected_path)}},
        "weight_files": weight_files,
        "source": {"dataset_sequence": source.relative_to(ROOT).as_posix(), "dataset_role": "hw_calibration",
                   "dataset_index_sha256": sha256_file(INDEX_PATH), "dataset_sequence_sha256": sha256_file(source),
                   "frozen_weight": S3_WEIGHT_PATH.relative_to(ROOT).as_posix(), "frozen_weight_sha256": sha256_file(S3_WEIGHT_PATH)},
        "test_used": False, "validation_used": False,
    }
    (output_directory / "first_layer_manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def second_layer_trace(spikes: np.ndarray, model: FixedProposalSNN, low_bits: int) -> np.ndarray:
    """Binary synapses are exact in both frozen modes; only the membrane uses LOA."""
    if spikes.ndim != 3 or spikes.shape[1:] != (4, 48) or not np.isin(spikes, (0, 1)).all():
        raise ValueError("Second-layer input must have shape (patches, 4, 48) and be binary")
    weights = model.weights[1].astype(np.int64)
    fractions = model.weight_fraction_bits[1].astype(np.int64)
    if weights.shape != (24, 48) or set(fractions.tolist()) != {4, 5, 6, 7}:
        raise ValueError("Unexpected frozen second-layer weights/scaling")
    # Bound every binary input, including patterns absent from calibration.
    bounds = np.stack((np.minimum(weights, 0).sum(-1), np.maximum(weights, 0).sum(-1)))
    aligned_bounds = bounds << (12 - fractions)
    if np.any(aligned_bounds < -(1 << 23)) or np.any(aligned_bounds >= (1 << 23)):
        raise ValueError("Second-layer current does not fit the 24-bit RTL interface")
    membrane = np.zeros((len(spikes), 24), dtype=np.int64)
    trace = np.zeros((len(spikes), 4, 24, 3), dtype=np.int64)
    for step in range(4):
        # Any partial sum of 48 signed int8 weights fits acc24, independent of order.
        current = (spikes[:, step].astype(np.int64) @ weights.T) << (12 - fractions)
        leaked = signed_round_shift(membrane * 7, 3)
        raw, _ = loa_add_signed(leaked, current, 25, low_bits)
        saturated = (raw < -(1 << 23)) | (raw >= (1 << 23))
        updated = np.clip(raw, -(1 << 23), (1 << 23) - 1)
        output_spikes = updated >= THRESHOLD_Q
        updated -= output_spikes * THRESHOLD_Q
        reference_current, tree_sats = model._current(spikes[:, step], 1, 0)
        reference_spikes, reference_membrane, membrane_sats = model._update(reference_current, membrane, True, 1)
        if (not np.array_equal(reference_current, current) or tree_sats != 0
                or not np.array_equal(reference_spikes, output_spikes)
                or not np.array_equal(reference_membrane, updated)
                or membrane_sats != int(saturated.sum())):
            raise AssertionError("Second-layer trace differs from frozen model")
        trace[:, step, :, 0] = (updated & 0xFFFFFF) | (output_spikes.astype(np.int64) << 24)
        trace[:, step, :, 2] = saturated
        membrane = updated
    return trace


def generate_two_layers(output_directory: Path, seed: int, real_patches: int) -> dict[str, Any]:
    """Reuse checked first-layer fixtures; add independent binary and cascade oracles."""
    import shutil
    from scripts.run_rtl_simulation import validate_manifest, check_rows

    manifest_path = output_directory / "two_layer_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("Refusing to overwrite existing two-layer vectors")
    if not 0 <= real_patches <= 336:
        raise ValueError("real-patches must be between 0 and 336")
    exact, approximate = frozen_first_layer()
    source = hardware_calibration_path()
    if (not np.array_equal(exact.weights[1], approximate.weights[1])
            or not np.array_equal(exact.weight_fraction_bits[1], approximate.weight_fraction_bits[1])):
        raise ValueError("Second-layer S1/S3 frozen weights/scaling differ")
    original, _ = validate_manifest(DEFAULT_OUTPUT, "engine")
    if (original["source"]["dataset_sequence_sha256"] != sha256_file(source)
            or original["source"]["frozen_weight_sha256"] != sha256_file(S3_WEIGHT_PATH)
            or original["configuration"]["real_patches"] != 336):
        raise ValueError("First-layer fixture provenance differs from frozen sources")
    features = np.asarray([[int(v, 16) for v in line.split()]
        for line in (DEFAULT_OUTPUT / "first_layer_inputs.txt").read_text(encoding="ascii").splitlines()],
        dtype=np.uint8).reshape(-1, 4, 128)[:10 + real_patches]
    if real_patches:
        sequence = build_s0p_sequence(source, 4, windows=(0,))
        if not np.array_equal(features[10:], sequence.temporal_features[0].transpose(1, 0, 2)[:real_patches]):
            raise ValueError("Saved first-layer features differ from the official calibration window")
    first_traces = [engine_trace(features, exact, 0), engine_trace(features, approximate, 2)]
    binary_cases = [np.zeros((4, 48), dtype=np.uint8), np.ones((4, 48), dtype=np.uint8)]
    for index in range(48):
        case = np.zeros((4, 48), dtype=np.uint8)
        case[index % 4, index] = 1
        binary_cases.append(case)
    # Channel-specific extreme currents exercise all four scales with both signs.
    for positive in (True, False):
        for weights in exact.weights[1]:
            binary_cases.append(np.tile(weights > 0 if positive else weights < 0, (4, 1)).astype(np.uint8))
    binary_cases.extend(np.random.default_rng(seed).integers(0, 2, (4, 4, 48), dtype=np.uint8))
    binary = np.stack(binary_cases)
    standalone = [second_layer_trace(binary, exact, 0), second_layer_trace(binary, approximate, 2)]
    cascade = [second_layer_trace((first_traces[0][..., 0] >> 24) & 1, exact, 0),
               second_layer_trace((first_traces[1][..., 0] >> 24) & 1, approximate, 2)]
    output_directory.mkdir(parents=True, exist_ok=True)
    files = {}
    for prefix, inputs, traces in (("second_layer", binary, standalone), ("two_layer", features, cascade)):
        path = output_directory / f"{prefix}_inputs.txt"
        atomic_text(path, (" ".join(f"{int(v):02X}" for v in case.flat) + "\n" for case in inputs))
        check_rows(path, len(inputs), [(16, 1 if prefix == "second_layer" else 255)] * inputs[0].size)
        files[path.stem] = {"path": path.name, "records": len(inputs), "sha256": sha256_file(path)}
        path = output_directory / f"{prefix}_expected.txt"
        combined = np.concatenate(traces, axis=-1).reshape(-1, 6)
        atomic_text(path, (f"{a:08X} {b} {c} {d:08X} {e} {f}\n" for a, b, c, d, e, f in combined))
        restored = np.asarray([[int(v, 16 if i in (0, 3) else 10) for i, v in enumerate(line.split())]
            for line in path.read_text(encoding="ascii").splitlines()], dtype=np.int64)
        if not np.array_equal(restored, combined):
            raise AssertionError("Second-layer vector readback mismatch")
        files[path.stem] = {"path": path.name, "records": len(combined), "sha256": sha256_file(path)}
    # Retain the layer-one intermediate oracle so the cascade bench checks both layers.
    path = output_directory / "two_layer_intermediate.txt"
    intermediate = np.concatenate(first_traces, axis=-1).reshape(-1, 6)
    atomic_text(path, (f"{a:08X} {b} {c} {d:08X} {e} {f}\n" for a, b, c, d, e, f in intermediate))
    files[path.stem] = {"path": path.name, "records": len(intermediate), "sha256": sha256_file(path)}
    weight_files = dict(original["weight_files"])
    for entry in weight_files.values():
        src, dst = DEFAULT_OUTPUT / entry["path"], output_directory / entry["path"]
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
    for bank in range(8):
        values = [decode_signed(v, 8) for v in (output_directory / f"layer1_bank{bank}.txt").read_text(encoding="ascii").split()]
        if values != exact.weights[0][:, bank::8].reshape(-1).tolist():
            raise ValueError("Saved first-layer bank differs from frozen weights")
    if [int(v, 16) for v in (output_directory / "layer1_shifts.txt").read_text(encoding="ascii").split()] != (exact.weight_fraction_bits[0] - 4).tolist():
        raise ValueError("Saved first-layer scaling differs from frozen model")
    for bank in range(8):
        path = output_directory / f"layer2_bank{bank}.txt"
        values = exact.weights[1][:, bank::8].reshape(-1)
        write_weight_file(path, values)
        if [decode_signed(v, 8) for v in path.read_text(encoding="ascii").split()] != values.tolist():
            raise AssertionError("Second-layer bank readback mismatch")
        weight_files[path.stem] = {"path": path.name, "values": 144, "sha256": sha256_file(path)}
    path = output_directory / "layer2_shifts.txt"
    shifts = 12 - exact.weight_fraction_bits[1].astype(np.int64)
    atomic_text(path, (f"{v:X}\n" for v in shifts))
    weight_files[path.stem] = {"path": path.name, "values": 24, "sha256": sha256_file(path)}
    result = {
        "schema_version": 1, "status": "ok", "experiment_id": "RTL-TWO-LAYER-VECTORS-v1",
        "stage": "P4-rtl-two-layer-vectors", "data_role": "synthetic_and_hw_calibration_golden_vectors",
        "primary_metrics": {"standalone_patches": len(binary), "cascade_patches": len(features),
            "output_records": (len(binary) + len(features)) * 96, "intermediate_records": len(intermediate)},
        "decision": "ready_for_rtl_simulation", "conclusion": "Both hidden layers match frozen S1/S3 arithmetic.",
        "configuration": {"standalone_patches": len(binary), "patches": len(features),
            "synthetic_cascade_patches": 10, "real_patches": real_patches, "complete_window": real_patches == 336,
            "window": 0, "time_steps": 4, "channels": [48, 24], "lanes": 8, "seed": seed,
            "binary_cases": {"zero": 1, "ones": 1, "single_input": 48, "channel_sign_extremes": 48, "random": 4},
            "layer2_left_shifts": shifts.tolist(), "layer2_synaptic_low_bits": [0, 0],
            "membrane_low_bits": [0, 2], "input_order": "patch,time,feature", "output_order": "patch,time,channel",
            "reset_between_patches": True},
        "files": files, "weight_files": weight_files,
        "source": {**original["source"], "first_layer_manifest_sha256": sha256_file(DEFAULT_OUTPUT / "first_layer_manifest.json"),
            "exact_weight_sha256": sha256_file(S1_WEIGHT_PATH)},
        "test_used": False, "validation_used": False,
    }
    atomic_text(manifest_path, [json.dumps(result, indent=2) + "\n"])
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--random-cases", type=int, default=128)
    suite = parser.add_mutually_exclusive_group()
    suite.add_argument("--engine", action="store_true")
    suite.add_argument("--two-layers", action="store_true")
    parser.add_argument("--real-patches", type=int, default=336)
    parser.add_argument("--directed-groups", action="store_true", help="Extra first-layer group-mask boundary cases")
    args = parser.parse_args()
    if args.directed_groups and not args.engine:
        parser.error("directed-groups requires --engine")
    return args


def main() -> None:
    arguments = parse_arguments()
    if arguments.two_layers:
        result = generate_two_layers(arguments.output_dir.resolve(), arguments.seed, arguments.real_patches)
    elif arguments.engine:
        result = generate_engine(arguments.output_dir.resolve(), arguments.seed, arguments.real_patches, arguments.directed_groups)
    else:
        result = generate(arguments.output_dir.resolve(), arguments.seed, arguments.random_cases)
    print(json.dumps(result["primary_metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
