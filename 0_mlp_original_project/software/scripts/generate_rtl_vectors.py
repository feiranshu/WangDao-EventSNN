"""生成并校验第一批板卡无关 RTL 黄金向量。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.fixed import (  # noqa: E402
    fixed_balanced_add_tree,
    load_approximate_proposal_snn,
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
    name = str(names[0])
    if name not in by_name:
        raise ValueError("HW-calibration 首条序列不在文件索引中")
    path = ROOT / location / by_name[name]
    if not path.is_file():
        raise FileNotFoundError(f"HW-calibration 数据文件缺失：{path}")
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
    model, configuration, _metadata = load_approximate_proposal_snn(S3_WEIGHT_PATH)
    if configuration.get("low_bits") != LOW_BITS:
        raise ValueError("冻结 S3 权重的近似低位数与 RTL 配置不一致")
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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--random-cases", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    result = generate(arguments.output_dir.resolve(), arguments.seed, arguments.random_cases)
    print(json.dumps(result["primary_metrics"], ensure_ascii=False))


if __name__ == "__main__":
    main()
