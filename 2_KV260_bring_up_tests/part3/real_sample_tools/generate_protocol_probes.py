"""Generate a minimal deterministic set of byte-lane protocol probes."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    DEFAULT_MODEL,
    all_time_to_lane_orders,
    assert_known_patch_baseline,
    load_model,
    order_name,
    pack_physical_lanes,
    sha256_file,
    signed_to_u32,
    temporal_from_lane_values,
    write_hex_words,
    write_json,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "generated" / "lane_probe",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument(
        "--candidate-count",
        type=int,
        default=200,
        help="deterministic random candidates searched by the greedy selector",
    )
    parser.add_argument("--maximum-probes", type=int, default=8)
    return parser.parse_args()


def evaluate_probe(model, lane_values, orders):
    result: dict[str, np.ndarray] = {}
    for order in orders:
        temporal = temporal_from_lane_values(lane_values, order)
        result[order_name(order)] = signed_to_u32(
            model.forward(temporal[:, None, :]).logits_q[0]
        )
    return result


def select_jointly_unique_probes(model, initial_seed, candidate_count, maximum):
    orders = list(all_time_to_lane_orders())
    names = [order_name(order) for order in orders]
    candidates = []
    for seed in range(initial_seed, initial_seed + candidate_count):
        rng = np.random.default_rng(seed)
        lane_values = rng.integers(0, 256, size=(4, 128), dtype=np.uint8)
        outputs = evaluate_probe(model, lane_values, orders)
        candidates.append((seed, lane_values, outputs))

    fingerprints = {name: b"" for name in names}
    selected = []
    while len(set(fingerprints.values())) < len(names):
        if len(selected) >= maximum:
            raise RuntimeError(
                f"failed to make all 24 fingerprints unique with {maximum} probes"
            )
        best = None
        best_score = -1
        best_individual = -1
        for seed, lane_values, outputs in candidates:
            if any(item[0] == seed for item in selected):
                continue
            combined = {
                name: fingerprints[name] + outputs[name].tobytes() for name in names
            }
            score = len(set(combined.values()))
            individual = len({outputs[name].tobytes() for name in names})
            # A stable final seed tie-break keeps generation reproducible.
            rank = (score, individual, seed)
            if rank > (best_score, best_individual, best[0] if best else -1):
                best = (seed, lane_values, outputs)
                best_score = score
                best_individual = individual
        if best is None:
            raise RuntimeError("probe candidate pool is empty")
        selected.append(best)
        for name in names:
            fingerprints[name] += best[2][name].tobytes()

    return selected, fingerprints


def main() -> int:
    arguments = parse_arguments()
    model, configuration, metadata = load_model(arguments.model)
    assert_known_patch_baseline(model)

    selected, fingerprints = select_jointly_unique_probes(
        model,
        arguments.seed,
        arguments.candidate_count,
        arguments.maximum_probes,
    )
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for stale in output.glob("lane_probe_*.txt"):
        stale.unlink()
    for stale in output.glob("lane_probe_*_lane_values.npy"):
        stale.unlink()

    probes = []
    for index, (seed, lane_values, outputs) in enumerate(selected):
        stem = f"lane_probe_{index:02d}"
        words_path = output / f"{stem}_words.txt"
        lanes_path = output / f"{stem}_lane_values.npy"
        write_hex_words(words_path, pack_physical_lanes(lane_values))
        np.save(lanes_path, lane_values, allow_pickle=False)
        expected_files = {}
        expected_hashes = {}
        for name, logits in sorted(outputs.items()):
            path = output / f"{stem}_expected_order_{name}.txt"
            write_hex_words(path, logits)
            expected_files[name] = path.name
            expected_hashes[name] = sha256_file(path)
        probes.append(
            {
                "index": index,
                "seed": seed,
                "individual_unique_output_count": len(
                    {value.tobytes() for value in outputs.values()}
                ),
                "input_words": words_path.name,
                "input_words_sha256": sha256_file(words_path),
                "lane_values_npy": lanes_path.name,
                "lane_values_sha256": sha256_file(lanes_path),
                "expected_files": expected_files,
                "expected_sha256": expected_hashes,
            }
        )

    manifest = {
        "format_version": 2,
        "purpose": "identify software-time to physical-byte-lane permutation",
        "definition": {
            "order_0123": (
                "software t0 reads lane0 bits[7:0], t1 reads lane1, "
                "t2 reads lane2, t3 reads lane3 bits[31:24]"
            ),
            "lane0": "bits[7:0]",
            "lane1": "bits[15:8]",
            "lane2": "bits[23:16]",
            "lane3": "bits[31:24]",
        },
        "candidate_seed_begin": arguments.seed,
        "candidate_seed_count": arguments.candidate_count,
        "selection": "greedy maximum joint fingerprint uniqueness",
        "probe_count": len(probes),
        "candidate_order_count": 24,
        "joint_unique_fingerprint_count": len(set(fingerprints.values())),
        "model": str(arguments.model.resolve()),
        "model_sha256": sha256_file(arguments.model),
        "model_configuration": configuration,
        "model_metadata": metadata,
        "probes": probes,
    }
    write_json(output / "lane_probe_manifest.json", manifest)

    print(f"generated: {output}")
    print(f"selected seeds: {[item[0] for item in selected]}")
    print(f"probe count: {len(probes)}")
    print("candidate orders: 24")
    print(
        "joint unique fingerprints: "
        f"{manifest['joint_unique_fingerprint_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
