"""Generate definitive probes for the observed four-block input layout."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from common import (
    assert_known_patch_baseline,
    decode_time_major_lanes,
    load_model,
    pack_physical_lanes,
    pack_time_major_features,
    sha256_file,
    signed_to_u32,
    write_hex_words,
    write_json,
)


LAYOUTS = (
    ("0123", "0123"),
    ("0123", "0213"),
    ("0123", "1023"),
    ("0132", "0123"),
    ("0132", "2103"),
    ("0213", "0123"),
    ("0213", "0213"),
    ("0213", "0231"),
    ("0213", "0321"),
    ("0213", "1023"),
    ("0231", "0123"),
    ("0231", "0213"),
    ("0231", "0231"),
    ("0231", "0321"),
    ("0231", "1023"),
    ("0321", "0123"),
    ("0321", "0213"),
)


def permutation(value: str):
    return tuple(int(character) for character in value)


def evaluate(model, lanes):
    outputs = {}
    for block_name, lane_name in LAYOUTS:
        temporal = decode_time_major_lanes(
            lanes, permutation(block_name), permutation(lane_name)
        )
        outputs[f"b{block_name}_l{lane_name}"] = signed_to_u32(
            model.forward(temporal[:, None, :]).logits_q[0]
        )
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--temporal",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "real_samples/sample_000/sample_000_temporal_uint8.npy",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "generated/time_major_probe",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--attempts", type=int, default=500)
    arguments = parser.parse_args()

    model, configuration, metadata = load_model()
    assert_known_patch_baseline(model)
    temporal = np.load(arguments.temporal, allow_pickle=False)
    if temporal.shape != (4, 128) or temporal.dtype != np.uint8:
        raise ValueError("real temporal sample must be uint8[4,128]")

    # Probe 0 is the real sample represented in the physical identity layout.
    words0 = pack_time_major_features(temporal, (0, 1, 2, 3), (0, 1, 2, 3))
    lanes0 = np.stack(
        [((words0 >> (8 * lane)) & 255).astype(np.uint8) for lane in range(4)]
    )
    outputs0 = evaluate(model, lanes0)

    # Probe 1 stays in the realistic compressed-count range and is selected
    # only to split layouts that probe 0 cannot distinguish.
    rng = np.random.default_rng(arguments.seed)
    selected = None
    values = np.asarray((22, 35, 44, 52), dtype=np.uint8)
    for attempt in range(arguments.attempts):
        lanes1 = np.zeros((4, 128), dtype=np.uint8)
        mask = rng.random((4, 128)) < 0.10
        lanes1[mask] = rng.choice(values, size=int(mask.sum()))
        outputs1 = evaluate(model, lanes1)
        fingerprints = {
            name: outputs0[name].tobytes() + outputs1[name].tobytes()
            for name in outputs0
        }
        if len(set(fingerprints.values())) == len(LAYOUTS):
            selected = (attempt, lanes1, outputs1)
            break
    if selected is None:
        raise RuntimeError("failed to find a jointly unique sparse probe")

    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    probes = ((lanes0, outputs0, "real_sample_identity"), (selected[1], selected[2], "sparse_disambiguator"))
    manifest_probes = []
    for index, (lanes, outputs, role) in enumerate(probes):
        stem = f"time_major_probe_{index:02d}"
        words_path = output / f"{stem}_words.txt"
        write_hex_words(words_path, pack_physical_lanes(lanes))
        expected = {}
        for name, logits in sorted(outputs.items()):
            path = output / f"{stem}_expected_{name}.txt"
            write_hex_words(path, logits)
            expected[name] = path.name
        manifest_probes.append(
            {
                "role": role,
                "input_words": words_path.name,
                "input_sha256": sha256_file(words_path),
                "expected_files": expected,
            }
        )

    manifest = {
        "format_version": 1,
        "layout": "four 32-word time blocks; four adjacent features per word",
        "block_for_time_definition": "digit t gives physical 32-word block used by software time t",
        "lane_for_feature_offset_definition": "digit r gives byte lane used by feature offset r in each group of four",
        "candidate_layouts": [f"b{block}_l{lane}" for block, lane in LAYOUTS],
        "candidate_count": len(LAYOUTS),
        "joint_unique_fingerprint_count": len(LAYOUTS),
        "sparse_probe_seed": arguments.seed,
        "sparse_probe_attempt": selected[0],
        "model_configuration": configuration,
        "model_metadata": metadata,
        "probes": manifest_probes,
    }
    write_json(output / "time_major_probe_manifest.json", manifest)
    print(f"TIME_MAJOR_PROBES_READY output={output}")
    print(f"layouts={len(LAYOUTS)} joint_unique={len(LAYOUTS)}")
    print(f"sparse_attempt={selected[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

