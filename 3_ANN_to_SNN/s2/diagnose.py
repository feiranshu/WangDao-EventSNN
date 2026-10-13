"""Read-only numerical health and cost replay for the trained S2 SNN checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from core import ResearchSNN, SNNRunner, CELLS, STATE_CHANNELS
from train import read_index, role_records, windows


HERE = Path(__file__).absolute().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=HERE / "runs" / "GEN2-S2-001-revision1" / "snn_epoch05.pt")
    parser.add_argument("--output", type=Path, default=HERE / "health.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"append-only health result already exists: {args.output}")
    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    assert checkpoint["model_kind"] == "snn" and checkpoint["epoch"] >= 5
    model = ResearchSNN().eval()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()
    index, manifest = read_index()
    assert checkpoint["provenance"]["index_sha256"] == manifest["index_sha256"]
    records = role_records(index, manifest, "train_dev")
    started = time.perf_counter()
    extrema = {"u1_abs_max": 0.0, "u2_abs_max": 0.0, "u3_abs_max": 0.0,
               "trace_max": 0.0, "trace_min": 0.0}
    totals = {"events": 0, "spikes_l1": 0, "spikes_l2": 0, "spikes_out": 0,
              "input_nonzero": 0, "positive_dt": 0}
    per_sequence = []
    with torch.no_grad():
        for record in records:
            runner = SNNRunner(model, record["t0"])
            for segment in windows(record):
                _, probability = runner.run_window(record["features"][segment],
                                                   record["x"][segment], record["y"][segment],
                                                   record["t"][segment])
                assert np.all(np.isfinite(probability))
                assert np.all((probability >= 0) & (probability <= 1))
                for state in runner.state:
                    assert np.all(np.isfinite(state))
                extrema["u1_abs_max"] = max(extrema["u1_abs_max"], float(np.max(np.abs(runner.state[0]))))
                extrema["u2_abs_max"] = max(extrema["u2_abs_max"], float(np.max(np.abs(runner.state[1]))))
                extrema["u3_abs_max"] = max(extrema["u3_abs_max"], float(np.max(np.abs(runner.state[2]))))
                extrema["trace_max"] = max(extrema["trace_max"], float(np.max(runner.state[3])))
                extrema["trace_min"] = min(extrema["trace_min"], float(np.min(runner.state[3])))
            for key in totals:
                totals[key] += runner.counts[key]
            per_sequence.append({"name": record["name"], "events": runner.counts["events"],
                                 "spikes_l1": runner.counts["spikes_l1"],
                                 "spikes_l2": runner.counts["spikes_l2"],
                                 "spikes_out": runner.counts["spikes_out"]})
    history_path = args.checkpoint.parent / "snn_history.json"
    history = json.loads(history_path.read_text(encoding="utf-8"))
    expected = history[-1]["train_dev"]["cost"]
    for key in totals:
        assert totals[key] == expected[key], f"replay cost mismatch: {key}"
    assert totals["spikes_l1"] * 32 + totals["spikes_l2"] * 4 == expected["refiner_binary_ac"]
    assert CELLS * STATE_CHANNELS * 4 == expected["voltage_and_trace_bytes"]
    report = {"checkpoint": args.checkpoint.as_posix(), "epoch": checkpoint["epoch"],
              "status": "finite_and_cost_replay_equal", "sequences": len(records),
              "state_extrema": extrema,
              "readout_parameters": {"trace_gain": model.trace_gain.detach().tolist(),
                                     "mem_gain": model.mem_gain.detach().tolist(),
                                     "bias": model.readout_bias.detach().tolist()},
              "counts": totals,
              "spike_fraction_l1": totals["spikes_l1"] / (totals["events"] * 48),
              "spike_fraction_l2": totals["spikes_l2"] / (totals["events"] * 32),
              "spike_fraction_out": totals["spikes_out"] / (totals["events"] * 4),
              "state_bytes": expected["voltage_and_trace_bytes"] + expected["timestamp_bytes"],
              "replay_wall_s": time.perf_counter() - started,
              "per_sequence": per_sequence}
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"S2 numerical health verified: {args.output}")


if __name__ == "__main__":
    main()
