"""将训练 checkpoint 冻结为可审计的 S0-P NPZ 权重。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "software"))

from wangdao.snn import ProposalSNN, export_proposal_snn


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--proposal-threshold", type=float, required=True)
    arguments = parser.parse_args()
    checkpoint_path = arguments.checkpoint.resolve()
    output_path = arguments.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖冻结权重：{output_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint.get("config")
    if not isinstance(config, dict):
        raise ValueError("checkpoint 缺少训练配置")
    model = ProposalSNN(
        beta=float(config["beta"]),
        threshold=float(config["threshold"]),
        surrogate_slope=float(config["surrogate_slope"]),
        readout=str(config["readout"]),
    )
    model.load_state_dict(checkpoint["model"])
    metadata = {
        "model_id": arguments.model_id,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "proposal_threshold": arguments.proposal_threshold,
        "source_checkpoint": str(checkpoint_path),
    }
    export_proposal_snn(output_path, model, config, metadata)
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(output_path),
                "output_sha256": sha256_file(output_path),
                **metadata,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
