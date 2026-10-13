"""Verify frozen development inputs and CPU/CUDA numerical gate before S3 training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).absolute().parent
S2 = HERE.parent / "s2"
sys.path.insert(0, str(S2))
from core import ResearchANN, ResearchSNN  # noqa: E402
from train import read_index, role_records  # noqa: E402
from runtime import DeviceSNNRunner  # noqa: E402
from train import sha256  # noqa: E402


def synthetic_gate(device: torch.device):
    torch.manual_seed(42)
    features = np.array([[20] * 36, [60] * 36, [100] * 36, [30] * 36], dtype=np.uint8)
    x = np.array([0, 4, 0, 4], dtype=np.int16)
    y = np.zeros(4, dtype=np.int16)
    t = np.array([1, 1, 3, 104], dtype=np.float64)
    target = np.array([0, 1, 0, 1], dtype=np.uint8)
    weight = torch.tensor([1.0, 3.0])
    result = {}
    for kind in ("ann", "snn"):
        cpu = ResearchANN() if kind == "ann" else ResearchSNN()
        other = ResearchANN() if kind == "ann" else ResearchSNN()
        other.load_state_dict(cpu.state_dict())
        other.to(device)
        if kind == "ann":
            from torch.nn import functional as F
            from core import class_logits
            value_cpu = cpu(torch.from_numpy(features.astype(np.float32) / 255.0))
            value_other = other(torch.as_tensor(features.astype(np.float32) / 255.0, device=device))
            ce_cpu = F.cross_entropy(class_logits(value_cpu), torch.from_numpy(target.astype(np.int64)),
                                     reduction="none")
            ce_other = F.cross_entropy(class_logits(value_other), torch.as_tensor(target.astype(np.int64), device=device),
                                       reduction="none")
            target_index = torch.from_numpy(target.astype(np.int64))
            loss_cpu = (ce_cpu * weight[target_index]).sum() / weight[target_index].sum()
            loss_other = (ce_other * weight.to(device)[torch.as_tensor(target.astype(np.int64), device=device)]).sum() / float(weight[target_index].sum())
            prediction_delta = float((value_cpu - value_other.cpu()).abs().max().detach())
            state_delta = 0.0
        else:
            run_cpu = DeviceSNNRunner(cpu, 0)
            run_other = DeviceSNNRunner(other, 0)
            wc, _, mc, _ = run_cpu.run_window(features, x, y, t, target, weight)
            wo, _, mo, _ = run_other.run_window(features, x, y, t, target, weight.to(device))
            loss_cpu, loss_other = wc / mc, wo / mo
            prediction_delta = 0.0
            state_delta = max(float((a - b.cpu()).abs().max().detach())
                              for a, b in zip(run_cpu.state, run_other.state))
        loss_delta = abs(float(loss_cpu.detach()) - float(loss_other.detach().cpu()))
        loss_cpu.backward()
        loss_other.backward()
        gradient_delta = max(float((a.grad - b.grad.cpu()).abs().max().detach())
                             for a, b in zip(cpu.parameters(), other.parameters()))
        opt_cpu = torch.optim.SGD(cpu.parameters(), lr=1e-4)
        opt_other = torch.optim.SGD(other.parameters(), lr=1e-4)
        opt_cpu.step()
        opt_other.step()
        update_delta = max(float((a - b.cpu()).abs().max().detach())
                           for a, b in zip(cpu.parameters(), other.parameters()))
        result[kind] = {"prediction_max_abs_delta": prediction_delta,
                        "state_max_abs_delta": state_delta,
                        "loss_abs_delta": loss_delta,
                        "gradient_max_abs_delta": gradient_delta,
                        "one_step_parameter_max_abs_delta": update_delta}
        if max(result[kind].values()) > 1e-4:
            raise RuntimeError(f"{kind} CPU/{device} synthetic gate failed: {result[kind]}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    index, manifest = read_index()
    train = role_records(index, manifest, "train_main")
    dev = role_records(index, manifest, "train_dev")
    if len(train) != 79 or len(dev) != 10:
        raise RuntimeError("development role cardinality mismatch")
    ann_root = HERE / "runs" / "ann"
    ann_provenance = json.loads((ann_root / "provenance.json").read_text(encoding="utf-8"))
    ann_history = json.loads((ann_root / "history.json").read_text(encoding="utf-8"))
    for key, path in (("s2_core_sha256", S2 / "core.py"),
                      ("s2_loader_sha256", S2 / "train.py"),
                      ("s3_runtime_sha256", HERE / "runtime.py"),
                      ("s3_train_sha256", HERE / "train.py"),
                      ("s3_protocol_sha256", HERE / "PROTOCOL.md")):
        if ann_provenance[key] != sha256(path):
            raise RuntimeError(f"packaged ANN source mismatch: {key}")
    if (ann_provenance["index_sha256"] != manifest["index_sha256"] or len(ann_history) != 10
            or ann_history[-1]["epoch"] != 10 or ann_provenance["model"] != "ann"):
        raise RuntimeError("packaged ANN reference incomplete or wrong source")
    device = torch.device("cuda" if args.require_cuda else "cpu")
    report = {"status": "pass", "allowed_roles": ["train_main", "train_dev"],
              "validation_test_hw_read": False,
              "index_sha256": manifest["index_sha256"],
              "feature_protocol_sha256": manifest["protocol_sha256"],
              "teacher_int8_sha256": manifest["teacher_int8_sha256"],
              "s3_protocol_sha256": sha256(HERE / "PROTOCOL.md"),
              "train_sequences": len(train), "dev_sequences": len(dev),
              "train_candidates": sum(len(r["features"]) for r in train),
              "dev_candidates": sum(len(r["features"]) for r in dev),
              "packaged_ann_epochs": len(ann_history),
              "torch_version": torch.__version__, "numpy_version": np.__version__,
              "cuda_available": bool(torch.cuda.is_available()),
              "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
              "synthetic_cpu_device_gate": synthetic_gate(device)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "train_sequences", "dev_sequences",
                                                  "train_candidates", "dev_candidates", "cuda_device")}), flush=True)


if __name__ == "__main__":
    main()
