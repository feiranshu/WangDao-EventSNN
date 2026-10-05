"""Train C0-C4; defaults to a bounded smoke run, never accesses Test."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from wangdao.causal import CausalConfig
from wangdao.causal_training import TrainConfig, train_experiment
from wangdao.runtime import choose_device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["C0", "C1", "C2", "C3", "C4"], default="C0")
    p.add_argument("--output-dir", type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Full 79 Train-main / 10 Train-dev experiment")
    mode.add_argument("--smoke", action="store_true", help="One sequence, eight blocks, one epoch")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--cache-mib", type=int, default=256)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--no-memory", action="store_true")
    p.add_argument("--mlp-hidden", type=int, nargs="+", default=[64, 32])
    p.add_argument("--initialize-from", type=Path, help="Optional predecessor; required trained C3 for formal C4")
    a = p.parse_args()
    if a.threads < 1:
        p.error("--threads must be positive")
    torch.set_num_threads(a.threads)
    model = CausalConfig(variant=a.variant, no_memory=a.no_memory, mlp_hidden=tuple(a.mlp_hidden))
    cfg = TrainConfig(epochs=a.epochs, learning_rate=a.learning_rate, seed=a.seed,
                      cache_mib=a.cache_mib, smoke=not a.run)
    result = train_experiment(a.output_dir, model, cfg, choose_device(a.device), a.initialize_from)
    print(result["decision"], "checkpoint:", a.output_dir / result["best_checkpoint"])


if __name__ == "__main__":
    main()
