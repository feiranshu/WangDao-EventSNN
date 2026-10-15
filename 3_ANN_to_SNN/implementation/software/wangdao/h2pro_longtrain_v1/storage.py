"""Atomic checkpoints and OS-owned locks, including after a killed process."""
import contextlib
import copy
import os
from pathlib import Path
import random
import signal
import time
from typing import cast, Any

import numpy as np
import torch

from ..h2pro_snn.stream_data import atomic_json
from ..h2pro_snn.training import rng_state, restore_rng


def cpu_tree(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x.copy())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {k: cpu_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(cpu_tree(v) for v in x)
    return copy.deepcopy(x)


def save_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        torch.save(cpu_tree(payload), f)
        f.flush()
        os.fsync(f.fileno())
    # The previous committed checkpoint remains intact until this atomic replace.
    os.replace(temp, path)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def all_rng():
    np_state = cast(tuple[Any, ...], np.random.get_state())
    return {"torch": rng_state(), "python": random.getstate(),
            "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:])}


def set_rng(state):
    restore_rng(state["torch"])
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))


@contextlib.contextmanager
def process_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("a+b")
    try:
        if path.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            raise RuntimeError("Another process owns this run/cache; do not start twice") from e
        yield
    finally:
        # Closing releases the kernel lock; the file's existence is not ownership.
        f.close()


class PauseRequested(Exception):
    pass


class PauseControl:
    def __init__(self, directory):
        self.path = Path(directory) / "PAUSE"
        self.requested = False
        self.previous = {}

    def __enter__(self):
        def handler(signum, frame):
            self.requested = True
        for s in (signal.SIGINT, getattr(signal, "SIGBREAK", signal.SIGINT)):
            if s not in self.previous:
                self.previous[s] = signal.signal(s, handler)
        return self

    def __exit__(self, *args):
        for s, old in self.previous.items():
            signal.signal(s, old)

    def pending(self):
        return self.requested or self.path.exists()


def progress(directory, **values):
    atomic_json(Path(directory) / "progress.json", {"updated_unix_s": time.time(), **values})
