"""Atomic full state, advisory pause at step boundaries, OS-owned run lock."""
import contextlib
import copy
import os
from pathlib import Path
import random
import signal
import numpy as np
import torch


def cpu_tree(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return type(x)(cpu_tree(v) for v in x)
    return copy.deepcopy(x)


def atomic_checkpoint(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    with temp.open('wb') as f:
        torch.save(cpu_tree(payload),f);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def load_checkpoint(path):return torch.load(path,map_location='cpu',weights_only=True)


def rng_state():
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=(n[0],n[1].tolist(),*n[2:]),
                torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def set_rng(r):
    random.setstate(r['python']);n=r['numpy'];np.random.set_state((n[0],np.array(n[1],np.uint32),*n[2:]))
    torch.set_rng_state(r['torch_cpu'])
    if r['torch_cuda']:torch.cuda.set_rng_state_all(r['torch_cuda'])


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def process_lock(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'.run.lock').open('a+b') as f:
        if f.tell()==0:f.write(b'0');f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:raise RuntimeError('Another process owns this run/cache') from e
        yield


class PauseRequested(Exception):pass


class PauseControl:
    def __init__(self,directory):self.path=Path(directory)/'PAUSE';self.requested=False;self.previous={}
    def __enter__(self):
        def handler(signum,frame):self.requested=True
        for s in (signal.SIGINT,getattr(signal,'SIGBREAK',signal.SIGINT)):
            if s not in self.previous:self.previous[s]=signal.signal(s,handler)
        return self
    def __exit__(self,*args):
        for s,old in self.previous.items():signal.signal(s,old)
    def check(self):
        if self.requested or self.path.exists():raise PauseRequested()
