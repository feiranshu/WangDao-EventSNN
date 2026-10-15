"""Resident candidate arrays: no feature construction or disk access per batch."""
from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import torch

from ..h2pro_snn.foundation import role_records
from .common import CACHE_VERSION, output_path, read_json, sha256, verify_baseline


def manifest(directory):
    directory = output_path(directory)
    m = read_json(directory/'manifest.json')
    if (m['version'] != CACHE_VERSION or m['status'] != 'complete'
            or m['identity'] != read_json(directory/'identity.json')
            or m['identity']['baseline_lock_sha256'] != verify_baseline()['baseline_lock_sha256']
            or set(m['roles']) != {'train_main', 'train_dev'}):
        raise ValueError('Wrong feature cache identity')
    for role in ('train_main', 'train_dev'):
        rows, records = m['roles'][role], role_records(role)
        if len(rows) != len(records):
            raise ValueError('Incomplete feature cache')
        for row, rec in zip(rows, records):
            if (row['sequence'] != rec['path'] or row['source_sha256'] != rec['sha256']
                    or row['role'] != role or Path(row['cache_file']).name != row['cache_file']
                    or sha256(directory/row['cache_file']) != row['cache_sha256']):
                raise ValueError('Feature cache role/hash mismatch')
    return m


class ResidentData:
    def __init__(self, directory, device):
        started = time.perf_counter()
        self.directory = output_path(directory)
        self.manifest = manifest(directory)
        self.device = device
        self.roles, self.sequences = {}, {}
        nbytes = 0
        for role, rows in self.manifest['roles'].items():
            parts = {key: [] for key in ('features', 'labels', 'teacher')}
            seqs = []
            offset = 0
            for row in rows:
                with np.load(self.directory/row['cache_file'], allow_pickle=False) as z:
                    a = {k: z[k].copy() for k in z.files}
                c, y = a['candidate'], a['labels']
                if (set(a) != {'features', 'labels', 'teacher', 'candidate'}
                        or c.dtype != np.bool_ or c.shape != (row['events'],)
                        or y.shape != c.shape or not np.isin(y, [0, 1]).all()
                        or int(y.sum()) != row['target_events'] or int(c.sum()) != row['candidate_events']
                        or int(y[c].sum()) != row['candidate_targets']
                        or a['features'].dtype != np.uint8 or a['features'].shape != (int(c.sum()), 36)
                        or a['teacher'].dtype != np.float32 or a['teacher'].shape != (int(c.sum()), 4)
                        or not np.isfinite(a['teacher']).all()):
                    raise ValueError('Invalid candidate features/labels/teacher')
                n = len(a['features'])
                seqs.append({'row': row, 'labels': y, 'candidate': c, 'lo': offset, 'hi': offset+n})
                offset += n
                parts['features'].append(a['features'])
                parts['labels'].append(y[c].astype(np.uint8))
                parts['teacher'].append(a['teacher'])
            self.roles[role] = {}
            for key, values in parts.items():
                host = torch.from_numpy(np.concatenate(values))
                nbytes += host.numel()*host.element_size()
                if device.type == 'cuda':
                    host = host.pin_memory()
                self.roles[role][key] = host.to(device, non_blocking=device.type == 'cuda')
            self.sequences[role] = seqs
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
        self.stats = {'resident_bytes': nbytes, 'placement': str(device),
                      'load_seconds': time.perf_counter()-started, 'per_batch_disk_reads': 0,
                      'per_batch_feature_cpu_computation': False, 'per_batch_input_h2d_bytes': 0,
                      'epoch_order_h2d_bytes': self.size('train_main')*8,
                      'sequence_count': sum(map(len, self.sequences.values()))}

    def size(self, role):
        return len(self.roles[role]['features'])

    def batch(self, role, indices):
        return {key: tensor[indices] for key, tensor in self.roles[role].items()}

    def order(self, epoch, seed):
        # One deterministic host permutation per epoch, independent of process RNG.
        # Resuming reconstructs precisely the same order before slicing the cursor.
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(self.size('train_main'))
        return torch.from_numpy(order).to(self.device)
