"""Versioned identities and the local 4060 experiment contract."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path

from ..h2pro_direct_v1.common import lock_directory, output_path
from ..h2pro_snn.foundation import IMPLEMENTATION, read_json, sha256, verify_baseline
from ..h2pro_snn.stream_data import atomic_json

PACKAGE = Path(__file__).absolute().parent
VERSION = 'h2pro_r2t8_distill_v1'
CACHE_VERSION = 'h2pro_frozen_candidate_features_v1'
DEFAULT_CACHE = IMPLEMENTATION/'results/runs/h2pro_distill_v1_features'
DEFAULT_RUN = IMPLEMENTATION/'results/runs/h2pro_distill_r2t8_v1_s42'
DEFAULT_TEACHER = IMPLEMENTATION/'results/runs/h2pro_teacher_v1'
DEFAULT_EVIDENCE = IMPLEMENTATION/'results/runs/h2pro_proposal_diagnosis_v1'

PROTOCOL = {
    'version': VERSION, 'direction': 4, 'dimensions': [36, 64, 32, 4],
    'proposal': 'frozen exact H2 ANN; routing and all C1 features unchanged',
    'teacher': 'frozen H2-pro exact-int8 export; cached original four-prototype logits',
    'student': 'original r2/T8; initialize ALL layers from the frozen int8 export',
    'weight_update': 'FP32 master weights; int8 round/clamp STE, original fixed per-output scales',
    'activation': 'original clip [0,1] and round-to-even uint8; clamp gradient retained',
    'if': 'T=8, theta=255, v0=127, binary soft reset; independent per event and layer',
    'gradient': 'straight-through round and IF count; no surrogate changes to the forward model',
    'training_backend': 'dense exact-count reference with FP64 integer-valued sums and scale decode',
    'inference': 'exported int8 weights supported by the existing explicit binary spike-add kernel',
    'readout': 'unchanged max of two prototypes per class, threshold 0.35',
    'loss': '(1-alpha)*unweighted CE + alpha*temperature^2*KL(teacher||student)',
    'sampling': 'all Train-main candidates exactly once per epoch, uniform shuffled order',
    'no_direction_5_or_6': True, 'hard_mining': False, 'spike_regularization': False,
    'roles': ['train_main', 'train_dev'], 'reserved_splits_read': False,
    'evaluation': 'full closed-prefix events including fixed Proposal rejects; overall/HD/per-sequence',
    'best_selection': 'highest full-event Train-dev IoU, tie by recall; epoch zero eligible',
    'cost': 'training dense MAC separate from inference-equivalent MAC/AC and scalar/state operations',
    'numerical_checks': 'nonfinite loss, gradients, parameters; no finite-amplitude stop guards',
    'timing': 'layer-serial coding, T is not milliseconds; original 100 ms closure remains',
}


@dataclass(frozen=True)
class Config:
    seed: int = 42
    batch_size: int = 8192
    learning_rate: float = 1e-4
    kd_alpha: float = .5
    temperature: float = 2.

    def validate(self):
        if (type(self.seed) is not int or not 0 <= self.seed < 2**32
                or type(self.batch_size) is not int or self.batch_size < 1):
            raise ValueError('Require seed in [0,2^32) and a positive batch size')
        if (not all(math.isfinite(v) for v in (self.learning_rate, self.kd_alpha, self.temperature))
                or self.learning_rate <= 0 or not 0 <= self.kd_alpha <= 1 or self.temperature <= 0):
            raise ValueError('Invalid learning rate, KD mixing coefficient or temperature')
        return self


def source_hashes():
    paths = list(PACKAGE.glob('*.py')) + [
        IMPLEMENTATION/'software/scripts/train_h2pro_distill.py',
        IMPLEMENTATION/'train_local_4060_distill.ps1']
    for folder, names in {
        'h2pro_direct_v1': ('core.py', 'common.py'),
        'h2pro_longtrain_v1': ('storage.py',),
        'h2pro_snn': ('foundation.py', 'stream_data.py', 'diagnostics.py', 'training.py'),
    }.items():
        paths += [PACKAGE.parent/folder/name for name in names]
    return {p.relative_to(IMPLEMENTATION).as_posix():
            hashlib.sha256(p.read_text('utf-8').encode('utf-8')).hexdigest() for p in sorted(paths)}


def run_identity(cache, config, engine):
    return {'version': VERSION, 'protocol': PROTOCOL, 'config': asdict(config.validate()),
            'baseline': verify_baseline(), 'cache_manifest_sha256': sha256(Path(cache)/'manifest.json'),
            'sources': source_hashes(), 'engine': engine}
