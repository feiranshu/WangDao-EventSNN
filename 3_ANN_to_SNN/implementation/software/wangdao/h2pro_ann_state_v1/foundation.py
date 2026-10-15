"""Independent identity, including all immutable numerical dependencies."""
import hashlib
from pathlib import Path

from ..h2pro_snn_v2_2.foundation import (
    IMPLEMENTATION, read_json, sha256, verify_baseline, research_sources as previous_sources,
)

PACKAGE = Path(__file__).absolute().parent


def research_sources():
    result = previous_sources()
    paths = sorted(PACKAGE.glob("*.py")) + sorted(PACKAGE.glob("*.json"))
    paths += [IMPLEMENTATION / "software/scripts/train_h2pro_ann_state.py",
              IMPLEMENTATION / "train_local_4060_ann_state.ps1"]
    for path in paths:
        result[path.relative_to(IMPLEMENTATION).as_posix()] = hashlib.sha256(
            path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    return result
