"""Track v2.1 diagnostics and frozen v2 numerical dependencies."""
import hashlib
from pathlib import Path

from ..h2pro_snn_v2.foundation import (IMPLEMENTATION, read_json, sha256, verify_baseline,
                                   research_sources as v2_sources)

PACKAGE = Path(__file__).absolute().parent


def research_sources():
    sources = v2_sources()
    paths = sorted(PACKAGE.glob("*.py")) + sorted(PACKAGE.glob("*.json"))
    paths += [IMPLEMENTATION / "software/scripts/train_h2pro_refiner_v2_1.py",
              IMPLEMENTATION / "train_local_4060_v2_1.ps1"]
    for path in paths:
        sources[path.relative_to(IMPLEMENTATION).as_posix()] = hashlib.sha256(
            path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    return sources
