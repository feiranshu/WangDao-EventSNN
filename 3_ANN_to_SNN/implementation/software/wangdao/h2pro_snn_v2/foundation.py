"""Track v2 and its frozen shared dependencies without changing v1 identity."""
import hashlib
from pathlib import Path

from ..h2pro_snn.foundation import (IMPLEMENTATION, read_json, sha256, verify_baseline,
                                   research_sources as v1_sources)

PACKAGE = Path(__file__).absolute().parent


def research_sources():
    sources = v1_sources()
    paths = sorted(PACKAGE.glob("*.py")) + sorted(PACKAGE.glob("*.json"))
    paths += [IMPLEMENTATION / "software/scripts/train_h2pro_refiner_v2.py",
              IMPLEMENTATION / "train_local_4060_v2.ps1"]
    for path in paths:
        sources[path.relative_to(IMPLEMENTATION).as_posix()] = hashlib.sha256(
            path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
    return sources
