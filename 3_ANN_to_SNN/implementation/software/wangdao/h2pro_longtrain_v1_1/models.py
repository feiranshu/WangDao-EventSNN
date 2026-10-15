"""Unchanged mathematical models and optimization; independently versioned monitoring."""
import hashlib
from pathlib import Path

from ..h2pro_longtrain_v1.models import (ARMS, IMPLEMENTATION, make_model, training_module,
    profile_for as previous_profile, sources as previous_sources)

POLICY = "finite_values_advisory_nonfinite_stops_v1"


def profile_for(arm):
    p = previous_profile(arm)
    p["training"]["advisory_limits"] = p["training"].pop("stability_limits")
    return p


def sources():
    result = previous_sources()
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [IMPLEMENTATION/"software/scripts/train_h2pro_long_v1_1.py",
              IMPLEMENTATION/"train_local_4060_long_v1_1.ps1"]
    for path in paths:
        result[path.relative_to(IMPLEMENTATION).as_posix()] = hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
    return result
