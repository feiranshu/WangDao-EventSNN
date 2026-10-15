"""Immutable ANN controls plus a matched spiking stem/hidden, linear-head control."""
from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib

from ..h2pro_ann_state_v1 import training as state_training
from ..h2pro_ann_readout_v1 import training as linear_training
from ..h2pro_ann_readout_v1.refiner import RefinerConfig as LinearConfig, StreamRefiner as LinearModel
from ..h2pro_snn_v2.neurons import lif
from ..h2pro_ann_readout_v1.foundation import IMPLEMENTATION, research_sources as previous_sources

ARMS = ("ann_state", "ann_linear", "snn_linear")
ORIGINS = {"ann_state": "h2pro_ann_state_v1_s42", "ann_linear": "h2pro_ann_readout_v1_s42"}


@dataclass(frozen=True)
class SpikingConfig(LinearConfig):
    version: str = "phase_grid_snn_linear_v1"

    def __post_init__(self):
        values = asdict(self)
        if values["version"] != "phase_grid_snn_linear_v1":
            raise ValueError("Wrong paired SNN version")
        values["version"] = "phase_grid_ann_readout_v1"
        LinearConfig(**values)


class SpikingLinearModel(LinearModel):
    def _lif(self, voltage):
        return lif(voltage, self.config.threshold)

    def architecture(self):
        result = super().architecture()
        result.update(control="Matched ANN-linear control: only stem/hidden floor quantization restored",
                      numerics="multi-spike positive-rate STE, detached subtractive reset; signed linear head",
                      model_kind="hybrid SNN; continuous signed readout",
                      backend="active-cell stem; dense hidden/head convolution (no spike sparse kernel)",
                      activation_note="Stem/hidden are integer unit spikes; head positive amplitude is diagnostic only")
        return result


def training_module(arm):
    if arm not in ARMS:
        raise ValueError("Unknown control")
    return state_training if arm == "ann_state" else linear_training


def profile_for(arm):
    p = training_module(arm).load_profile()
    if arm == "snn_linear":
        p["model"]["version"] = "phase_grid_snn_linear_v1"
    return p


def make_model(arm, profile, prior):
    if arm == "snn_linear":
        return SpikingLinearModel(SpikingConfig(**profile["model"]), prior)
    if arm == "ann_state":
        return state_training.StreamRefiner(state_training.RefinerConfig(**profile["model"]), prior)
    if arm == "ann_linear":
        return LinearModel(LinearConfig(**profile["model"]), prior)
    raise ValueError("Unknown control")


def sources():
    result = previous_sources()
    paths = sorted(Path(__file__).parent.glob("*.py"))
    paths += [IMPLEMENTATION / "software/scripts/train_h2pro_long.py", IMPLEMENTATION / "train_local_4060_long.ps1"]
    for p in paths:
        result[p.relative_to(IMPLEMENTATION).as_posix()] = hashlib.sha256(p.read_text(encoding="utf-8").encode()).hexdigest()
    return result
