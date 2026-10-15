"""One-factor control: remove floor quantization, retain the v2.2 state policy.

This recurrent ANN has continuous nonnegative activations. Negative residuals,
pre-reset signed prototype readout and detached-reset gradient are deliberately
retained. It isolates quantization, not the merits of a conventional ANN/RNN.
"""
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from ..h2pro_snn_v2_2.refiner import (
    RefinerConfig as SNNConfig, StreamRefiner as SNNRefiner, signed_voltage_tail,
)

LAYERS = ("stem", "hidden", "head")
STAT_NAMES = ("current_absmax", "current_rms", "voltage_absmax", "voltage_rms",
              "activation_in_threshold_units_max", "activation_in_threshold_units_mean", "positive_fraction")


@dataclass(frozen=True)
class RefinerConfig(SNNConfig):
    version: str = "phase_grid_ann_state_v1"

    def __post_init__(self):
        if self.version != "phase_grid_ann_state_v1":
            raise ValueError("Expected the independently versioned analog-state control")
        values = asdict(self)
        values["version"] = "phase_grid_refiner_v2_2"
        SNNConfig(**values)  # All other model validation remains exactly matched.


def analog_cell(voltage, threshold):
    activation = F.relu(voltage / threshold)
    # Retain the identical detached subtraction and operation order. In exact
    # arithmetic the remaining state is min(voltage, 0); do not silently change
    # the reset gradient to the true derivative of min() in this control.
    return activation, voltage - threshold * activation.detach()


class StreamRefiner(SNNRefiner):
    def __init__(self, config=RefinerConfig(), foreground_prior=0.1):
        super().__init__(config, foreground_prior)

    def _lif(self, voltage):
        return analog_cell(voltage, self.config.threshold)

    def architecture(self):
        result = super().architecture()
        result.update(
            readout="4 continuous signed pre-reset prototypes per query + one-time bias",
            numerics="continuous ReLU(v/theta), detached subtraction; theta-scaled charge; negative residual retained",
            control="Only floor quantization removed from all three cells; other v2.2 dynamics/losses held fixed",
            backend="active-cell stem and dense analog convolutions",
            activation_note="Values in threshold units are continuous activations, never integer spikes or AC counts",
            gradient_note="Reset subtraction is detached to match SNN surrogate training; not a conventional RNN gradient",
        )
        return result
