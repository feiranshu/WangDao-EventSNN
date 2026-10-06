"""Software reference model for the event-driven SNN accelerator."""

from .aer import PacketType, pack_packet, unpack_packet
from .bittrue import approximate_add, exact_add, lif_step
from .events import EventTable, SliceConfig, load_ev_uav_npz
from .pipeline import PreprocessResult, preprocess
from .quantization import PotCodes, decode_signed_pot, quantize_signed_pot

__all__ = [
    "EventTable",
    "PacketType",
    "PotCodes",
    "PreprocessResult",
    "SliceConfig",
    "approximate_add",
    "decode_signed_pot",
    "exact_add",
    "lif_step",
    "load_ev_uav_npz",
    "pack_packet",
    "preprocess",
    "quantize_signed_pot",
    "unpack_packet",
]
