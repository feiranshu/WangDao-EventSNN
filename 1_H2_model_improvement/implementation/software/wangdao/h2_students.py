"""Shallow H2 Refiner students, distillation, and versioned artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as functional

from .h2.torch_model import QuantizedMLP, prototype_logits


STUDENT_ARTIFACT_FORMAT = 1
STUDENT_ARTIFACT_FAMILY = "h2_shallow_refiner"
STUDENT_VARIANTS = ("S-R1", "S-R2")
FEATURE_VARIANTS = ("A1", "A2", "A3", "A3-null")
PROPOSAL_WEIGHT_COUNT = 7680


def feature_width(feature_variant: str) -> int:
    if feature_variant not in FEATURE_VARIANTS:
        raise ValueError("feature variant must be A1, A2, A3, or A3-null")
    return 36 if feature_variant in ("A3", "A3-null") else 32


def student_dimensions(variant: str, input_width: int) -> Tuple[int, ...]:
    if variant == "S-R1":
        return (int(input_width), 32, 4)
    if variant == "S-R2":
        return (int(input_width), 64, 4)
    raise ValueError("student variant must be S-R1 or S-R2")


def build_student(variant: str, feature_variant: str, seed: int = 42) -> QuantizedMLP:
    torch.manual_seed(int(seed))
    return QuantizedMLP(student_dimensions(variant, feature_width(feature_variant)), quantized=False)


def weight_count(dimensions: Tuple[int, ...]) -> int:
    return int(sum(left * right for left, right in zip(dimensions[:-1], dimensions[1:])))


def complexity_summary(model: QuantizedMLP) -> Dict[str, Any]:
    refiner_weights = weight_count(tuple(model.dimensions))
    total = PROPOSAL_WEIGHT_COUNT + refiner_weights
    return {
        "proposal_dimensions": [128, 48, 24, 16],
        "refiner_dimensions": list(model.dimensions),
        "serial_weight_layers": 3 + len(model.layers),
        "proposal_weights": PROPOSAL_WEIGHT_COUNT,
        "refiner_weights": refiner_weights,
        "total_weights": total,
        "reduction_vs_h2_11904": 1.0 - total / 11904.0,
        "scope": "matrix weights only; excludes scales, activations, caches, and representation",
    }


def temperature_kl(student_raw: torch.Tensor, teacher_raw: torch.Tensor,
                   temperature: float = 2.0) -> torch.Tensor:
    """KL(teacher || student) after aggregating the two prototypes per class."""
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if student_raw.shape != teacher_raw.shape or student_raw.ndim != 2:
        raise ValueError("teacher and student logits must have matching [events, 4] shape")
    student = prototype_logits(student_raw) / float(temperature)
    teacher = prototype_logits(teacher_raw.detach()) / float(temperature)
    return functional.kl_div(
        functional.log_softmax(student, dim=1),
        functional.softmax(teacher, dim=1),
        reduction="batchmean",
    ) * float(temperature) ** 2


def student_objective(student_raw: torch.Tensor, labels: torch.Tensor,
                      teacher_raw: Optional[torch.Tensor] = None,
                      temperature: float = 2.0, kd_weight: float = 0.2) -> Dict[str, torch.Tensor]:
    if not 0.0 <= float(kd_weight) <= 1.0:
        raise ValueError("kd_weight must be in [0, 1]")
    logits = prototype_logits(student_raw)
    probability = torch.softmax(logits, dim=1)[:, 1].clamp(1.0e-6, 1.0 - 1.0e-6)
    target = labels.to(torch.float32)
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    alpha = torch.where(target > 0.5, 0.75, 0.25)
    focal = (-alpha * torch.pow(1.0 - pt, 2.0) * torch.log(pt)).mean()
    intersection = torch.sum(probability * target)
    union = torch.sum(probability) + torch.sum(target) - intersection
    soft_iou = 1.0 - (intersection + 1.0) / (union + 1.0)
    task = focal + 0.5 * soft_iou
    kd = task.new_zeros(()) if teacher_raw is None else temperature_kl(
        student_raw, teacher_raw, temperature
    )
    total = task + float(kd_weight) * kd
    return {"total": total, "task": task, "focal": focal, "soft_iou": soft_iou, "kd": kd}


def save_training_checkpoint(path: Path, model: QuantizedMLP, metadata: Dict[str, Any]) -> None:
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    package = {
        "format_version": 3,
        "artifact_family": STUDENT_ARTIFACT_FAMILY,
        "variant": metadata["student_variant"],
        "dimensions": tuple(model.dimensions),
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "metadata": dict(metadata),
    }
    with path.open("xb") as stream:
        torch.save(package, stream)


def load_training_checkpoint(path: Path, device: torch.device = torch.device("cpu")):
    package = torch.load(Path(path), map_location=device, weights_only=True)
    required = {"format_version", "artifact_family", "variant", "dimensions", "state_dict", "metadata"}
    if not isinstance(package, dict) or not required.issubset(package):
        raise ValueError("invalid shallow Refiner checkpoint")
    if int(package["format_version"]) != 3 or package["artifact_family"] != STUDENT_ARTIFACT_FAMILY:
        raise ValueError("unsupported shallow Refiner checkpoint")
    dimensions = tuple(int(value) for value in package["dimensions"])
    if package["variant"] not in STUDENT_VARIANTS or dimensions != student_dimensions(
        package["variant"], dimensions[0]
    ):
        raise ValueError("checkpoint dimensions do not match its student variant")
    model = QuantizedMLP(dimensions, quantized=False).to(device)
    model.load_state_dict(package["state_dict"], strict=True)
    return model, dict(package["metadata"])


@dataclass(frozen=True)
class StudentArtifact:
    model: QuantizedMLP
    metadata: Dict[str, Any]


def export_student_artifact(model: QuantizedMLP, output: Path,
                            metadata: Dict[str, Any]) -> StudentArtifact:
    output = Path(output)
    variant = str(metadata.get("student_variant"))
    feature_variant = str(metadata.get("feature_variant"))
    if variant not in STUDENT_VARIANTS or feature_width(feature_variant) != model.dimensions[0]:
        raise ValueError("artifact metadata does not match model dimensions")
    arrays: Dict[str, np.ndarray] = {
        "format_version": np.asarray(STUDENT_ARTIFACT_FORMAT, dtype=np.int64),
        "artifact_family": np.asarray(STUDENT_ARTIFACT_FAMILY),
        "variant": np.asarray(variant),
        "feature_variant": np.asarray(feature_variant),
        "dimensions": np.asarray(model.dimensions, dtype=np.int64),
        "numeric_mode": np.asarray("float32"),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True)),
        "layer_count": np.asarray(len(model.layers), dtype=np.int64),
    }
    for index, layer in enumerate(model.layers):
        arrays[f"layer{index}_weight"] = layer.weight.detach().cpu().numpy().astype(np.float32)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    return load_student_artifact(output)


def load_student_artifact(path: Path) -> StudentArtifact:
    with np.load(Path(path), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {"format_version", "artifact_family", "variant", "feature_variant",
                "dimensions", "numeric_mode", "metadata_json", "layer_count"}
    if not required.issubset(arrays):
        raise ValueError("student artifact is missing required metadata")
    if int(arrays["format_version"].item()) != STUDENT_ARTIFACT_FORMAT:
        raise ValueError("unsupported student artifact format")
    if str(arrays["artifact_family"].item()) != STUDENT_ARTIFACT_FAMILY:
        raise ValueError("artifact family is not an H2 shallow Refiner")
    if str(arrays["numeric_mode"].item()) != "float32":
        raise ValueError("unsupported student numeric mode")
    variant = str(arrays["variant"].item())
    feature_variant = str(arrays["feature_variant"].item())
    dimensions = tuple(int(value) for value in arrays["dimensions"].tolist())
    if dimensions != student_dimensions(variant, feature_width(feature_variant)):
        raise ValueError("student artifact dimensions are inconsistent")
    model = QuantizedMLP(dimensions, quantized=False)
    count = int(arrays["layer_count"].item())
    if count != len(model.layers):
        raise ValueError("student artifact layer count mismatch")
    with torch.no_grad():
        for index, layer in enumerate(model.layers):
            key = f"layer{index}_weight"
            if key not in arrays or arrays[key].dtype != np.float32 or arrays[key].shape != tuple(layer.weight.shape):
                raise ValueError("student artifact weight shape or dtype mismatch")
            if not np.isfinite(arrays[key]).all():
                raise ValueError("student artifact contains nonfinite weights")
            layer.weight.copy_(torch.from_numpy(arrays[key].copy()))
    metadata = json.loads(str(arrays["metadata_json"].item()))
    if metadata.get("student_variant") != variant or metadata.get("feature_variant") != feature_variant:
        raise ValueError("student artifact metadata payload mismatch")
    return StudentArtifact(model=model, metadata=metadata)
