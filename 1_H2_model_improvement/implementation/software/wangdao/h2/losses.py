"""Training objectives for the H2 proposal and D1 event refiner."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as functional

from .torch_model import prototype_logits


def proposal_loss(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
    target = targets.to(torch.float32)
    probability = torch.sigmoid(logits).clamp(1.0e-6, 1.0 - 1.0e-6)
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    alpha = torch.where(target > 0.5, 0.75, 0.25)
    focal = (-alpha * torch.pow(1.0 - pt, 2.0) * torch.log(pt)).mean()
    intersection = torch.sum(probability * target)
    dice = 1.0 - (2.0 * intersection + 1.0) / (
        torch.sum(probability) + torch.sum(target) + 1.0
    )
    return {"total": focal + 0.5 * dice, "focal": focal, "soft_dice": dice}


def refiner_loss(
    raw_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_probability: torch.Tensor,
    teacher_mask: torch.Tensor,
    distillation_weight: float = 0.20,
) -> Dict[str, torch.Tensor]:
    """Compute the selected D1 loss: focal + IoU + masked distillation."""
    logits = prototype_logits(raw_logits)
    probability = torch.softmax(logits, dim=1)[:, 1].clamp(1.0e-6, 1.0 - 1.0e-6)
    target = labels.to(torch.float32)
    pt = torch.where(target > 0.5, probability, 1.0 - probability)
    alpha = torch.where(target > 0.5, 0.75, 0.25)
    focal = (-alpha * torch.pow(1.0 - pt, 2.0) * torch.log(pt)).mean()
    intersection = torch.sum(probability * target)
    union = torch.sum(probability) + torch.sum(target) - intersection
    soft_iou = 1.0 - (intersection + 1.0) / (union + 1.0)
    mask = teacher_mask.to(torch.float32)
    distill_values = functional.binary_cross_entropy(
        probability,
        teacher_probability.to(torch.float32).clamp(1.0e-4, 1.0 - 1.0e-4),
        reduction="none",
    )
    distill = (
        (distill_values * mask).sum() / mask.sum()
        if bool(mask.sum().detach() > 0)
        else probability.sum() * 0.0
    )
    total = focal + 0.5 * soft_iou + float(distillation_weight) * distill
    return {
        "total": total,
        "focal": focal,
        "soft_iou": soft_iou,
        "distill": distill,
    }
