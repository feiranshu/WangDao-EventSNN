"""Only the experimental objective changes; both teacher and mask are detached."""
import numpy as np
import torch
from torch.nn import functional as F

from ..h2pro_direct_v1.core import probabilities
from ..h2pro_distill_v1.model import class_logits, losses as original_losses, require_finite, update as original_update
from .common import HARD_WEIGHT


def categories(teacher_raw, student_raw, labels, *, role):
    if role != 'train_main':
        raise ValueError('Hard-example construction is Main-only')
    labels = np.asarray(labels)
    if labels.shape != (len(teacher_raw),) or len(student_raw) != len(labels) or not np.isin(labels,[0,1]).all():
        raise ValueError('Invalid Main labels or predictions')
    teacher, student = probabilities(teacher_raw) >= np.float32(.35), probabilities(student_raw) >= np.float32(.35)
    result = np.zeros(len(labels), np.uint8)
    result[(labels == 1) & teacher & ~student] = 1
    result[(labels == 0) & ~teacher & student] = 2
    return result


def weighted_losses(raw, labels, teacher_raw, weights, config):
    if weights.shape != labels.shape or weights.requires_grad:
        raise ValueError('Detached per-event weights required')
    logits, teacher = class_logits(raw), class_logits(teacher_raw.detach())
    ce = F.cross_entropy(logits, labels.long(), reduction='none')
    t = config.temperature
    kd = F.kl_div(F.log_softmax(logits/t,dim=1), F.softmax(teacher/t,dim=1), reduction='none').sum(1)*t**2
    weights = weights.detach().to(dtype=ce.dtype)
    denominator = weights.sum()
    ce_mean, kd_mean = (ce*weights).sum()/denominator, (kd*weights).sum()/denominator
    return (1-config.kd_alpha)*ce_mean+config.kd_alpha*kd_mean, ce_mean, kd_mean


def update(model, optimizer, batch, config, arm):
    # Retaining the exact historical control also retains its FP summation order.
    if arm == 'control':
        result = original_update(model, optimizer, batch, config)
        result['plain_losses'] = result['losses']
        return result
    if arm != 'hard_weighted':
        raise ValueError('Unknown arm')
    model.train(); optimizer.zero_grad(set_to_none=True)
    raw, spikes = model(batch['features'])
    weights = 1.+(HARD_WEIGHT-1.)*(batch['hard_category'] != 0).to(raw.dtype)
    terms = weighted_losses(raw, batch['labels'], batch['teacher'], weights, config)
    with torch.no_grad():
        plain = original_losses(raw.detach(), batch['labels'], batch['teacher'], config)
    require_finite([raw, *terms], 'hard-example outputs/loss')
    terms[0].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    if len(grads) != 3:
        raise RuntimeError('Every original weight matrix must receive a gradient')
    require_finite(grads, 'hard-example gradients')
    norm = torch.linalg.vector_norm(torch.stack([g.double().norm() for g in grads]))
    optimizer.step()
    require_finite(model.parameters(), 'hard-example parameters')
    return {'losses': [float(t.detach()) for t in terms], 'plain_losses': [float(t) for t in plain], 'gradient_norm': float(norm),
            'logit_absmax': float(raw.detach().abs().max()), 'spikes': spikes.sum(0).cpu().tolist(),
            'events': len(raw)}
