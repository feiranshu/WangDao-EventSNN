"""Bit-exact original r2/T8 forward, with straight-through training derivatives.

Integer weights and uint8 activations make all dot products exactly representable
in FP64. Decode order matches the frozen NumPy contract. The dense count backend
is a training/reference implementation, not sparse AC execution on the GPU.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from typing import cast

from ..h2pro_direct_v1.core import load_layers, validate_layers


class RoundSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return torch.round(value)

    @staticmethod
    def backward(ctx, gradient):
        return gradient


class CountSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, activation):
        return torch.floor((activation*8 + 127)/255)

    @staticmethod
    def backward(ctx, gradient):
        return gradient*(8/255)


def class_logits(raw):
    return raw.reshape(-1, 2, 2).max(dim=2).values


class R2T8(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        layers = load_layers() if layers is None else layers
        validate_layers(layers)
        self.weights = nn.ParameterList()
        for i, (w, s) in enumerate(layers):
            self.register_buffer(f'scale{i}', torch.from_numpy(s.copy()))
            # Independent master copy; the teacher export is never a parameter.
            self.weights.append(nn.Parameter(torch.from_numpy(w.astype(np.float32)*s[None, :])))

    def integer_weight(self, index):
        scale = getattr(self, f'scale{index}').double()
        # Round before clipping: identical integer forward, but a float master
        # that rounds to 127 still receives a gradient inside that valid bin.
        return cast(torch.Tensor, RoundSTE.apply(self.weights[index].double()/scale)).clamp(-128, 127)

    def forward(self, features):
        if features.dtype != torch.uint8 or features.ndim != 2 or features.shape[1] != 36:
            raise ValueError('Original 36-column uint8 candidate features required')
        value = features.double()
        counts = []
        for i in range(3):
            if i:
                value = cast(torch.Tensor, CountSTE.apply(value))
                counts.append(value.detach().sum(dim=1))
            raw = value @ self.integer_weight(i)
            value = raw*getattr(self, f'scale{i}').double()/(8 if i else 255)
            if i < 2:
                value = cast(torch.Tensor, RoundSTE.apply(value.clamp(0, 1)*255))
        return value.float(), torch.stack(counts, dim=1).to(torch.int64)

    @torch.no_grad()
    def export_layers(self):
        return tuple((self.integer_weight(i).to(torch.int8).cpu().numpy().copy(),
                      getattr(self, f'scale{i}').cpu().numpy().copy()) for i in range(3))


def losses(raw, labels, teacher_raw, config):
    logits = class_logits(raw)
    teacher = class_logits(teacher_raw.detach())
    supervised = F.cross_entropy(logits, labels.long())
    temperature = config.temperature
    distillation = F.kl_div(F.log_softmax(logits/temperature, dim=1),
                           F.softmax(teacher/temperature, dim=1), reduction='batchmean')*temperature**2
    total = (1-config.kd_alpha)*supervised + config.kd_alpha*distillation
    return total, supervised, distillation


def require_finite(values, where):
    values = list(values)
    if values and not bool(torch.stack([torch.isfinite(v).all() for v in values]).all()):
        raise FloatingPointError('Nonfinite '+where+'; resume from the last committed checkpoint')


def update(model, optimizer, batch, config):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    raw, spikes = model(batch['features'])
    terms = losses(raw, batch['labels'], batch['teacher'], config)
    require_finite([raw, *terms], 'outputs/loss')
    terms[0].backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    if len(gradients) != 3:
        raise RuntimeError('Every original weight matrix must have a gradient')
    require_finite(gradients, 'gradients')
    norm = torch.linalg.vector_norm(torch.stack([g.double().norm() for g in gradients]))
    optimizer.step()
    require_finite(model.parameters(), 'student weights')
    return {'losses': [float(v.detach()) for v in terms], 'gradient_norm': float(norm),
            'logit_absmax': float(raw.detach().abs().max()),
            'spikes': spikes.sum(dim=0).cpu().tolist(), 'events': len(raw)}
