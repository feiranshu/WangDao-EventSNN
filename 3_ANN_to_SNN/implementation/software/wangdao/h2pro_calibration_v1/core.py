"""Calibrated binary IF transport; original integer weights are never rescaled."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..h2pro_direct_v1.core import (Cost as BaseCost, BACKENDS, STAGES, validate_layers,
    validate_steps, quantize_hidden, probabilities, load_layers)


@dataclass(frozen=True)
class Encoder:
    threshold: int = 255
    initial_membrane: int = 127

    def validate(self):
        if (type(self.threshold) is not int or not 1<=self.threshold<=np.iinfo(np.int32).max
                or type(self.initial_membrane) is not int or not 0<=self.initial_membrane<self.threshold):
            raise ValueError('Require integer theta>0 and 0<=initial membrane<theta')


@dataclass
class Cost(BaseCost):
    encoder_saturated_cells: int = 0
    encoder_cells: int = 0
    reference_saturation_comparisons: int = 0


def count_spikes(activation, steps, encoder):
    validate_steps(steps); encoder.validate()
    if activation.dtype!=np.uint8 or activation.ndim!=2:
        raise ValueError('Original uint8 activation matrix required')
    return np.minimum(steps,(activation.astype(np.int64)*steps+encoder.initial_membrane)//encoder.threshold).astype(np.int32)


def accumulate(activation, weight, steps, encoder, backend):
    validate_steps(steps); encoder.validate()
    if (activation.dtype!=np.uint8 or activation.ndim!=2 or weight.dtype!=np.int8 or
            weight.ndim!=2 or activation.shape[1]!=weight.shape[0] or activation.shape[1]>64
            or backend not in BACKENDS):
        raise ValueError('Invalid calibrated IF transport')
    rows,width=activation.shape; fanout=weight.shape[1]
    cost=Cost(encoder_saturated_cells=int(np.count_nonzero(activation>encoder.threshold)),
              encoder_cells=activation.size)
    if backend=='count_reference':
        counts=count_spikes(activation,steps,encoder)
        raw=counts@weight.astype(np.int32)
        spikes=int(counts.sum(dtype=np.int64))
        cost.executed_synaptic_mac=rows*weight.size
        cost.count_reference_integer_mul=activation.size
        cost.count_reference_integer_add=activation.size
        cost.count_reference_integer_div=activation.size
        cost.reference_saturation_comparisons=activation.size
    else:
        # int64 also covers saturated residual charge at the validated maximum T.
        current=activation.astype(np.int64)
        membrane=np.full(activation.shape,encoder.initial_membrane,dtype=np.int64)
        raw=np.zeros((rows,fanout),np.int32); spikes=0
        for _ in range(steps):
            membrane+=current
            fired=membrane>=encoder.threshold
            for channel in np.flatnonzero(fired.any(axis=0)):
                selected=np.flatnonzero(fired[:,channel])
                raw[selected]+=weight[channel].astype(np.int32)
            count=int(fired.sum()); spikes+=count
            membrane[fired]-=encoder.threshold
        cost.executed_synaptic_ac=spikes*fanout
        cost.if_state_adds=rows*width*steps
        cost.if_comparisons=rows*width*steps
        cost.if_reset_subtractions=spikes
    cost.rate_network_ac=spikes*fanout; cost.unit_spikes=spikes
    return raw,cost


def decode(raw, scale, steps, encoder):
    floating=raw.astype(np.float64)*scale[None,:].astype(np.float64)/steps
    if encoder.threshold!=255:
        floating=floating*encoder.threshold/255.
    return floating


def forward(features,layers,stage='r2',steps=16,encoders=(Encoder(),Encoder()),backend='spike_add'):
    validate_steps(steps); validate_layers(layers)
    if (stage not in STAGES or backend not in BACKENDS or len(encoders)!=2
            or features.dtype!=np.uint8 or features.ndim!=2 or features.shape[1]!=36):
        raise ValueError('Retain original features, layers and known conversion stages')
    for e in encoders:e.validate()
    value=features; cost=Cost()
    for index,(weight,scale) in enumerate(layers):
        spiking=(index==1 and stage in ('r1','r2')) or (index==2 and stage=='r2')
        if spiking:
            e=encoders[index-1]
            raw,transport=accumulate(value,weight,steps,e,backend)
            cost.merge(transport)
            floating=decode(raw,scale,steps,e)
            if e.threshold!=255:
                cost.scalar_scale_mul+=floating.size
                cost.scalar_normalization_div+=floating.size
        else:
            raw=value.astype(np.int32)@weight.astype(np.int32)
            cost.executed_synaptic_mac+=len(value)*weight.size
            cost.rate_network_mac+=len(value)*weight.size
            floating=raw.astype(np.float64)*scale[None,:].astype(np.float64)/255.
        cost.scalar_scale_mul+=floating.size; cost.scalar_normalization_div+=floating.size
        if index<2:
            value=quantize_hidden(floating)
            cost.quantizer_scale_mul+=value.size; cost.quantizer_round+=value.size
            cost.quantizer_comparisons+=4*value.size
        else:
            return floating.astype(np.float32),cost
    raise AssertionError('Missing original readout')


def margin(raw):
    logits=raw.reshape(-1,2,2).max(axis=2).astype(np.float64)
    return logits[:,1]-logits[:,0]
