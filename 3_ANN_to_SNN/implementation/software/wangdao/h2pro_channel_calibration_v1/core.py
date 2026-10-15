"""Original int8 synapses with separately accounted heterogeneous IF decoding."""
from dataclasses import asdict, dataclass
import numpy as np

from ..h2pro_direct_v1.core import (validate_layers, validate_steps, quantize_hidden,
                                  load_layers, probabilities)
from ..h2pro_calibration_v1.core import Encoder, decode as shared_decode


@dataclass(frozen=True)
class Channels:
    theta: tuple
    membrane: tuple

    def validate(self, width):
        if len(self.theta) != width or len(self.membrane) != width:
            raise ValueError('One threshold and initial membrane per presynaptic channel required')
        for t, m in zip(self.theta, self.membrane):
            Encoder(t, m).validate()

    def record(self):
        return {'theta': list(self.theta), 'membrane': list(self.membrane)}

    @classmethod
    def shared(cls, width, theta=255, membrane=127):
        return cls((theta,)*width, (membrane,)*width)

    @classmethod
    def read(cls, obj):
        return cls(tuple(obj['theta']), tuple(obj['membrane']))


def defaults():
    return (Channels.shared(64), Channels.shared(32))


def validate(encoders, steps):
    validate_steps(steps)
    if len(encoders) != 2:
        raise ValueError('Two hidden-layer encoders required')
    for width, e in zip((64, 32), encoders):
        e.validate(width)
        # Both GPU FP64 integer arithmetic and CPU int64 must be exact.
        if width*128*steps*max(e.theta) >= 2**53:
            raise ValueError('Weighted decoder exceeds exact integer accumulation range')


def counts(value, steps, encoder):
    if value.dtype != np.uint8 or value.ndim != 2:
        raise ValueError('Original uint8 hidden activations required')
    encoder.validate(value.shape[1]); validate_steps(steps)
    return np.minimum(steps, (value.astype(np.int64)*steps + encoder.membrane)//encoder.theta).astype(np.int32)


def decode_numerator(raw, scale, steps, encoder):
    unique = sorted(set(encoder.theta))
    if len(unique) == 1:
        # Preserve historical operation order, including its rounding, exactly.
        return shared_decode(raw, scale, steps, Encoder(unique[0], 0))
    return raw.astype(np.float64)*scale.astype(np.float64)[None, :]/steps/255.


def accumulate(value, weight, steps, encoder, backend):
    """No rescaled weight arrays. Explicit backend adds original int8 rows."""
    if (value.dtype != np.uint8 or value.ndim != 2 or weight.dtype != np.int8 or weight.ndim != 2
            or weight.shape[0] != value.shape[1] or backend not in ('count_reference', 'spike_add')):
        raise ValueError('Invalid frozen synapse contract')
    encoder.validate(value.shape[1]); validate_steps(steps)
    unique = sorted(set(encoder.theta)); n, width = value.shape
    if width*128*steps*max(unique) >= 2**53:
        raise ValueError('Decoder exceeds exact integer bound')
    theta = np.asarray(encoder.theta, np.int64)
    if backend == 'count_reference':
        c = counts(value, steps, encoder)
        raw = (c.astype(np.int64)*(theta if len(unique) > 1 else 1)) @ weight.astype(np.int64)
    else:
        membrane = np.broadcast_to(encoder.membrane, value.shape).astype(np.int64).copy()
        groups = [np.flatnonzero(theta == t) for t in unique]
        sums = [np.zeros((n, weight.shape[1]), np.int64) for _ in unique]
        c = np.zeros(value.shape, np.int32)
        for _ in range(steps):
            membrane += value
            fired = membrane >= theta
            c += fired
            for ids, bucket in zip(groups, sums):
                for channel in ids:
                    selected = np.flatnonzero(fired[:, channel])
                    bucket[selected] += weight[channel].astype(np.int64)
            membrane -= fired*theta
        raw = sums[0] if len(unique) == 1 else sum(bucket*t for bucket, t in zip(sums, unique))
    return raw, c


def cost(n, stage, steps, encoders, spike_sums, backend='count_reference'):
    """Neural inference only. Decoder multiplications are NOT hidden in AC."""
    validate(encoders, steps)
    if stage not in ('ann', 'r1', 'r2'):
        raise ValueError('Unknown conversion stage')
    active = 0 if stage == 'ann' else 1 if stage == 'r1' else 2
    fanouts = (32, 4)
    n1, n2 = map(int, spike_sums)
    out = {'refiner_equivalent_mac': n*(4480 if active == 0 else 2432 if active == 1 else 2304),
           'equivalent_ac': 32*n1+4*n2,
           'hidden1_spikes': n1, 'hidden2_spikes': n2,
           'executed_reference_refiner_mac': n*4480 if backend == 'count_reference' else 0,
           'executed_explicit_refiner_mac': n*(4480 if active == 0 else 2432 if active == 1 else 2304) if backend == 'spike_add' else 0,
           'executed_explicit_synaptic_ac': 32*n1+4*n2 if backend == 'spike_add' else 0,
           'base_decode_scale_mul': n*100, 'base_decode_divisions': n*100,
           'extra_threshold_decode_mul': 0, 'extra_threshold_decode_divisions': 0,
           'extra_threshold_bucket_adds': 0,
           'count_reference_threshold_mul': 0,
           'count_reference_step_mul': n*sum((64,32)[:active]) if backend=='count_reference' else 0,
           'count_reference_membrane_add': n*sum((64,32)[:active]) if backend=='count_reference' else 0,
           'count_reference_threshold_div': n*sum((64,32)[:active]) if backend=='count_reference' else 0,
           'equivalent_if_state_adds': n*sum((64,32)[:active])*steps,
           'equivalent_if_comparisons': n*sum((64,32)[:active])*steps,
           'equivalent_if_resets': n1+n2, 'hidden_quantizer_mul': n*96,
           'hidden_quantizer_round': n*96, 'threshold_buckets': [],
           'encoder_parameter_bytes_expanded_int32': 2*4*sum((64,32)[:active]),
           'peak_serial_if_state_elements_per_candidate': 64 if active else 0,
           'peak_decoder_bucket_elements_per_candidate': 0,
           'original_int8_weight_bytes': 4480, 'original_scale_bytes': 400,
           'physical_energy_measured': False, 'memory_traffic_measured': False}
    for i in range(active):
        u = len(set(encoders[i].theta)); fanout = fanouts[i]
        out['threshold_buckets'].append(u)
        out['peak_decoder_bucket_elements_per_candidate'] = max(out['peak_decoder_bucket_elements_per_candidate'], u*fanout)
        if u > 1:
            out['extra_threshold_decode_mul'] += n*u*fanout
            out['extra_threshold_decode_divisions'] += n*fanout
            out['extra_threshold_bucket_adds'] += n*(u-1)*fanout
            if backend == 'count_reference':
                out['count_reference_threshold_mul'] += n*(64,32)[i]
        elif encoders[i].theta[0] != 255:
            out['extra_threshold_decode_mul'] += n*fanout
            out['extra_threshold_decode_divisions'] += n*fanout
    return out


def forward(features, layers, stage='r2', steps=8, encoders=None, backend='count_reference'):
    encoders = defaults() if encoders is None else encoders
    validate(encoders, steps); validate_layers(layers)
    if stage not in ('ann','r1','r2') or backend not in ('count_reference','spike_add'):
        raise ValueError('Unknown stage/backend')
    if features.dtype != np.uint8 or features.ndim != 2 or features.shape[1] != 36:
        raise ValueError('Frozen original 36 uint8 features required')
    value, spikes = features, [0,0]
    for i, (w, s) in enumerate(layers):
        spiking = (i == 1 and stage in ('r1','r2')) or (i == 2 and stage == 'r2')
        if spiking:
            raw, c = accumulate(value, w, steps, encoders[i-1], backend)
            floating = decode_numerator(raw, s, steps, encoders[i-1])
            spikes[i-1] = int(c.sum(dtype=np.int64))
        else:
            raw = value.astype(np.int64) @ w.astype(np.int64)
            floating = raw.astype(np.float64)*s.astype(np.float64)[None,:]/255.
        if i < 2:
            value = quantize_hidden(floating)
        else:
            return floating.astype(np.float32), cost(len(value), stage, steps, encoders, spikes, backend)
    raise AssertionError('Missing unchanged readout')
