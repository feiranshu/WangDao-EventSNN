# Architecture specification

## Trust boundary

The Python reference model owns the numerical contract. RTL and firmware must match the exported manifest and golden vectors; neither may infer quantization rules from a checkpoint filename or training configuration.

## Data flow

1. PS loads an official EV-UAV sample or a live-compatible event trace.
2. The slicer creates boundaries using timestamps and event counts only.
3. Targets are generated after boundaries are frozen.
4. PS emits one slice-start packet, zero or more event packets, and one slice-end packet with `TLAST=1`.
5. The PL input FIFO applies AXI backpressure. Overflow and malformed packets set sticky status bits.
6. The event router updates state through time-multiplexed LIF processing elements.
7. Signed power-of-two weights are stored as one byte: zero flag, sign, two reserved zeros and a biased four-bit exponent.
8. P3/P4/P5 INT8 tensors return to PS. The detection head and NMS remain software in the first complete system.

The checked-in `snn_accel_core` is a synthesizable reference slice containing the ingress and arithmetic control path. Replicating/time-multiplexing it into the layer scheduler is the next hardware milestone; it is deliberately not presented as a completed EMS-YOLO backbone.

## Arithmetic

Exact mode computes a signed saturating sum. Approximate mode:

```text
low  = a[K-1:0] OR b[K-1:0]
high = (a >>> K) + (b >>> K)       # no carry from low field
raw  = (high <<< K) OR low
sum  = saturate(raw, WIDTH)
```

The LIF state applies `decayed = membrane - (membrane >>> leak_shift)`, adds the synaptic value, emits a spike when the threshold is reached and resets to zero.

## Dynamic precision

Manual exact and approximate modes override all automatic behavior. Auto mode counts accepted event packets in a slice and selects the next slice:

- `count >= density_high`: exact;
- `count <= density_low`: approximate;
- otherwise retain the previous mode.

This hysteresis prevents mode chatter. Thresholds are selected on the validation set under the declared accuracy constraint.

## State cache

Weights are spatially shared and are not the subject of the 3×3 locality hypothesis. The cache stores membrane/intermediate state tiles. `state_tile_cache.sv` supplies a direct-mapped synthesizable primitive and counters; `tile_prefetch_3x3.sv` emits bounded neighboring tile coordinates. The layer scheduler must connect misses to PL DDR and count every external transaction.

