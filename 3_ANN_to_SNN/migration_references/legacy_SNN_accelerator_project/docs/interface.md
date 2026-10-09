# Frozen PS–PL interface v1.0

## AER stream

| Bits | Field | Constraint |
|---|---|---|
| 1:0 | type | event=0, slice-start=1, slice-end=2, reset=3 |
| 10:2 | x | 0–319 |
| 18:11 | y | 0–255 |
| 19 | polarity | 0 or 1 |
| 23:20 | time bin | 0–15 |
| 39:24 | slice ID | wraps modulo 65536 |
| 63:40 | reserved | must be zero |

Only a slice-end packet asserts `TLAST`. A producer must hold `TDATA/TLAST` while `TVALID=1` and `TREADY=0`.

## Feature stream

P3/P4/P5 tensors are signed INT8 in row-major `[C,H,W]` order and packed eight values per 64-bit beat. Spatial shapes are P3=`32×40`, P4=`16×20`, P5=`8×10`; channel counts and layer scales come from `model_manifest.json`. The first beat carries the slice ID in `TUSER[15:0]`, and the last beat of each tensor asserts `TLAST`.

## AXI-Lite register map

| Offset | Name | Access | Meaning |
|---:|---|---|---|
| 0x00 | CONTROL | R/W | mode[1:0], soft-reset pulse[2], clear-errors pulse[3], enable[4] |
| 0x04 | STATUS | R | busy, FIFO overflow, illegal packet, approximate active |
| 0x08 | DENSITY_LOW | R/W | auto-mode low threshold |
| 0x0C | DENSITY_HIGH | R/W | auto-mode high threshold |
| 0x10/14 | EVENT_COUNT | R | accepted event packets |
| 0x18/1C | CYCLE_COUNT | R | enabled accelerator cycles |
| 0x20/24 | CACHE_HITS | R | state-cache hits |
| 0x28/2C | EXTERNAL_READS | R | state-memory reads |
| 0x30/34 | SATURATIONS | R | arithmetic saturation events |

The firmware reads 64-bit counters high/low/high to avoid torn values.

