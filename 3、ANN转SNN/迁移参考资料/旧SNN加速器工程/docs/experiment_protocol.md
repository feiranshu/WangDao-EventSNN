# Experiment and acceptance protocol

## Reproducibility

- Record repository commit, upstream commits, environment lock, GPU, board serial/revision, Vivado version and data checksums.
- Use seeds 17, 29 and 43 for all software comparisons.
- Select checkpoints and dynamic thresholds on validation data only. Evaluate official test data after the configuration is frozen.
- Keep identical data order, input representation and training budget when comparing B1–B6.

## Baselines

| ID | Configuration |
|---|---|
| B0 | Reported 10-epoch result |
| B1 | Stabilized floating-point teacher |
| B2 | INT8, exact accumulator |
| B3 | Signed power-of-two, exact accumulator |
| B4 | Signed power-of-two, static approximate accumulator |
| B5 | Signed power-of-two, density-controlled dynamic precision |
| B6 | B5 plus 3×3 state-tile prefetch/cache |

Run the slicing ablation independently: fixed-duration versus bounded-count using P25/P50/P75 event counts calculated from training windows.

## Software metrics

Report mAP@0.5, mAP@0.5:0.95, precision, recall, F1 and mean±standard deviation across seeds. Include strata for target size, illumination and noise if those official metadata are available; do not infer unprovided labels.

Minimum gates:

- B0 reproduction within ±0.02 mAP@0.5 of 0.446;
- B5/B6 no more than 0.02 absolute mAP@0.5 and 0.03 mAP@0.5:0.95 below B1;
- recall at comparable precision at least 0.64.

## RTL and board metrics

- Exact RTL must match every golden vector; approximate RTL must match its declared arithmetic exactly.
- Exercise reset, empty slices, maximum coordinates, backpressure, saturation and malformed packets.
- Run a 30-minute trace without deadlock or silent event loss.
- Report post-route Fmax, LUT, FF, BRAM and DSP separately for the PE and infrastructure.
- Measure identical traces ten times and report mean, standard deviation and P95 latency.
- Board power is idle-subtracted. Store raw meter/sensor readings and the script used to integrate energy.

Minimum gates are 150 MHz, zero DSP in the synaptic datapath, 1.2× test-set P99 event rate, 20% fewer external state accesses with cache, and 10% lower net energy/event in approximate mode.

## Evidence rules

Every paper table must be generated from committed CSV/JSON summaries. Hand-edited result cells, cherry-picked seeds and Vivado estimates presented as measured board power are prohibited.

