# H2-pro: causal H2-plus Refiner research

This Stage 1 study used the frozen H2 Proposal and historical A4-2 weights as its
starting point. The frozen C1 result is now deployed from `../../implementation/`; A4-2
is retained in the Step7 intermediate-model directory. It replaces Refiner input column 2, formerly full-sequence relative time,
with C0 zero, C1 historical Proposal motion, or C2 local motion-compensated event
coherence. All variants retain 36 uint8 inputs and the 36→64→32→4 Refiner.

`src.stream_state.CausalPipeline` is the inference API. Call `reset(t0)` for each
independent sequence, `push(events, watermark)` with monotonic event packets and
a watermark strictly beyond every packet event, then `flush_completed_windows()`.
The result gives original global event indices, candidate mask, candidate features,
full-window probabilities, predictions and right-boundary time. `finalize()`
returns closed windows and the count of unpublished events in the trailing partial
window, then closes the stream until the next `reset`. Watermarks close empty
windows; packets earlier than the last watermark are rejected. The first event
time is the fixed `t0` used for offline EV-UAV replay.
For ANN→SNN teacher work, use each result's `features` array as the sole 36-column
`uint8` input. `run_stage1.load_trained_float(path)` loads a trained float teacher;
`CausalPipeline(..., initial_refiner=False)` loads its exact-int8 counterpart.

The C2 formula and all training choices are fixed in `protocol.json`. The C2 score
uses merged-polarity 4×4-cell counts from the two preceding 25 ms bins and a 3×3
velocity search. It is a candidate hypothesis, not a claimed accuracy result.
The old four-scale count definitions, current Proposal context and four historical
Proposal columns are preserved. The model loaded from the frozen A4-2 exact-int8
archive has its first-layer column 2 zeroed and is an **untrained initialization**.

The `run_stage1.py` entry point checks source hashes, prepares role-specific
candidate caches and runs matched C0/C1/C2 float training with exact-int8 exports.
After promotion, its historical A4-2 initialization weights live under
`../step7/intermediate_H2-plus/weights/`.
`python verify_migrated_lock.py` checks the original 13 acceptance-lock hashes
after rebasing the former absolute research and A4-2 weight paths; the lock itself is unchanged.
It refuses to read Validation, Test or hardware-calibration roles. All three
10-epoch runs completed locally with seed 42; no server run is required.
Run `python -m unittest discover -s tests -v` for causal and compatibility tests.
`benchmark.py` records Train-main Python compute time and retained NumPy state;
these measurements do not represent ASIC timing or SRAM.

## Current status

The development result is frozen as `frozen/H2-pro-Causal-ANN-v1`: C1 Proposal
motion, epoch 10, exact-int8, threshold 0.35. C2-LMEC did not pass the development
selection guards. The v2 selection guard was amended after seeing v1 Train-dev
results and before opening either reserved role; it is an exploratory decision.
Validation and Test were each evaluated once after the acceptance lock. The Test
IoU improvement is small and its paired sequence bootstrap interval includes zero.
H2-pro is now the formal final version by the user's causal real-time deployment decision; A4-2/H2-plus has moved to the Step7 intermediate-model archive. See
`reports/FINAL_REPORT.md` for the comparison, limitations and exact commands.
For a three-model Validation/Test A01–A22 and B01–B04 comparison on identical
closed-window event prefixes, see `reports/AB_COMPARISON.md` and the accompanying
`AB_ALL_METRICS_SAME_PREFIX.csv`. It is a post-freeze read-only comparison and
preserves the earlier deployment decision as historical evidence while the formal model registry uses H2-pro.
