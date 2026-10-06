# Execution roadmap

## Definition of done by milestone

1. **Baseline freeze — 2026-08-23:** environment, data checksum, one-command train/eval, reproduction within ±0.02 mAP@0.5.
2. **Teacher and slicing — 2026-09-20:** three seeds, fixed/dynamic comparison, frozen detection protocol.
3. **Hardware-aware model — 2026-10-18:** INT8, power-of-two and approximate modes, manifest and golden vectors.
4. **Arithmetic RTL — 2026-11-15:** bit-exact LIF tests and first XCZU3EG utilization/timing report.
5. **Ingress/cache RTL — 2026-12-20:** event-loss tests, cache counters and post-route timing.
6. **PS–PL integration — 2027-01-24:** continuous EV-UAV replay through DMA with P3/P4/P5 return and PS detection.
7. **Evidence freeze — 2027-02-21:** complete ablation, power, throughput and robustness tables.
8. **Release — 2027-03-20:** paper/report, demo, code release, checksums and financial log.

## Weekly control

- Monday: interface/risk review; no interface change without a version bump and compatibility note.
- Friday: runnable demonstration and archived result summary.
- A milestone failing for two consecutive weeks triggers the documented scope reduction before work starts on optional attention or live-camera support.

## Ownership

- 束斐然：data adapter, training, QAT/KD, export and experiment automation.
- 郑鸿凯：bit-level review, RTL, verification, Vivado implementation and method writing.
- Both: PS–PL integration, board measurement, artifact release and final review.

