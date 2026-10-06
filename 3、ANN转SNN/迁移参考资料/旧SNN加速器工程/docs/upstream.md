# Upstream integration

External source and dataset licenses must remain intact. Upstream repositories are intentionally not vendored.

```powershell
./scripts/bootstrap_upstream.ps1 -IncludeDatasetCode
```

This fetches:

- `third_party/EMS-YOLO` from `https://github.com/BICLab/EMS-YOLO`;
- `third_party/Ev-UAV` from `https://github.com/ChenYichen9527/Ev-UAV`.

Record the printed commits in an experiment manifest before modifying adapters. Do not commit EV-UAV data or third-party checkpoints.

The EMS-YOLO adapter must:

1. preserve upstream model code and add a local wrapper rather than patching it silently;
2. expose `hardware_forward(inputs, hw_cfg)` for the repository's `quantized_forward` entry point;
3. return P3/P4/P5 tensors and the membrane features used by distillation;
4. replace supported weights through `power_of_two_ste` during QAT;
5. use `HardwareAwareLIFCell` or prove bit equivalence to it;
6. export weights only through `export_model`.

Dataset placement defaults to `data/EV-UAV/{train,val,test}`. Generate and save SHA-256 checksums before the first training run.

The slicer uses microseconds internally. Confirm the timestamp unit of the pinned EV-UAV revision and pass `--timestamp-unit ns|us|ms|s` explicitly during preprocessing; never tune time-window values against an undocumented raw unit.
