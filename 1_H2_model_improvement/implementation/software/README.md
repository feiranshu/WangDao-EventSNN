# 软件目录

## 目录结构

```text
software/
├─ README.md                         本说明
├─ scripts/                          命令行入口；详见scripts/README.md
├─ tests/                            单元测试；详见tests/README.md
├─ wangdao/                          Python包；详见wangdao/README.md
└─ weights/                          冻结权重；详见weights/README.md
```

正式部署最小链路是`scripts/run_a4_2.py`、`wangdao/a4_2_runtime.py`、`wangdao/h2/`、原H2 Proposal权重和A4-2 int8 Refiner权重。其余训练、诊断、学生模型和发布代码作为复现与后续研究工具保留。

