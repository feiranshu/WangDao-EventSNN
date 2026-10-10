# A+B 完整性核查（2026-10-10）

正式软件模型注册表包含 H2、H2-pro 与 EV-SpSegNet。重建后 H2 与 H2-pro 的 A01–A22 Validation/Test、B01/B02 Common、B03/B04 Validation/Test 均为 `complete`；`evaluation_coverage.csv` 两者为 `complete`。EV-SpSegNet 仍为 `incomplete`，论文值保留 `paper_reported`，待官方同口径 CUDA 补测。

H2/H2-pro 的正式比较采用相同首事件锚定的闭合 100 ms 窗口前缀。H2 历史评分来自全序列推理后裁剪，包含旧非因果时间维度；H2-pro 是因果流式推理。原 2026-10-09 的 H2/H2-plus 完整性核查仅作为当时注册表的历史审计，不再描述现行模型名单。

复核命令：

```powershell
python 1_H2_model_improvement/implementation/software/scripts/test_evaluation_layout.py
```
