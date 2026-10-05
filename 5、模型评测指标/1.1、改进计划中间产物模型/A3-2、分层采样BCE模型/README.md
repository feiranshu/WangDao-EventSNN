# A3-2 分层采样 BCE 模型指标

数据角色：Train-dev；seed：42；Validation/Test 未使用。正式比较使用 `int8/`，`float/`只用于量化差距诊断。

- 评测协议：`evuav-33e2aff-official-compatible-v1+step3-density-20000`
- 正式 checkpoint：`epoch_001.pt`
- int8 Event AP：0.9276145702
- int8 Event IoU：0.7827233762
- 结论：高密度 Recall 非劣性检查未通过，没有晋级。

完整训练与决策证据位于`../../../1、H2模型改进/seven_step_improvement_study/step3/results/runs/A3-2-seed42/`。
