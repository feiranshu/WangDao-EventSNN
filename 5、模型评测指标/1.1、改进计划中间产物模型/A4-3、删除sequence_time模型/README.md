# A4-3 删除 sequence_time 模型指标

删除非因果 `sequence_time` 后，以31维输入重新训练 H2 Refiner。数据角色为 Train-dev，seed 42，Validation/Test 未使用。

- 统一协议：`evuav-33e2aff-official-compatible-v1+step3-density-20000`
- 正式数值路径：`int8/exact_int8_replay`
- 正式选择点：`epoch_010.pt`
- int8 Event AP：0.9264221784；Event IoU：0.7754747022。
- 结论：仅保留为因果性诊断，未通过与32维重训对照的配对入口。

完整训练与决策证据位于`../../../1、H2模型改进/seven_step_improvement_study/step4/results/step4_formal_20261004/runs/A4-3-seed42/`。
