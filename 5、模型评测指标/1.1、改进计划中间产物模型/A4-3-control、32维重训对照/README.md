# A4-3-control 32维重训对照指标

这是 A4-3 的32维 H2 匹配重训对照，用于区分删除 `sequence_time` 与普通重训带来的变化，不是独立晋级编号。

- 数据角色：Train-dev；seed 42；Validation/Test 未使用。
- 统一协议：`evuav-33e2aff-official-compatible-v1+step3-density-20000`
- 正式选择点：`epoch_001.pt`
- 结论：对照本身非劣，但 A4-3 配对入口未通过，不产生替代模型。

完整证据位于`../../../1、H2模型改进/seven_step_improvement_study/step4/results/step4_formal_20261004/runs/A4-3-control-seed42/`。
