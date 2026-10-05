# A4-1 密度特征模型指标

32维 H2 Refiner 输入增加一个当前窗口因果密度特征。数据角色为 Train-dev，seed 42，Validation/Test 未使用。

- 统一协议：`evuav-33e2aff-official-compatible-v1+step3-density-20000`
- 正式数值路径：`int8/exact_int8_replay`
- 正式选择点：`epoch_000.pt`
- 结论：与 H2 数值相同，配对入口未通过，不能宣称密度特征带来收益。

完整训练与决策证据位于`../../../1、H2模型改进/seven_step_improvement_study/step4/results/step4_formal_20261004/runs/A4-1-seed42/`。
