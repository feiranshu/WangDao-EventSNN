# A4-2-null 历史零输入对照指标

这是 A4-2 的 36 维匹配零输入对照，用于判断四维历史输入是否产生真实收益，不是独立部署候选。

- 数据角色：Train-dev。
- 协议：`evuav-33e2aff-official-compatible-v1+step3-density-20000`。
- `float/`、`int8/`：保留 Step 4 原选择点 seed42、epoch 0、阈值 0.30，供历史追溯和总 PR 图使用。
- `stability/seed43/`、`stability/seed44/`：Step 6 固定 epoch 8、阈值 0.40 的匹配对照完整结果。
- seed42 在固定 epoch 8、阈值 0.40 下的 AP 为 0.9277020956、IoU 为 0.7851615840；正式归因以 Step 6 决策记录为准。
- 结论：三个随机种子下，A4-2 的 AP 与 IoU 均高于各自匹配 null，支持历史特征确有增益。

来源：`../../../1、H2模型改进/seven_step_improvement_study/step4/` 与 `step6/` 的正式结果。seed42 固定点未单独导出完整 PR 文件，因此未用不匹配的 PR 曲线替代它。
