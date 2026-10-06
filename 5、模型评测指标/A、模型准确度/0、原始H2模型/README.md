# 原始 H2 模型评测

本目录保留两部分 H2 结果：

- `历史Step1结果/`：Step 1 的原始历史输出。完整 10 条 Train-dev 的 Event AP 为 0.927411，精确 PR 点数为 357,225。
- `统一int8评测/`：与 H2-plus 使用相同冻结索引、相同指标协议和 `exact_int8_replay` 数值路径的正式比较基线，包含 Train-dev、Validation 和 Test。

历史图表中，`H2_PR_curve.png` 为 PR 曲线，`H2_F1_threshold.png` 为阈值—F1 曲线，`pr_curve_exact.npz` 保存全部精确点，`pr_curve_plot.csv` 为绘图抽样点。H2-F 与 H2-R 是同一模型的不同工作点；当前正式对比使用阈值 0.30 的 H2 基线。
