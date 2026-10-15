# A 模型准确度

A01–A13 为事件级，A14–A22 为可观测事件轨迹指标。正式软件模型 H2 与 H2-pro 的 Validation/Test 使用相同首事件锚定的完整 100 ms 窗口前缀。H2 预测由旧全序列评分裁剪，仍含非因果时间输入；H2-pro 是流式因果推理。H2-plus 的历史比较仍在 `../../1_H2_model_improvement/seven_step_improvement_study/H2-pro/reports/AB_COMPARISON.md`，但不在正式注册表。

`01_metric_comparisons/` 保存逐指标 CSV；`02_model_records/formal/<evaluation_id>/<split>/` 保存事件级和轨迹级源结果。EV-SpSegNet 已正式注册，其论文值仅作 `paper_reported` 参照，同口径完整测量仍待补齐。

现行 [Pd/Fa v2](03_rules_and_protocols/localization_v2_protocol.md) 使用统一半开窗与曝光分母，旧 A06/A07 值保留为 legacy_only，尚待从冻结预测重算。已有 IoU 配对区间见 01_metric_comparisons/uncertainty；汇总、空值与不确定性定义见 [统计协议](03_rules_and_protocols/statistical_reporting_protocol.md)。
