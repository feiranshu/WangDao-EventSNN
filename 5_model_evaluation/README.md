# 模型评测统一入口

正式软件模型注册表在 `model_registry.csv`：H2、H2-pro、EV-SpSegNet。H2-plus/A4-2 已退出正式注册，原 A/B 源记录移至 `../1_H2_model_improvement/seven_step_improvement_study/step7/intermediate_H2-plus/`。第 5 板块只保留 A 模型准确度、B 模型复杂度、C 部署效率与硬件成本；各项比较见 `01_metric_comparisons/`，源记录见 `02_model_records/`，协议见 `03_rules_and_protocols/`。

软件正式模型要求 A Validation/Test 与 B Common/Validation/Test 完整；板端正式模型要求 A Validation/Test 与 C Common/Validation/Test 完整。H2/H2-pro A01–A22、B01–B04c 均完成，EV-SpSegNet 同口径 CUDA 补测仍未完成，论文值只标记 `paper_reported`；C 等待上板。

H2/H2-pro 的 A/B 正式横向比较采用首事件锚定的相同完整 100 ms 窗口前缀。H2 旧评分来自全序列预测后裁剪，保留旧非因果时间输入；H2-pro 为流式因果预测，因此因果性差异必须随指标一起阅读。B03/B04 只统计 Proposal 和条件 Refiner 的 exact-int8 MAC，不代表 CPU 延迟或硬件资源。历史的 8 秒全序列 H2/H2-plus 正式记录在 Step7 中间模型目录中保留。

`evaluation_coverage.csv` 是完整性检查；`paper_reported`、`pending` 不算 `complete`。跨模型比较只维护 CSV。
