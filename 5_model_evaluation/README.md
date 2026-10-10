# 模型评测统一入口

正式软件模型注册表在 `model_registry.csv`：H2、H2-pro、EV-SpSegNet。H2-plus/A4-2 已退出正式注册，原 A/B 源记录移至 `../1_H2_model_improvement/seven_step_improvement_study/step7/intermediate_H2-plus/`。第 5 板块只保留 A 模型准确度、B 模型复杂度、C 部署效率与硬件成本；各项比较见 `01_metric_comparisons/`，源记录见 `02_model_records/`，协议见 `03_rules_and_protocols/`。

软件正式模型要求 A Validation/Test 与 B Common/Validation/Test 完整；板端正式模型要求 A Validation/Test 与 C Common/Validation/Test 完整。H2/H2-pro A01–A22、B01–B04c 均完成，EV-SpSegNet 同口径 CUDA 补测仍未完成，论文值只标记 `paper_reported`；C01–C06 等待上板。

H2/H2-pro 的 A/B 正式横向比较采用首事件锚定的相同完整 100 ms 窗口前缀。H2 旧评分来自全序列预测后裁剪，保留旧非因果时间输入；H2-pro 为流式因果预测，因此因果性差异必须随指标一起阅读。B03/B04 只统计 Proposal 和条件 Refiner 的 exact-int8 MAC，不代表 CPU 延迟或硬件资源。历史的 8 秒全序列 H2/H2-plus 正式记录在 Step7 中间模型目录中保留。

C07 Runtime 记录完整序列输入的模型推理运行时间（`ms/sequence`），同时接受软件与板端实测；目前作为软件补充项，不改变软件 A+B 准入规则。定义及待测表位于 [C07_runtime](C_deployment_efficiency_and_hardware_cost/01_metric_comparisons/C07_runtime/README.md)，已按当前注册表保留 H2、H2-pro、EV-SpSegNet 的 Validation/Test 待测行，尚无同口径实测值。H2-pro 的闭合窗口与尾段处理须按 [runtime 协议](C_deployment_efficiency_and_hardware_cost/03_rules_and_protocols/runtime_measurement_protocol.md) 记录；研究中的窗口级 CPU benchmark 不直接填入 C07。

`evaluation_coverage.csv` 是完整性检查；`paper_reported`、`pending` 不算 `complete`，软件 C 的 `not_required` 也不代表 runtime 已测量。跨模型比较只维护 CSV。
