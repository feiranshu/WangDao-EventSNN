# 模型评测统一入口

正式软件模型注册表在 `model_registry.csv`：H2、H2-pro、EV-SpSegNet。H2-plus/A4-2 已退出正式注册，原 A/B 源记录移至 `../1_H2_model_improvement/seven_step_improvement_study/step7/intermediate_H2-plus/`。第 5 板块只保留 A 模型准确度、B 模型复杂度、C 部署效率与硬件成本；各项比较见 `01_metric_comparisons/`，源记录见 `02_model_records/`，协议见 `03_rules_and_protocols/`。

现行规则为 [评价标准 v2](EVALUATION_STANDARD_V2.md)。软件要求 A+B（含新增 B05），板端要求 A+C01–C10。H2/H2-pro 历史准确度和 B01–B04 数值保留，但旧 Pd/Fa 标记 `legacy_only`，新版重算及 B05 清单待补，因此整体准入为 incomplete。EV-SpSegNet 仍待同协议补测，论文值为 paper_reported。

H2/H2-pro 的 A/B 正式横向比较采用首事件锚定的相同完整 100 ms 窗口前缀。H2 旧评分来自全序列预测后裁剪，保留旧非因果时间输入；H2-pro 为流式因果预测，因此因果性差异必须随指标一起阅读。B03/B04 只统计 Proposal 和条件 Refiner 的 exact-int8 MAC，不代表 CPU 延迟或硬件资源。历史的 8 秒全序列 H2/H2-plus 正式记录在 Step7 中间模型目录中保留。

C07 Runtime 继续记录完整序列运行时间，已升级为 5 次预注册重复、原始样本及证据核验的 [v2 协议](C_deployment_efficiency_and_hardware_cost/03_rules_and_protocols/runtime_measurement_protocol.md)。C08 内存峰值、C09 覆盖/积压为软件补充及板端必需，C10 为板端数值一致性；均保持真实待测状态。窗口级 CPU benchmark 不能直接填 C07。

`evaluation_coverage.csv` 核验源文件与证据哈希、数值和比较协议；paper_reported、pending、legacy_only、unverified 不算 complete，软件 C 的 not_required 不代表测量完成。完整性通过也不替代对真实执行/仪器的审查。
