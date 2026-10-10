# H2-pro 冻结结果

本目录只交付正式 H2-pro 的源摘要与部署决定：`model_profile.json` 保存 B01/B02，`validation/` 与 `test/` 各自保存事件、轨迹和 B03/B04 摘要。所有值来自模型及阈值冻结后的同一闭合 100 ms 前缀。三模型 A01–A22/B01–B04 比较及 IoU bootstrap 保存在 `../../../seven_step_improvement_study/H2-pro/reports/`，H2-plus 只作为该研究对照出现。

正式第 5 板块只登记 H2 与 H2-pro。H2 和 H2-plus 的旧预测由全序列评分裁剪，旧非因果时间输入未消除；H2-pro 是流式闭合窗口预测。B03/B04 只覆盖 Proposal 与条件 Refiner exact-int8 MAC，不包含特征构建、后处理、CPU 延迟或内存工作区。正式决策与限制见 `deployment_decision.json`。
