# A4-2最终交付结果

本目录是从Step5冻结接口和Step7正式回传结果中提取的精简交付副本。完整训练、全阈值曲线、bootstrap事件分数和服务器日志仍保存在`../../../seven_step_improvement_study/`，这里不重复存放大体积可追溯产物。

## 目录结构

```text
final_A4-2/
├─ README.md                         本目录说明
├─ deployment_decision.json         用户确认的最终部署决策及指标取舍
├─ frozen_candidate_step5.json      Step5冻结模型、epoch、阈值和权重哈希
├─ step7_protocol.json              打开独立数据前冻结的Step7协议
├─ step7_preflight.json             环境、数据索引和模型哈希检查
├─ step7_decision.json              原始Step7门槛判定，不因部署决策而改写
├─ validation/
│  ├─ split_summary.json            Validation完整固定工作点指标和门槛
│  ├─ comparison.csv/.svg           H2-F、H2-R、A4-2对比表与图
│  ├─ pr_comparison.svg             PR曲线叠加图
│  ├─ per_sequence_comparison.csv   24条序列逐项比较
│  └─ paired_bootstrap.json         10,000次配对序列bootstrap
└─ test/                            与validation同结构的独立Test结果
```

`step7_decision.json`保留“高密度Recall保护线未通过”的原始科学判断；`deployment_decision.json`记录用户随后基于整体IoU、Precision和Fa优先级作出的工程部署选择。两个文件用途不同，均不得删除或互相覆盖。

