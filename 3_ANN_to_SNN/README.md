# ANN→SNN 转换与近似计算

本目录负责冻结 ANN 的 SNN 转换、数值近似、软件黄金模型，以及模型优化、消融和选型。ANN 改进见[目录 1](../1_H2_model_improvement/)，RTL 与板级部署见[目录 4](../4_model_deployment/)。

**当前实现状态、各版本结果、操作命令、有效计划和变更记录统一维护在 [implementation/README.md](implementation/README.md)。**

| 入口 | 用途 |
| --- | --- |
| [implementation/README.md](implementation/README.md) | 唯一实现状态文档 |
| [implementation/AGENTS.md](implementation/AGENTS.md) | 长期协作规则 |
| [implementation/software](implementation/software/) | 模型、数值参考、运行脚本、权重和测试 |
| [implementation/results](implementation/results/) | 原始实验报告；本地 runs 保存训练日志、缓存与断点 |
| [model_selection_metrics/README.md](model_selection_metrics/README.md) | S01/S02 指标定义与使用规则 |
| [migration_references](migration_references/) | 工程来源与迁移参考材料 |
| [正式评测目录](../5_model_evaluation/README.md) | 入选模型的 A/B/C 评测规则与证据 |

原 PLAN.md 与 implementation/MIGRATION_MANIFEST.md 的有效内容已合并到实现 README；历史文档保存在仓库根级[本地归档](../archive/3_ANN_to_SNN/document_consolidation_20261014/README.md)，不再作为当前操作依据。归档、运行日志和缓存遵守已有 Git 忽略规则，可能只在原机器存在。
