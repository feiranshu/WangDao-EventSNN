# MLP 转 SNN 与近似计算

本部分是模型侧唯一主线，负责从冻结 ANN 教师到 SNN 的转换、训练、定点化、PoT、LOA 近似计算、软件黄金模型，以及围绕这些变体进行优化、消融和选型。KV260 工程、RTL、位流、板端驱动与实测步骤统一放在 `../4_model_deployment/`。

## 当前内容

- `implementation/`：可运行的第一代 SNN 模型工程。
- [`model_selection_metrics/`](model_selection_metrics/README.md)：S01 时间步—效果—成本曲线与 S02 转换/数值近似效果损失；它们由原模型评测 D 类迁入，只用于优化、消融和选型。
- `migration_references/`：早期 SNN 加速器与独立验证工程，只作参考。

拆分前历史、被新实现替换的文件和研究材料统一保存在根级本地目录`../archive/3_ANN_to_SNN/`，不属于 GitHub 发布内容。

第一代模型覆盖 S0-P、S0-R、S1 定点、S2 混合 PoT、S3 LOA，以及 STC、Refiner 适配、独立时空 SNN、读出方式和数据加载优化的正负实验。正式代码、冻结权重和软件结果位于 `implementation/`；迁移与替换记录见 `implementation/MIGRATION_MANIFEST.md`。

## 与正式评测和上板的边界

S01/S02 是研究决策指标，不进入 `5_model_evaluation` 的正式分指标比较。软件筛选通过的正式模型必须完成 A+B；实际部署到开发板的正式配置使用独立 `evaluation_id`，必须完成 A+C。

模型结构、权重、阈值、时间步、定点格式和近似规则在这里冻结；上板部分只消费冻结契约，不在板级调试时反向修改模型。第一代 KV260 部署入口见 `../4_model_deployment/01_first_generation_baseline/README.md`。
