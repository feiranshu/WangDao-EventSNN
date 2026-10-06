# MLP 转 SNN 与近似计算

本部分是模型侧唯一主线，负责从冻结 ANN 教师到 SNN 的转换、训练、定点化、PoT、LOA 近似计算和软件黄金模型。KV260 工程、RTL、位流、板端驱动与实测步骤统一放在 `../4、模型上板/`。

## 当前内容

- `implementation/`：可运行的第一代 SNN 模型工程。
- `archive/`：拆分前历史、被新实现替换的文件和研究材料。
- `迁移参考资料/`：早期 SNN 加速器与独立验证工程，只作参考。

第一代模型已覆盖：

- S0-P：浮点 Proposal-SNN + ANN Refiner；
- S0-R：ANN Proposal + 浮点 Refiner-SNN；
- S1：int8 权重、24 位累加器和 24 位 Q12 膜电位；
- S2：混合 PoT 消融；
- S3：LOA 近似加法；
- STC、Refiner 适配、独立时空 SNN、读出方式和数据加载优化的正负实验。

正式模型代码、冻结权重和软件结果均在 `implementation/`。同学完成的第一代工程已于 2026-10-03 合并，文件归属和替换记录见 `implementation/MIGRATION_MANIFEST.md`。

## 与上板部分的边界

模型结构、权重、阈值、定点格式和近似规则在这里冻结；上板部分只消费冻结契约，不在板级调试时反向修改模型。第一代 KV260 部署入口见 `../4、模型上板/1、第一代SNN模型/README.md`。
