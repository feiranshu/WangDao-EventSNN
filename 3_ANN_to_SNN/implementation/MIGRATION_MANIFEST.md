# 模型侧迁移清单

## 2026-10-01：从 H2 改进目录拆分

- 迁入 S0/S1/S2/S3、STC、定点、PoT、LOA、计算代理和冻结权重。
- 为独立运行保留 H2、数据索引、公共数据和指标模块的只读副本。
- ANN A1/A2/A3、浅层 ANN 学生和 C0—C4 因果 ANN 仍归第 1 部分。

## 2026-10-03：合并同学的第一代 SNN 工程

来源：`4_model_deployment/1_first_generation_SNN_model/Wangdao-project/Wangdao-project-main/`。

- 40 个已有文件经 SHA-256 核对完全相同，没有重复写入。
- 5 个较新实现替换模型侧旧版本：`generate_rtl_vectors.py`、`select_s0r.py`、`train_s0r.py`、`s0r.py`、`snn.py`。
- 被替换版本保存在根级本地归档 `../../archive/3_ANN_to_SNN/first_generation_SNN_model_pre_merge_implementation_20261003/implementation/`。
- 新增 Refiner 受限研究、独立时空 SNN、读出对照、加载优化、对应测试和正式结果。
- RTL、testbench、黄金向量、Vivado 工程、位流/XSA、设备树、板端脚本和板级结果仍归第 4 部分。

拆分前完整工程另存为根级本地归档 `../../archive/first_generation_SNN_model_pre_split_complete_project_20261003.zip`。

## 2026-10-08：迁入模型选型指标

- `5_model_evaluation` 正式取消 D 类，只保留 A/B/C。
- 原 D/C01“时间步—效果—成本曲线”迁入 `../model_selection_metrics/`，改编号为 S01。
- 原 D/C02“转换与数值近似效果损失”迁入同目录，改编号为 S02。
- S01/S02 只用于模型优化、消融和选型；正式模型仍按目录 5 的 A/B/C CSV 评测。
- SNN 运算对比统一遵守 `有效运算总量 = 有效 MAC + 有效 AC`，两项分开留存，总量与稠密 MAC 比较。
