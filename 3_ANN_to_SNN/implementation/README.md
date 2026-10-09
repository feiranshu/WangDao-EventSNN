# MLP 转 SNN 实现状态

本目录保存第一代模型侧完整实现。2026-10-03 已将同学的组合工程拆分合并到这里；KV260 专用 RTL、位流和板端工具保留在 `../../../4_model_deployment/1_first_generation_SNN_model/`。

## 模型关系

H2 是 ANN Proposal + ANN Refiner，只作教师与基线。

- S0-P：浮点 Proposal-SNN + H2 ANN Refiner，是主要浮点转换路线。
- S0-R：H2 ANN Proposal + 浮点 Refiner-SNN，是转换位置对照。
- S1：定点 Proposal-SNN + A1-H2 ANN Refiner，是 exact 第一代主线。
- S2：混合 PoT Proposal-SNN + 同一 ANN Refiner，是权重量化消融。
- S3：LOA Proposal-SNN + 同一 ANN Refiner，是近似加法消融。

S1/S3 都是 SNN 前端加 ANN 后端，不应写成完整 SNN。S0-Full、Refiner 适配、独立时空 SNN 和读出研究的未通过结果均已保留。

## 主要入口

- S0/S1/S2/S3：`software/scripts/train_s0p.py`、`train_s0r.py`、`evaluate_*.py`。
- STC：`analyze_stc.py`、`select_stc.py`。
- Refiner 对照：`train_s0r.py --study`、`select_s0r.py --study`。
- 独立时空 SNN：`study_temporal_segmentation.py`、`diagnose_temporal_snn.py`。
- SNN 读出：`study_snn_readout.py`、`train_snn_readout.py`、`select_snn_readout.py`。
- 硬件契约：`generate_rtl_vectors.py`、`generate_readout_vectors.py`。

## 证据和共享副本

- `results/` 保存第一代正式软件结果，包括准确性、定点、PoT、LOA、STC、Refiner 和时空 SNN 研究。
- `software/weights/` 保存 H2、S0-P、S0-R、S1、S2、S3 冻结权重。
- `data/index.json`、H2 子包、公共数据和指标模块是独立运行所需的只读副本。
- `results/runs/`、数据集和训练 checkpoint 不属于正式迁移内容。

本目录结果是 [`../model_selection_metrics/`](../model_selection_metrics/README.md) 中 S01/S02 的源证据。候选模型选定后，软件正式版本进入目录 5 的 A+B；部署到开发板的正式配置以独立 `evaluation_id` 进入 A+C。SNN 的有效 MAC、有效 AC 必须在 B04a/B04b 分开统计，正式与 B03 稠密 MAC 比较时使用 B04c 二者合计。

## 当前停止点

第一代模型研究和既有负面分支均已归档，不重新扫描阈值、位宽或近似位置。后续等待第 1 部分冻结新 ANN，再建立新版本转换链路。
