# MLP 转 SNN 实现状态

本目录保存第一代模型侧完整实现。2026-10-03 已将同学的组合工程拆分合并到这里；KV260 专用 RTL、位流和板端工具保留在 ../../../4、模型上板/1、第一代SNN模型/。

## 模型关系

H2 是 ANN Proposal + ANN Refiner，只作教师与基线。

- S0-P：浮点 Proposal-SNN + H2 ANN Refiner，是主要浮点转换路线。
- S0-R：H2 ANN Proposal + 浮点 Refiner-SNN，是转换位置对照。
- S1：定点 Proposal-SNN + A1-H2 ANN Refiner，是 exact 第一代主线。
- S2：混合 PoT Proposal-SNN + 同一 ANN Refiner，是权重量化消融。
- S3：LOA Proposal-SNN + 同一 ANN Refiner，是近似加法消融。

S1/S3 都是 SNN 前端加 ANN 后端，不应写成完整 SNN。S0-Full、Refiner 适配、独立时空 SNN 和读出研究的未通过结果均已保留。

## 主要入口

- S0/S1/S2/S3：software/scripts/train_s0p.py、train_s0r.py、evaluate_*.py。
- STC：analyze_stc.py、select_stc.py。
- Refiner 对照：train_s0r.py --study、select_s0r.py --study。
- 独立时空 SNN：study_temporal_segmentation.py、diagnose_temporal_snn.py。
- SNN 读出：study_snn_readout.py、train_snn_readout.py、select_snn_readout.py。
- 硬件契约：generate_rtl_vectors.py、generate_readout_vectors.py。

## 证据和共享副本

- results/ 保存第一代正式软件结果，包括准确性、定点、PoT、LOA、STC、Refiner 和时空 SNN 研究。
- software/weights/ 保存 H2、S0-P、S0-R、S1、S2、S3 冻结权重。
- data/index.json、H2 子包、公共数据和指标模块是独立运行所需的只读副本。
- results/runs/、数据集和训练 checkpoint 不属于正式迁移内容。

第 4 部分为了生成板端输入和复核冻结输出，保留了必要的只读模型依赖；模型侧源事实仍以本目录为准。

## 当前停止点

第一代模型研究和既有负面分支均已归档，不重新扫描阈值、位宽或近似位置。后续等待第 1 部分冻结新 ANN，再建立新版本的 MLP→SNN 转换链路。
