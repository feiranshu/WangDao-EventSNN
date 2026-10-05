# 模型评测指标

本目录只保留当前正式模型、H2 改进计划中间产物和对外使用的图表。正式比较统一采用 `exact_int8_replay`，指标协议为 `evuav-33e2aff-official-compatible-v1+step3-density-20000`。

## 目录结构

- `0、原始H2模型/`：原始 H2 历史评测，以及与 H2-plus 同协议的 Train-dev、Validation、Test int8 统一评测。
- `H2-plus模型/`：正式入选模型。历史实验 ID 为 `A4-2`，正式展示名称统一使用 `H2-plus`。
- `1、H2改进计划中间产物模型/`：未入选候选、匹配对照、旧全模型统一评测表和旧全模型 PR 图。
- `Charts and pictures/`：只保存正式入选模型的 PR 曲线对比和指标对比表。

## Charts and pictures 固定接口

该目录只维护两类输出：

1. `PR曲线对比/`：H2 与 H2-plus 在 Validation、Test 上的两张 int8 PR 曲线对比图。每张图同时提供 PNG 与 SVG。
2. `指标对比表/`：H2 与 H2-plus 在 Validation、Test 上的 AP、IoU、Precision、Recall、F1、Pd、Fa 和高密度指标对比。CSV 用于数据处理，PNG 用于汇报。

后续有新模型正式入选时，只更新这两类输出，并在正式模型目录中新增或替换对应模型评测；未入选模型及历史全模型对比统一放入 `1、H2改进计划中间产物模型/`。

## 当前正式模型

- H2：统一工作点阈值 0.30。
- H2-plus：seed 42、epoch 8、工作点阈值 0.40。

H2-plus 是 A4-2 的正式名称。为保证历史实验可追溯，已有 JSON、checkpoint 标识和研究目录中的 `A4-2` 实验 ID 不做批量改写。
