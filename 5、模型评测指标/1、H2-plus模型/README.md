# H2-plus 模型评测

H2-plus 是原 A4-2 历史特征模型的正式名称。它在 32 维 H2 Refiner 输入上增加四个因果 Proposal 历史特征，最终部署配置为 seed 42、`epoch_008.pt`、全局阈值 0.40。

- 正式数值路径：`exact_int8_replay`。
- 量化形式：uint8 激活、int8 权重、int32 乘加。
- 指标协议：`evuav-33e2aff-official-compatible-v1+step3-density-20000`。
- Train-dev int8：AP 0.9317999362，IoU 0.7868339501，高密度 IoU 0.4874826147。
- Step 6：seed42/43/44 均优于各自同 epoch、同阈值的匹配 null；seed43/44 完整 float/int8 结果位于 `stability/`。
- Step 7 Test：AP 0.9693089442，IoU 0.8689306583，Precision 0.9186917421，Recall 0.9413222793，F1 0.9298693394，Fa 1.0871152000e-05。
- 最终结论：H2-plus 已接受为正式部署模型。相对 H2，整体 IoU、Precision、F1、Fa 与高密度 IoU 改善，Recall、Pd 与高密度 Recall 略降。

目录结构：

- `float/`、`int8/`：Train-dev seed42 冻结开发结果。
- `stability/seed43/`、`stability/seed44/`：Step 6 稳定性复现。
- `validation/`、`test/`：Step 7 独立验收，同时保留 float 与 int8。
- `与H2对比/`：最终独立评测原始宽表。

为保持研究记录可追溯，原始 JSON 中的模型 ID `A4-2` 不改写；对外目录、图表和汇总表使用 `H2-plus`。
