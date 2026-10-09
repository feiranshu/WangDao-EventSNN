# 正式结果目录

本目录只保存最终A4-2的部署决策和精简独立评测报告，不保存训练缓存、候选预测或失败模型。完整研究结果仍在`../../seven_step_improvement_study/`。

## 目录结构

```text
results/
├─ README.md                         本说明
└─ final_A4-2/
   ├─ README.md                      最终结果文件逐项说明
   ├─ deployment_decision.json       工程部署选择
   ├─ frozen_candidate_step5.json    模型冻结接口
   ├─ step7_protocol.json            独立验收冻结协议
   ├─ step7_preflight.json           环境、权重和数据检查
   ├─ step7_decision.json            原始预设门槛结论
   ├─ validation/                    Validation指标、图表与bootstrap
   └─ test/                          Test指标、图表与bootstrap
```

根目录`../../../5_model_evaluation/`仍可作为跨模型指标总览；本目录是随最终工程一起交付的自包含证据副本。

