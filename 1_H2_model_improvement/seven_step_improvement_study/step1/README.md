# 第一步：H2基线复核、错误诊断与标注审查（已完成）

本目录独立保存七步研究计划第一步的代码和结果。当前路径为`seven_step_improvement_study/step1/`；位置变化不改变冻结H2权重、数据索引、评测口径或既有结论。

当前以已经生成的`results/`和根目录指标库为准，不需要重新运行步骤1。保留的`run_first.py`用于追溯生成过程；它仍引用`implementation/`中的模型、数据索引和公共评测模块，因此迁移后不能把下面的旧命令当作当前直接执行入口。若以后确需复核，应先单独适配路径并使用新的输出目录，不能覆盖现有结果。

步骤1读取的数据资产仍来自`../../implementation/data/index.json`、EV-UAV数据和`../../implementation/software/weights/H2_D1_seed42.npz`。既有结果位于本目录`results/`。

主要结果：

```text
seven_step_improvement_study/step1/results/
├─ REPORT.md                    # 面向研究决策的第一步结论
├─ step1_report.json            # 完整机器可读报告和溯源信息
├─ feature_diagnostics.csv      # TP/FN/FP/TN的32维特征统计
└─ label_review/
   ├─ review_manifest.csv       # 重点窗口、目标ID、边界框和内部一致性
   ├─ train_089_review.png      # 灰=背景、红=FN、绿=TP、蓝=FP
   └─ train_092_review.png
```

Step1 原始诊断资料现归档在 `../evaluation_cleanup_archive/2026-10-07/historical_step1_results/`。冻结 H2 的正式 Validation/Test 结果位于项目 `5_model_evaluation/A_model_accuracy/02_model_records/formal/h2__software_exact_int8/`；正式横向比较按单指标 CSV 保存，生成代码留在 implementation。

`label_review`只能检查NPZ内部标签与事件轨迹是否自洽。若没有原始图像或原始框标注，不能据此宣称语义标注完全正确。

本步骤已经结束。步骤2、3正式训练均无候选晋级；Step4确认四维因果历史候选A4-2有效，Step5冻结epoch 8/阈值0.4，Step6完成多seed复现，Step7完成独立验收，最终按整体IoU、Precision和误报控制优先接受A4-2。这里生成的序列`6889.9906`和窗口`6820 events/s`阈值是步骤1、2的历史协议，不回写结果；步骤3以后改用20,000 events/s序列边界。最终状态以`../README.md`为准。

