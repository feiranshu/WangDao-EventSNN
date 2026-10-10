# A 类复现入口

正式 H2/H2-pro A 源记录来自冻结后的 [同闭合前缀对比](../../../1_H2_model_improvement/seven_step_improvement_study/H2-pro/reports/AB_COMPARISON.md)。从冻结报告重新导出源记录与比较 CSV：

```powershell
python 1_H2_model_improvement/seven_step_improvement_study/H2-pro/export_formal_records.py
python 1_H2_model_improvement/implementation/software/scripts/rebuild_evaluation_tables.py --root 5_model_evaluation --allow-incomplete
python 1_H2_model_improvement/implementation/software/scripts/test_evaluation_layout.py
```

H2 旧预测由全序列评分裁剪，仍含旧非因果时间输入。H2-plus 原正式记录移入 Step7 中间模型，不参与正式比较。上面命令只转换冻结的源报告，不重新训练、选阈值或使用 Test 调参。

EV-SpSegNet 同口径官方 CUDA 补测仍待回传；论文直接给出的数值只能标记 `paper_reported`。
