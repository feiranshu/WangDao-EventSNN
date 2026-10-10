# B 类复现入口

H2/H2-pro 的同闭合前缀 B01–B04 源报告由 `1_H2_model_improvement/seven_step_improvement_study/H2-pro/reports/ab_same_prefix_{validation,test}_v2.json` 冻结。重建正式记录和 CSV：

```powershell
python 1_H2_model_improvement/seven_step_improvement_study/H2-pro/export_formal_records.py
python 1_H2_model_improvement/implementation/software/scripts/rebuild_evaluation_tables.py --root 5_model_evaluation --allow-incomplete
python 1_H2_model_improvement/implementation/software/scripts/test_evaluation_layout.py
```

B03/B04 只计 Proposal 和条件 Refiner exact-int8 MAC，普通 ANN 的 AC=0；未包含特征构建、后处理、CPU 延迟和内存工作区。旧 H2-plus 全 8 秒复杂度复现工具已移入 Step7 中间模型材料，不能写回正式注册表。EV-SpSegNet 官方 CUDA 补测仍待完成。
