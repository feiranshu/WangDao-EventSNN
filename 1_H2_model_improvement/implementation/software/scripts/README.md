# 命令行脚本

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `run_a4_2.py` | 最终A4-2单序列exact-int8推理入口 |
| `build_delivery_manifest.py` | 为最终implementation生成排除数据集和缓存的SHA-256交付清单 |
| `evaluate.py` | 原H2基线评测与历史结果复现 |
| `diagnose_h2.py` | H2 Proposal/Refiner错误归因与逐序列诊断 |
| `audit_data.py` | 数据索引、标签和数据角色审计 |
| `train_h2_refiner.py` | H2 Refiner训练实验入口 |
| `freeze_phase1.py` | 冻结阶段一模型、阈值和产物 |
| `release_h2_family.py` | H2家族发布与清单生成 |
| `evaluate_causal.py` | 因果/历史结构评测工具 |
| `train_causal.py` | 因果模型训练入口 |
| `train_h2_student.py` | H2学生模型训练入口，非最终A4-2依赖 |

正式使用只需`run_a4_2.py`。历史训练脚本不得在Test上重新选点后覆盖最终权重。

