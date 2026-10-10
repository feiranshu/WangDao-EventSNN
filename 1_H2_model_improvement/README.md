# H2 ANN 模型改进工程

正式最终版本是 **H2-pro-Causal-ANN-v1**：保留 H2 Proposal，Refiner 仍为 36→64→32→4，将 H2-plus 的非因果全序列相对时间维度换成已闭合窗口 Proposal 历史运动 C1 特征。冻结配置是 seed42、epoch10、exact-int8、阈值 0.35。100 ms 窗口按 watermark 发布，尾部不完整窗口不发布。

H2-plus/A4-2 已退出正式部署与第 5 板块注册表，旧权重、结果和正式记录归入 [Step7 中间模型](seven_step_improvement_study/step7/intermediate_H2-plus/README.md)。原 Step5/Step7 结论保持为历史事实。[H2-pro 研究目录](seven_step_improvement_study/H2-pro/README.md)保存训练、冻结及评测证据；[implementation](implementation/README.md)是正式运行入口。

| 划分 | 模型 | AP | IoU | Precision | HOTA |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | H2 | 96.1159% | 86.1858% | 94.2416% | 39.7741% |
| Validation | H2-plus（中间模型） | 96.1495% | 86.7004% | 94.4953% | 43.7617% |
| Validation | H2-pro | 96.3589% | 87.8318% | 94.9168% | 43.7255% |
| Test | H2 | 96.7807% | 85.9198% | 90.5479% | 44.0271% |
| Test | H2-plus（中间模型） | 96.9565% | 86.8645% | 91.8429% | 43.1441% |
| Test | H2-pro | 97.0823% | 87.1383% | 91.9882% | 42.1332% |

Test 上 H2-pro 对 H2-plus 的 IoU 增量为 0.2737 个百分点，分层配对 bootstrap 95% 区间为 −0.0073 至 +0.6117 个百分点，跨 0；HOTA 低 1.011 个百分点。选用 H2-pro 主要是因果实时性版本决策，不能声称所有指标稳定领先。H2/H2-plus 对照来自旧全序列预测裁剪，仍带非因果时间输入。完整 A/B 指标见 [对比报告](seven_step_improvement_study/H2-pro/reports/AB_COMPARISON.md)。

```bash
python -m pip install -e 1_H2_model_improvement/implementation
python 1_H2_model_improvement/implementation/software/scripts/run_h2_pro.py path/to/input.npz --output runs/h2_pro_predictions.npz
```

本目录下只保留两个子文件夹：正式 `implementation/`，以及包含原七步、中间模型和 H2-pro 研发证据的 `seven_step_improvement_study/`。数据集不入库；第 5 板块正式注册 H2、H2-pro、EV-SpSegNet，后者同口径补测未完成。

