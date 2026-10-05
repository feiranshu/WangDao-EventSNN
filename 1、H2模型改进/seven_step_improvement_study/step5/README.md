# 步骤五：A4-2选型修正与候选冻结

Step5已经完成本地正式回放。它不训练新模型、不占用A5编号，也不进行特征删除。唯一研究对象是Step4的A4-2四维历史特征模型及其同维null对照。

## 完成内容

- 只遍历Step4预注册epoch `0/1/2/5/8/10`及原阈值网格，共检查120个组合；
- 先检查资格，再在合格组合中排名；
- 补齐非劣性中的总体Pd/Fa和高密度Fa保护；
- 分开生成原协议修正版与新版开发期决策；
- 从真正选中的checkpoint重新生成float/int8指标、PR曲线、阈值曲线和历史可用性分组；
- 验证决策checkpoint、复制checkpoint、int8导出、指标来源与阈值完全一致。

## 正式结论

原Step4科学门槛不变时，A4-2仍不是完整替代模型，但正确主检查点应为epoch8、阈值0.40，而不是epoch0。新版开发期协议冻结A4-2 epoch8、阈值0.40；此后Step6和Step7均严格沿用该身份，最终按整体IoU、Precision和误报控制优先接受A4-2。

| 指标 | H2-R | 冻结A4-2 | 变化 |
| --- | ---: | ---: | ---: |
| int8 AP | 0.927411 | 0.931800 | +0.4389个百分点 |
| 总体IoU | 78.0533% | 78.6834% | +0.6301个百分点 |
| 高密度Recall | 56.9128% | 56.8014% | -0.1114个百分点 |
| 高密度IoU | 47.5340% | 48.7483% | +1.2142个百分点 |
| 普通密度IoU | 94.6533% | 94.5291% | -0.1242个百分点 |
| train_089 IoU | 16.7130% | 21.0329% | +4.3199个百分点 |
| train_092 IoU | 59.1566% | 59.4055% | +0.2489个百分点 |

这些仍是Train-dev开发结果，不能写成独立泛化结论。Validation/Test均未读取。

## 目录

```text
step5/
├─ README.md
├─ README_SERVER.md
├─ step5_protocol.json
├─ run_step5.py
├─ step5_common.py
├─ collect_step5_results.py
├─ build_server_package.py
├─ tests/test_step5.py
├─ verification/
├─ results/step5_formal_20261004/
└─ upload/
```

正式结果的入口为`results/step5_formal_20261004/step5_decision.json`，冻结接口为`frozen_candidate.json`。`combination_audit.json`保留全部组合及逐项检查，`history_availability.json`使用epoch8模型重算并与H2比较。

精简回传归档为`results/step5_results_20261004.zip`，不包含候选缓存，SHA-256为`3633a2f60ba22ffa60623f19d01085159fef77c5f8445d017deba5a1625b07c7`。独立服务器包为`upload/H2_step5_server_20261004.zip`，SHA-256为`1f755c64171874ac08ceaaf991dcf62f961b4fa59365fbbc36adeee5a6f5f3ee`；该包独立解压后的8项测试全部通过。

## 本地复核命令

```bash
python -m unittest discover -s tests -v
```

```bash
python run_step5.py run-all --step4-results ../step4/results/step4_formal_20261004 --output runs/step5_review --device cpu --run
```

```bash
python run_step5.py verify --results results/step5_formal_20261004 --step4-results ../step4/results/step4_formal_20261004 --output verification/manual_artifact_check.json --device cpu
```

所有命令均为单行。重新复核必须使用新输出目录，不能覆盖正式结果。

## 服务器回传复核

服务器结果已保存到`step5_results_server/step5_formal/`。冻结checkpoint、组合审计、原协议修正版、新版决策、H2指标、A4-2关键int8表格和一致性验证均与本地相同；重新序列化的NPZ/SVG、绝对路径及约`1e-16`级浮点表示差异不改变任何结论。Step5无需修改计划，后续以`../step6/`为准。
