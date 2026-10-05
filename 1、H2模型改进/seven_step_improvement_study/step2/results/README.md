# 步骤二正式回传结果

`step2_formal_20261003/`是服务器正式运行的回传目录。`RESULT_MANIFEST.json`列出的45个文件已在本地重新计算SHA-256，全部与清单一致。

最终决策见`step2_formal_20261003/step2_decision.json`：A2-1、A2-2均拒绝，A2-3、A2-4按诊断条件跳过，`step2_passes=false`，Validation/Test均未使用。

训练checkpoint和完整历史保留在本目录以便审计；根目录模型指标库只复制数值、CSV和SVG，不复制权重、缓存或生成代码。
