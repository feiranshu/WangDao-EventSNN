# 步骤六：A4-2三seed优化稳定性与历史归因

Step5服务器回传结果与本地冻结结果一致，因此Step6不修改研究方向。冻结部署候选仍为A4-2 seed42、epoch 8、全局阈值0.40；Step6只补充seed43/44及其匹配null，不重新搜索模型、epoch或阈值。

## 固定实验

四个正式任务分别为：

| GPU | 任务 | 作用 |
| ---: | --- | --- |
| 0 | A4-2 seed43 | 历史特征候选稳定性 |
| 1 | A4-2-null seed43 | 同seed、同结构、同预算归因对照 |
| 2 | A4-2 seed44 | 历史特征候选稳定性 |
| 3 | A4-2-null seed44 | 同seed、同结构、同预算归因对照 |

四个任务均从冻结H2初始化，使用自然全遍历、BCE、batch 4096和Step4相同的两阶段学习率，共训练10轮。每轮在终端显示并保存loss、学习率、梯度、激活饱和率、float/int8 AP、IoU、高密度Recall和耗时。正式比较固定读取epoch 8、阈值0.40；epoch 10仅用于观察收敛轨迹，不能重新选点。

## 原Step7解封条件与正式结论

必须同时满足：

1. seed42通过新版全部保护条件；
2. seed43和seed44均通过非劣性；
3. 至少2/3 seed总体IoU高于H2且AP不低于H2；
4. 三seed平均IoU变化大于0、平均AP变化不小于0；
5. 至少2/3 seed相对同seed、同epoch、同阈值null的IoU和AP均不下降，且至少一项严格提高。

正式结果中，seed42/43通过、三个seed的AP和总体IoU均高于H2、三seed平均增益为正、三个matched-null归因均通过；唯一失败项是seed44普通密度Fa相对H2-R上升6.36%，超过5%保护线。因此原协议的`step6_passes=false`保持不变，不增加seed45、不续训到epoch20、不重新搜索阈值。

用户在审阅完整Step6结果后，明确将“历史特征方向验证成功”与“原保护门槛未全部通过”分开处理，并授权建立Step7补充独立验收。Step7固定使用原部署候选A4-2 seed42、epoch8、阈值0.40，同时评测Validation与Test；这不会追溯修改本步骤的失败结论。实现见`../step7/`。

Step7现已完成：Validation安全门槛全部通过；Test整体IoU、Precision、AP和误报控制改善，只有高密度Recall保护门槛未过。用户最终按整体IoU、Precision和误报控制优先接受A4-2；本步骤的`step6_passes=false`仍原样保留。

## 统计与报告

- 10,000次、随机种子20261004、按密度分层的序列级配对percentile bootstrap；
- 每个密度层按原序列数有放回抽取，候选与H2共用抽样；
- AP使用抽中序列的完整事件标签与分数精确重算；IoU、Pd、Fa从底层计数重算；
- 固定报告全Train-dev、去089、去092、同时去089/092、仅089和仅092；
- 报告历史可用性0/1/2、逐序列结果、float/int8差异、参数量、历史缓冲及完整流水线延迟；
- 本步骤自身没有Validation/Test入口；独立数据只由后来冻结的`../step7/`协议读取。

## 目录

```text
step6/
├─ README.md
├─ README_SERVER.md
├─ step6_protocol.json
├─ run_step6.py
├─ step6_common.py
├─ step6_training.py
├─ collect_step6_results.py
├─ build_server_package.py
├─ tests/test_step6.py
├─ run_server.sh
├─ requirements.txt
├─ verification/
└─ upload/
```

本步骤正式结果已经由服务器四卡CUDA生成并回传到`results/`。研究结束后，旧外层总压缩包已确认不再需要并删除；正式交付以项目当前目录和`implementation/`为准。

初版服务器包在正式运行前用浮点逐位相等检查Step4密度参数，可能因不同NumPy/libm实现的末位差异误报。2026-10-05修正版改为：Train-main候选数和序列数必须精确一致，并逐候选验证两套参数生成的uint8密度编码完全一致；差异及参数写入`density_compatibility.json`。这不会放宽数据身份检查，而且A4-2/A4-2-null实际只使用history列，不使用density列。已解压初版包可按`README_SERVER.md`中的单行热修复继续原运行目录，无需重传完整包。

初版上传包`upload/H2_step6_server_20261005.zip`及SHA-256 `dabbe24db29cfc75dacc8d1081a2c64ae7b5ac1a75888cba4faca611e78033b6`保留作为历史证据；修正版完整包为`upload/H2_step6_server_20261005_hotfix1.zip`，SHA-256为`99022495bd29264c0dc3adb686d9ed25f70f38dcaac4d57a48e30f03be4c88bf`，不覆盖初版。当前没有伪造本地正式结果，服务器回传前根目录指标库也不会提前创建seed43/44指标目录。
