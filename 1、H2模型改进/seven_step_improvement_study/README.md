# H2 ANN七步改进研究历程

版本：2026-10-05-final。本目录保存从H2基线诊断到A4-2独立验收的完整研究证据，用于追溯、复核和复现；它不是部署入口。最终运行代码、权重和精简结果位于[implementation](../implementation/README.md)。

## 最终状态

| 步骤 | 状态 | 结论 |
| --- | --- | --- |
| Step1 基线与诊断 | 完成 | Refiner是主要FN来源，089/092是开发期困难序列 |
| Step2 采样、损失、特征组审查 | 完成，无晋级 | A2没有超越H2；停止A2权重与删维路线 |
| Step3 前置、探针与A3 | 完成，无晋级 | zero-update通过；A3不晋级；history探针最有价值 |
| Step4 新增信息 | 完成 | A4-2历史信息有效；密度、删时间与未解封组合停止 |
| Step5 选型修正与冻结 | 完成 | 冻结A4-2 seed42、epoch8、阈值0.40 |
| Step6 三seed稳定性与归因 | 完成，原门槛失败 | 三seed平均增益和matched-null归因成立；seed44普通密度Fa超过5%保护线 |
| Step7 Validation/Test独立验收 | 完成 | Validation安全门槛全过；Test整体收益成立，高密度Recall保护门槛未过 |

最终按“整体IoU、Precision和误报控制优先”的部署目标接受A4-2。独立Test相对H2-R：整体IoU +0.9431个百分点、Precision +1.2933个百分点、AP +0.1712个百分点、总体Fa −6.11%、高密度IoU +3.0945个百分点；已知取舍为高密度Recall −1.2138个百分点。原Step6和Step7严格门槛结果不被改写。

## 固定模型身份

```text
模型：A4-2
结构：H2 Proposal + 36→64→32→4 Refiner
输入：原32维 + 4维因果历史信息
训练点：seed42 / epoch8
全局阈值：0.40
正式推理：exact-int8
```

## 目录结构与作用

```text
seven_step_improvement_study/
├─ README.md                         本文件：研究总览、最终状态和目录导航
├─ step1/                            建立H2基线、PR/阈值指标、错误归因及标注审查
│  └─ README.md                      Step1文件、脚本、指标和结果说明
├─ step2/                            采样/损失方案、32维特征组审查及A2正式结果
│  └─ README.md                      Step2模型、服务器包、修复和结果说明
├─ step3/                            zero-update、D3-0信息上限探针及A3训练
│  └─ README.md                      Step3协议、代码、测试、上传包和结果说明
├─ step4/                            密度、历史、删时间和配对null受控实验
│  └─ README.md                      A4系列定义、执行入口和正式结果说明
├─ step5/                            修正选型逻辑、审计全部组合并冻结A4-2
│  └─ README.md                      冻结接口、指标、权重和选择依据说明
├─ step6/                            seed43/44复现、matched-null归因和bootstrap
│  └─ README.md                      多seed执行、稳定性门槛和结果说明
└─ step7/                            冻结A4-2在Validation/Test上的独立验收
   ├─ README.md                      Step7协议、代码目录和最终结果摘要
   ├─ README_SERVER.md               历史服务器复现单行命令
   └─ step7_results_20261005/         Validation/Test完整正式结果和图表
```

各步骤下的代码文件、协议文件、服务器压缩包、日志和结果均由该步骤自己的README解释。本层只说明大类用途，避免与子目录文档重复。

## 使用规则

- 本目录中的旧候选、smoke结果、训练缓存和历史服务器包不得替代`implementation/`中的最终权重。
- Validation/Test结果不得用于继续调参；最终A4-2身份保持seed42、epoch8和阈值0.40。
- 高密度边界保持20,000 events/s；Step1/2采用的早期口径只作为历史记录保留。
- 复核某一步时从对应README进入；部署或集成时从`../implementation/README.md`进入。
