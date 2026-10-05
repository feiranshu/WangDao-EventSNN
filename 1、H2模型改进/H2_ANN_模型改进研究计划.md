# H2 ANN模型改进研究计划与最终结论

计划版本：2026-10-05-final。完整实验协议、各步结论和边界以[完整路线](implementation/PLAN.md)为准。

## 最终结论

本项目只改进传统ANN两级MLP。Proposal保持128→48→24→16；最终A4-2在原H2基础上把Refiner由32→64→32→4改为36→64→32→4，新增4维因果历史信息。正式模型固定为seed42、epoch 8、全局阈值0.40和exact-int8推理。

独立Test上，A4-2相对H2-R的整体IoU提高0.9431个百分点、Precision提高1.2933个百分点、AP提高0.1712个百分点、F1提高0.5427个百分点，总体Fa下降6.11%；高密度IoU提高3.0945个百分点、高密度Fa下降13.49%。整体IoU差值的10,000次配对序列bootstrap 95%区间为+0.0565至+2.2878个百分点。

原有严格保护门槛并未全部通过：Step6的seed44普通密度Fa相对增幅为6.36%，超过5%保护线；Step7的Test高密度Recall相对H2-R下降1.2138个百分点，超过0.3个百分点保护线。因此科研记录仍保留`step6_passes=false`和Step7严格验收未全过。根据最终部署目标更重视整体IoU、Precision和误报控制的明确选择，A4-2被正式接受为本次工程交付模型。该结论是带已知取舍的工程决策，不能写成所有指标无条件优于H2。

## 七步路线结论

1. H2基线、错误归因与标注审查：完成；Refiner是主要FN来源，089/092是Train-dev困难序列。
2. 采样、损失与32维分组审查：完成；A2没有晋级，不再安排删维实验。
3. zero-update、D3-0与A3训练：完成；工程链路无异常，A3没有晋级，history探针最有潜力。
4. 新增信息与配对对照：完成；密度特征没有稳定收益，删`sequence_time`退化，四维历史信息有效。
5. 选型修正与冻结：完成；冻结A4-2 seed42、epoch8、阈值0.40，不占A5编号。
6. 三seed优化稳定性与历史归因：完成；平均AP/IoU增益和三组matched-null归因成立，原协议因seed44普通密度Fa单项超线而未晋级。
7. Validation与Test独立验收：完成；Validation安全门槛全过，Test的IoU、AP和置信度检查通过，仅高密度Recall保护门槛未过。按最终部署优先级采纳A4-2。

## 固定研究边界

- Train-main只训练，Train-dev用于开发选择；Validation/Test只作冻结后的独立评测，不反馈调参。
- 步骤3以后高密度边界固定为20,000 events/s；正式选型使用exact-int8回放和单一全局阈值。
- Step5之后不重新搜索seed、epoch、阈值、特征、损失或结构。
- A4-2的优势定位是整体IoU、Precision和误报控制；部署时必须同时披露高密度Recall取舍。
- SNN、近似计算、RTL和上板研究不属于本目录，统一在`../3、mlp转SNN/`开展。
- `implementation/`是最终运行入口；`seven_step_improvement_study/`只承担完整研究历程、复核和复现；`archive/`不保存正式模型或正式指标。

## 交付目录

```text
1、H2_S1模型改进研究计划/
├─ README.md                         GitHub首页、最终指标、快速入口和总目录
├─ H2_ANN_模型改进研究计划.md       本文件：研究路线、边界和最终决策
├─ .gitignore                        排除数据集、缓存和本地临时文件
├─ implementation/                   最终A4-2部署工程、权重、测试与精简结果
├─ seven_step_improvement_study/      Step1—7完整研究历程和原始实验产物
└─ archive/                          已清理归档；仅保留删除范围和依据说明
```

研究完成前的旧整体压缩包、修复前服务器包、smoke权重、临时预测和候选缓存均已确认不被部署、SNN或KV260工作区引用，并从`archive/`删除。GitHub交付以当前文件夹本身为准，不再保留旧总包。
