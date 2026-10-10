> 当前版本（2026-10-10）：正式部署为 H2-pro-Causal-ANN-v1，C1、seed42 epoch10、exact-int8、阈值0.35；以下 A4-2 Step1–7 结论属于历史研究，旧正式文件在 Step7 中间模型目录。

# H2 ANN模型改进完整路线

版本：2026-10-05-final。本文件记录Step1—7正式结论、冻结协议以及最终部署决策。

## 1. 当前证据与边界

研究对象限于ANN两级MLP。Proposal保持128→48→24→16；H2 Refiner为32→64→32→4。Train-main只训练，Train-dev只作开发选择；Validation/Test在Step7前封存。步骤3以后高密度序列边界固定20,000 events/s。正式口径为int8权重、uint8激活、int32累加的软件回放和单一全局阈值。

| 步骤 | 最终结论 |
| --- | --- |
| Step1 | Refiner造成H2-F约89%的FN；089/092是Train-dev困难序列 |
| Step2 | A2采样/损失没有晋级；32维分组审查完成，不再安排删维实验 |
| Step3 | zero-update通过；A3不晋级；history探针最有潜力 |
| Step4 | density无稳定收益；删`sequence_time`退化；A4-2 history产生正向结果 |
| Step5 | 修正选型并冻结A4-2 seed42 epoch8、阈值0.40 |
| Step6 | 三seed平均AP/IoU增益与matched-null归因成立；原晋级因seed44普通密度Fa超过5%保护线而失败 |
| Step7 | Validation/Test独立验收完成；Validation保护门槛全过，Test仅高密度Recall保护未过；按部署优先级接受A4-2 |

H2-R的Train-dev exact-int8 AP为0.9274106331、总体IoU为78.0533%、高密度Recall为56.9128%、高密度IoU为47.5340%。冻结A4-2对应指标为0.9317999362、78.6834%、56.8014%、48.7483%。该部分只能表述为开发集上观察到改善；最终部署依据还包括Step7独立Test结果。

Step4原实现先在每个epoch选择单个最高IoU阈值，再检查资格。Step5改为对全部注册`epoch×threshold`组合先判资格后排名，并补齐非劣性中的总体Pd/Fa、高密度Fa和缺失指标处理。只修程序而保留原门槛时共有120个组合、5个非劣组合、0个完整替代组合；正确主点也是epoch8/0.40。新版开发协议有2个合格组合，最终按冻结排序选择epoch8/0.40。

新版协议是在看到Train-dev后制定，不能追认为Step4预注册规则。它在读取Validation/Test前冻结。

## 2. Step5完成状态与冻结接口

Step5不训练、不占A5编号。正式产物位于`../seven_step_improvement_study/step5/results/step5_formal_20261004/`：

- `corrected_original_decision.json`：原协议修正版；
- `revised_candidate_decision.json`：新版候选决策；
- `combination_audit.json`：全部注册组合与逐项门槛；
- `frozen_candidate.json`：Step6/7唯一配置入口；
- `history_availability.json`：使用epoch8重算的历史可用性0/1/2诊断；
- `artifacts/`与`metrics/`：冻结checkpoint、int8权重、H2/A4-2指标、PR与bootstrap输入。

一致性验证确认：决策checkpoint与Step4源文件哈希相同；复制checkpoint、int8导出、指标和阈值一致；导出int8回放最大绝对误差为0；Validation/Test均未读取。

后续不得重新搜索epoch、阈值或seed42模型。Step5冻结文件至少约束模型、seed、epoch、全局阈值、checkpoint与int8权重哈希、Proposal与索引哈希、历史特征、量化、后处理、密度边界和选择协议。

## 3. 统一资格条件

所有变化相对同一分区、同一评估器得到的冻结H2计算。百分点为绝对差值，Fa为相对变化。

| 指标 | 要求 |
| --- | --- |
| 总体Event IoU | 高于H2-R |
| 全事件AP | 不低于H2-R |
| 总体Pd / Fa | 相对H2-F，Pd下降≤0.5个百分点，Fa增加≤10% |
| 高密度Recall / IoU | 相对H2-R，各下降≤0.3个百分点 |
| 高密度Fa | 相对H2-R，增加≤10% |
| 普通密度IoU / Recall / Pd | 相对H2-R，各下降≤0.3个百分点 |
| 普通密度Fa | 相对H2-R，增加≤5% |
| Train-dev 089/092 IoU | 各自不低于H2-R，仅用于开发阶段 |

Fa基线为0时，候选也必须为0。关键指标缺失、非有限或必需分组无法计算时判不通过。旧版总体IoU +0.5个百分点、高密度Recall/IoU各 +2个百分点继续作为进阶目标报告，不作为唯一替换门槛。

## 4. Step6：三seed优化稳定性与历史归因

复用seed42正式结果，补充四个独立任务：A4-2 seed43、A4-2-null seed43、A4-2 seed44、A4-2-null seed44，GPU0—3各运行一个。所有任务从冻结H2初始化，使用相同四维history、自然全遍历、BCE、Step4两阶段学习率、batch4096和10轮预算；不开EMA/QAT，不增加轮数、seed、结构、损失或特征。

所有candidate统一使用Step5冻结的epoch8、阈值0.40，不按seed重新选点。部署模型始终为seed42。每个seed报告：

- candidate与null在同seed、同epoch8、同阈值0.40的主归因比较；
- candidate固定0.40与原注册组合中最佳合格null的保守诊断。最佳null不增加晋级硬门槛。

原计划进入Step7须同时满足：

1. seed42通过新版全部条件；
2. seed43/44均通过非劣性：AP下降≤0.1个百分点、总体及密度组保护全部通过，包括总体Pd/Fa和高密度Fa；
3. 至少2/3 seed总体IoU高于H2且AP不低于H2；
4. 三seed平均ΔIoU>0、平均ΔAP≥0；
5. 至少2/3 seed相对匹配null的IoU和AP均不下降，至少一项严格提高。

正式结果满足除seed44非劣性以外的全部聚合条件。seed44普通密度Fa相对H2-R增加6.36%，使原`step6_passes=false`；该结果永久保留，不追加seed45、epoch20或重新选点。与此同时，三seed的AP与总体IoU均有正增益，平均增益为正，三个matched-null归因均通过，足以支持“历史特征方向有效”，但不足以把原Step6写成正式晋级。

统计固定为10,000次、随机种子20261004、密度分层、序列级配对percentile bootstrap。每个密度层有放回抽取与原数量相同的序列；H2与候选使用相同抽样。IoU由TP/FP/FN总和重算，Pd/Fa由底层计数和分母重算，AP由抽中序列的全事件标签和分数重算。三seed不作为三倍独立样本。

固定敏感性报告：全Train-dev、去089、去092、同时去089/092、仅089、仅092；另报FP/FN、历史可用性0/1/2、密度分组、逐序列、float/int8差异、参数量、历史缓冲和完整流水线延迟。高密度样本很少，Step6不能宣称广泛高密度泛化。

## 5. Step7：Validation与Test补充独立验收

用户在完整审阅Step6后明确授权继续独立评测。Step7读取`frozen_candidate.json`和Step6完整决策，保留`step6_passes=false`，并在打开独立数据前冻结新的补充协议。固定seed42 checkpoint、epoch8、阈值0.40、历史实现、int8权重、量化、Proposal及阈值、后处理、数据索引、评估器和20,000 events/s边界。epoch8是按预注册排序选出的正式点；epoch10虽有部分单项更高，但没有同等级三seed证据，不能在Step7改选。

每个独立分区使用同一评估程序现场计算H2-F@序列化0.28阈值、H2-R@0.30和A4-2@0.40，所有差值现场计算，不引用旧表格。

Validation和Test按用户要求在同一次正式运行中依次完整评估，各自只读一次，不搜索、不重训，也不让Validation结果决定模型或阈值。Test不再以Validation通过为读取前提，但两者结果必须分别报告，不能合并后掩盖某个分区的退化。

预冻结的严格替换要求是Validation与Test固定安全条件全部通过；Test总体IoU高于H2-R、AP不低于H2-R，并且Δ总体IoU的95%配对bootstrap区间下界>0。正式结果中，Validation安全门槛全部通过；Test总体IoU、AP和置信度检查通过，但高密度Recall相对H2-R下降1.2138个百分点，超过0.3个百分点保护线，因此预冻结的严格验收结果为未全部通过。

独立Test上，A4-2相对H2-R的AP提高0.1712个百分点、总体IoU提高0.9431个百分点、Precision提高1.2933个百分点、F1提高0.5427个百分点、总体Fa下降6.11%；高密度IoU提高3.0945个百分点、高密度Fa下降13.49%。总体IoU差值的10,000次配对序列bootstrap 95%区间为+0.0565至+2.2878个百分点。

在完整披露上述严格门槛结果和高密度Recall取舍后，用户明确把整体IoU、Precision和误报控制作为最终部署优先级，并决定以A4-2作为本次模型改进的最终结果。因此A4-2已迁入`implementation`。这是独立于原科研门槛的工程接受决策，不追改`step6_passes=false`或Step7的`passes_all_independent_acceptance_checks=false`，也不得宣称A4-2在所有指标上无条件占优。

089/092只属于Train-dev保护，Validation/Test不临时寻找替代特殊序列或增加门槛。基础设施中断可按同配置恢复未完成运行；不得因指标不理想而重复评测。

## 6. 工程与产物规则

- 每步保留协议、环境、哈希、日志、指标、checkpoint、决策和排除缓存的结果清单。
- README服务器命令必须单行可复制；上传包包含SHA、manifest和POSIX权限验证。
- 根目录模型指标库一个模型一个子目录，只保存数值、图表和说明；代码与checkpoint留在步骤工作区。
- A4-2正式运行代码、seed42 epoch8 checkpoint、exact-int8权重、模型卡和精简Validation/Test结果已迁入`implementation`。
- 旧整体压缩包、修复前服务器包、smoke权重、临时预测和候选缓存已在确认无部署/SNN/板级依赖后删除；最终GitHub交付以当前项目文件夹为准。
- SNN、近似计算、RTL与板级工作不进入本ANN路线。
