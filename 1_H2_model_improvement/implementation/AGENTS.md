# H2 ANN模型改进协作规则

## 范围

本目录是 H2-pro-Causal-ANN-v1 的最终部署工程。H2-pro 的完整开发证据位于`../seven_step_improvement_study/H2-pro/`；H2-plus/A4-2 的七步原始研究与当时的部署决定是历史记录，现归入`../seven_step_improvement_study/step7/intermediate_H2-plus/`。正式模型固定为 C1 Proposal 历史运动特征、seed42 epoch10、exact-int8、阈值0.35。用户已明确要求以因果实时性为版本决策。后续不得用 smoke 或开发候选替换本目录的冻结模型。

SNN、LIF、膜电位、PoT、LOA、近似计算和RTL工作统一放在`../../3_ANN_to_SNN/`。

## 数据与评测

- Train-main只训练；Train-dev用于checkpoint、阈值、早停和候选选择。
- Validation与Test只在模型、量化路径、阈值、数据索引及后处理冻结后各运行一次；两者均不得反馈调参。
- `data/index.json`是唯一冻结索引，不得用受保护数据调参。
- 步骤3以后序列事件率`>= 20000 events/s`定义为高密度；步骤1/2旧结果保留原序列`6889.9906`和窗口`6820 events/s`口径，不覆盖历史产物。
- 冒烟、Train-dev、探针和软件回放不得写成Validation、Test或板级结论。

## 实验规则

- 准确性优先于复杂度，一次正式受控实验只改变一个主要因素。
- A2-1/A2-2编号及结果冻结；A2-3/A2-4保留为诊断跳过记录。
- A3-1/A3-2结果与编号冻结。Step4编号为A4-1密度、A4-2历史、A4-3删`sequence_time`、A4-4密度+历史、A4-5有效信息+删时间；D3-0、zero-update、null、control和seed不占编号。
- 步骤3、4都从冻结H2开始，不继承A2/A3候选权重。Step4正式训练前必须通过第0轮和null一步等价审计；浮点与量化分别校验。
- A3-1是自然遍历+BCE；A3-2是四个互斥层抽样+BCE重要性校正。两者均使用`1e-5`，首轮不开EMA/QAT。
- 每个预定评估点同时产生浮点和项目既定int8结果，正式选型以int8为准。
- A4-1/A4-2必须分别与同维、同参数、同预算null比较；A4-3配32维重训对照；A4-4/A4-5只有冻结入口条件满足时启动。
- Step4正式结果已冻结：A4-1和A4-3停止，A4-4/A4-5不补跑；Step5不占A5编号，冻结A4-2 epoch8/阈值0.40。
- Step6只补seed43/44及匹配null，固定Step5 epoch/阈值，不增加删维、扩宽、损失、特征、训练轮数或seed。
- Step7固定A4-2 seed42 epoch8/0.40，同时一次性报告Validation与Test；保留Step6历史失败记录，不以独立数据换模型或阈值。
- 最终工程决策以整体IoU、Precision和误报控制优先；文档和发布必须同时披露Test高密度Recall下降，不得改写原严格门槛结果。
- A3失败不自动封死A4，但失败A3权重不得成为A4起点或晋级模型。
- 连续两次有证据指导的受控尝试仍无稳定收益时停止对应分支。

## 结果与文档

- 每一步的代码、协议、日志和结果放在`../seven_step_improvement_study/`对应目录。
- 正式H2-pro运行代码、冻结权重、模型卡和精简Validation/Test结果放在本目录；部署只能从本目录取文件。H2-plus旧正式材料留在Step7中间模型，不回填正式注册表。
- 根目录`../../5_model_evaluation/`只保留 A/B/C 三类，按`01_metric_comparisons/`、`02_model_records/`、`03_rules_and_protocols/`保存正式评测；跨模型比较只输出 CSV，不生成图表。模型源数据放在`02`的对应 `evaluation_id` 目录，不保存生成代码、缓存或 checkpoint；历史材料统一进入根级本地`../../archive/5_model_evaluation/`，不得上传 GitHub。
- 正式训练每轮保存进度，预定评估轮保存完整Train-dev浮点/int8指标；续训写入新目录并保留来源及完整状态。
- 步骤6使用10,000次密度分层、序列级配对bootstrap；区间跨0时不能宣称稳定提升。
- 修改编号、实验顺序、密度协议、门槛或状态后，同步更新研究入口、七步README、PLAN和服务器说明。
- 上传包必须包含最新说明、清单和SHA-256，并进行完整解压校验。
- 只有用户明确要求时才提交Git；不得擅自推送、拉取、合并或变基。
