# 评价标准 v2（2026-10-16）

本次统一定义、计算入口和准入检查，不改变冻结模型、阈值或历史测量数值。`evaluation_coverage.csv` 表示当前 v2 准入状态；2026-10-10 的审计仅描述旧规则。

## 优先修订

1. A06/Pd、A07/Fa 统一使用 `evuav-localization-half-open-v2`：原始浮点毫秒、首事件锚点、半开窗、完整曝光分母、布尔虚警图、无目标 null。旧评测器保留用于历史重现。
2. B04 是非零连接理论运算量，不证明执行后端已跳零。MAC/AC 分开保存；未加权和不作为跨位宽/架构的速度或能耗排名。
3. A/B 评分范围固定为首事件锚定的完整 100 ms 前缀。H2 是非因果历史参照，H2-pro 是因果模型；相同评分事件不等于相同可用信息。旧全序列 EV-SpSegNet 包不得直接导入。
4. `complete` 必须通过数值、来源哈希、模型/数据身份、协议和适用证据检查。空值、论文值、`legacy_only`、`unverified` 均不算完成；不能手工改 CSV 状态绕过检查。

## 指标与准入

- A01–A22 保留。旧 A06/A07 标记 `legacy_only`，从冻结预测重算 v2 后才可完成。A04 主名称为 Target-event Recall，ACC 仅为论文别名。
- B01–B04c 保留；新增 **B05 持久运行状态内存**，软件准入必需。B02 静态张量、B05 跨窗口状态、C08 峰值运行内存分别统计。
- C01–C07 保留；新增 **C08 内存峰值、C09 覆盖率/积压、C10 部署数值一致性**。板端准入要求 A+C01–C10；软件 C07/C08/C09 为补充。
- 已有 H2-pro−H2 IoU 配对 bootstrap 区间关联到 A/01/uncertainty；其他置信区间待计算，不填 0。
- S01/S02 仍用于 SNN 研究，补各层发放率、状态更新及实际 T 分布；未冻结候选不进入正式注册表。

C08b 在纯 CPU 软件配置可说明不适用；板端设备内存必须测量或用完整清单证明零占用。必需项的 not_applicable 不自动豁免准入，适用性变化须有证据并版本化，不能用空值冒充完成。

## 比较与来源契约

`comparison_contract.json` 保存冻结索引哈希、各划分 24 条序列及文件哈希、已审查历史记录的规范化 JSON 哈希、允许的新协议。历史白名单仅识别原记录，不能给新输出贴旧标签。

CSV 新增 source_sha256、verification_status、comparison_scope、causality。source_file 相对于本目录。不同 scope 不进入同一排名；不同因果性只能作标注清楚的历史准确度参照。时间比较还须固定设备、后端、数值精度、线程、时钟、batch 和输出覆盖，变更配置使用独立 evaluation_id。

新源记录采用 `evaluation-record-v2`：evaluation_id、platform_id、split、protocol_id、comparison、provenance。comparison 包含 scope_id、causality、numeric_path、threshold、sequence_ids；provenance 包含 weight_sha256、config_sha256、index_sha256，以及记录目录内 artifacts（path、sha256）。哈希必须来自实际文件。

| 类型 | 新协议 |
| --- | --- |
| 事件准确度 | evuav-event-closed-prefix-v2，沿用 working_point 结构 |
| Pd/Fa | evuav-localization-half-open-v2，独立 localization_v2.json |
| 轨迹 | evuav-observable-event-track-hota-closed-prefix-h2pro-v1 |
| 算法复杂度 | nonzero-connection-closed-prefix-v2 |
| 持久状态 | persistent-state-inventory-v2 |
| 部署 | deployment-measurement-v2；C07 另声明 full-sequence-runtime-v2 |

字段和哈希检查证明身份与记录内部一致性，不证明传感器校准或真实执行过程；原始计时、功率轨迹、计数和执行日志须留存。

## 入口

在本目录执行：

```powershell
python ../1_H2_model_improvement/implementation/software/scripts/rebuild_evaluation_tables.py --root . --allow-incomplete
python ../1_H2_model_improvement/implementation/software/scripts/validate_evaluation_coverage.py --root .
```

第二条在必需项缺失时退出 1，这是预期准入行为。重建只读源记录，不运行模型。当前新版 Pd/Fa、B05 清单和部署测量尚待补齐，因此不能继续沿用旧版 overall complete。

本次不重新推理 Validation/Test、不修改阈值、不制造测量数据。冻结预测的协议纠错不用于选型；重复计时按预注册方案执行，不能根据受保护数据挑最快结果。
