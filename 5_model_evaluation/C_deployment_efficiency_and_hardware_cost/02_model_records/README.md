# 模型源记录

`formal/` 按稳定 `evaluation_id` 保存正式模型；`external_references/` 保存论文或外部仓库参照。比较 CSV 由这些源记录重建。

C07 Runtime 的软件与板端源记录均放在 `formal/<evaluation_id>/<validation|test>/summary.json`，主字段为 `summary.runtime_ms_per_sequence`。逐序列计时明细与平台配置放在同一划分目录，汇总方法及 JSON 字段示例见 [runtime 测量协议](../03_rules_and_protocols/runtime_measurement_protocol.md)。未测量时不创建伪实测记录；重建器会生成空值 `pending` 行。
