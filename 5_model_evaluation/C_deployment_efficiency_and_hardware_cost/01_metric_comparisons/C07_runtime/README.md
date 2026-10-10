# C07 Runtime（完整序列模型推理运行时间）

本目录只比较 `C07`，单位为 `ms/sequence`，越低越好。数值为同一数据划分内所有完整序列模型推理耗时的算术平均值，Validation 与 Test 分别保存；当前冻结划分各含 24 条完整 8 s 序列。8 s 是输入记录的时长，不是模型运行时间。

适用于软件 CPU/GPU 与板端配置。它覆盖完整模型路径，例如 H2/H2-pro 的 Proposal、实际触发的 Refiner 及其内部调度；不能只计一个 patch、窗口或单个加速核。完整序列指完整输入记录；H2-pro 按冻结协议只发布已闭合的 100 ms 窗口，须同时记录处理前缀与未发布尾段。精确计时边界见 [测量协议](../../03_rules_and_protocols/runtime_measurement_protocol.md)。

`validation.csv`、`test.csv` 为正式比较入口，模型顺序由根目录 `model_registry.csv` 决定。当前行仅为待测占位，`value` 留空、`status=pending`；缺值不得填 0。软件 C07 暂作补充项；板端 C07 属于 C 类强制项。

源字段为同一划分 `summary.json` 中的 `summary.runtime_ms_per_sequence`，协议 ID 为 `full-sequence-runtime-v1`。运行 `rebuild_evaluation_tables.py --category deployment` 重建表格，不直接手工填写比较 CSV。
