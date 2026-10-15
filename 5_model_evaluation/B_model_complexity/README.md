# B 模型复杂度

B01/B02 为参数与静态张量，B03 为条件前向结构连接 MAC，B04 为理论非零 MAC/突触 AC 及未加权和，新增 B05 为持久状态内存。旧 H2/H2-pro 计数保留；B05 完整清单待补，因此现行 v2 的 B Common 尚未完成。

[B 类规则](03_rules_and_protocols/metric_definitions_and_evaluation_rules.md)明确实际执行与理论计数的区别。EV-SpSegNet 论文值保留 paper_reported。只读取逐指标 CSV，不手工回填缺失测量。
