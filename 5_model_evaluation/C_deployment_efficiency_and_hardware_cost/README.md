# C 部署效率与硬件成本

C01/C02 延迟、C03 吞吐/实时性、C04 功率、C05 能耗、C06 资源、C07 runtime，新增 C08 内存峰值、C09 覆盖率/积压、C10 部署数值一致性。板端要求 A+C01–C10；软件 C07/C08/C09 为补充。

[测量标准 v2](03_rules_and_protocols/metric_definitions_and_evaluation_rules.md)固定公式、边界、单位和证据要求；[C07 协议](03_rules_and_protocols/runtime_measurement_protocol.md)固定完整输入及 5 次预注册重复。当前无正式板端注册配置，已有软件模型测量保持 pending。

源记录位于 02_model_records/formal/<evaluation_id>/<common|validation|test>/。C07 优先使用 runtime.json，其他项使用 summary.json；不同 scope 分文件。模板在 03_rules_and_protocols，不能作为实测。重建器校验后生成 CSV，缺项或 unverified 不通过准入。
