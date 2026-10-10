# C 部署效率与硬件成本

C 是板端正式模型的必测项，软件正式模型不强制。C07 Runtime 同时接受软件与板端记录，当前作为软件补充项，不改变 A+B 准入要求。当前没有板端正式模型，因此 C01–C06 只建立指标说明；C07 已为现有软件正式模型建立 Validation/Test 待测 CSV。

C01/C02 为内核与端到端延迟分布，C03 为吞吐与实时性，C04 为功耗，C05 为每序列能耗，C06 为硬件资源和工作频率，[C07 Runtime](01_metric_comparisons/C07_runtime/README.md) 为处理一条完整序列的模型推理运行时间，单位 `ms/sequence`，越低越好。C04a、C06a–C06e 使用 Common；其余使用 Validation/Test。Runtime 依赖实际平台与执行实现，因此放在 C，不属于 B 的模型静态复杂度或运算量。

所有结果必须来自固定平台、固定 bitstream/软件版本和一致输入流。C07 的计时边界、与 C01/C02 的区别及源字段见 [完整序列 runtime 协议](03_rules_and_protocols/runtime_measurement_protocol.md)。不同设备、数值精度或计时边界的结果须注明，不能直接作为同条件加速比。

新增板端正式配置时，先在根注册表中使用独立的 board `evaluation_id`，再将 `common/summary.json`、`validation/summary.json`、`test/summary.json` 放入 `02_model_records/formal/<evaluation_id>/`。软件 C07 也使用该目录结构，只需 Validation/Test 源记录。运行 `rebuild_evaluation_tables.py --category deployment` 生成对应 CSV；源值缺失标为 `pending`。板端强制项（含 C07）缺失会使覆盖率检查失败；软件 C07 暂不参与强制完整性判定。
