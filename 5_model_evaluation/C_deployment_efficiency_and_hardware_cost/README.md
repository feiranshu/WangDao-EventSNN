# C 部署效率与硬件成本

C 是板端正式模型的必测项，软件正式模型不强制。当前没有板端正式模型，因此只建立指标说明，不生成空数值 CSV。

C01/C02 为内核与端到端延迟分布，C03 为吞吐与实时性，C04 为功耗，C05 为每序列能耗，C06 为硬件资源和工作频率。C04a、C06a–C06e 使用 Common；其余使用 Validation/Test。所有结果必须来自固定平台、固定 bitstream/软件版本和一致输入流。

新增板端正式配置时，先在根注册表中使用独立的 board `evaluation_id`，再将 `common/summary.json`、`validation/summary.json`、`test/summary.json` 放入 `02_model_records/formal/<evaluation_id>/`。运行 `rebuild_evaluation_tables.py --category deployment` 后才生成对应 CSV；源值缺失会标为 `pending`，覆盖率检查返回失败。
