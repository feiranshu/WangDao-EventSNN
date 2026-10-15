# C07 完整序列运行时间 v2

记录协议 deployment-measurement-v2；runtime_protocol_id=full-sequence-runtime-v2，保留 ms/sequence。旧 v1 不得只换标签升级，当前尚无合格实测。

## 固定边界

起点：完整原始事件已在主机内存、模型加载并预热、状态复位，即将调用完整流水线。终点：输入处理和收尾完成，冻结协议要求的原生输出在主机可读，设备任务和必要传输全部结束。

计入特征/编码、Proposal、实际触发的 Refiner、全部 SNN 时间步、状态更新、调度、模型必需解码、MMIO/XRT 提交/等待/搬运。不计加载、预热、文件读取、真实采集等待、评测器额外聚类/跟踪、准确度计算和写日志。用单调墙钟，开始前清空任务，结束等待异步执行和输出拷回，不能只测提交。

序列 batch=1、串行，每次每序列复位，不复用状态或缓存预测。固定线程、内部批处理、设备时钟和实现。H2-pro 只发布闭合 100 ms 窗，尾段保留未发布，不能为计时补窗/截断输入。scope_id=full-input-sequence-v1；A/B 同前缀准确度不证明完整输入输出覆盖。

## 预热、重复和统计

预热用预先固定的 synthetic/Train-dev 输入，iterations≥1。固定 5 次完整划分遍历，每次全部 24 序列；次数、顺序和配置在查看耗时前锁定。不挑最快轮次、不按 Test 调参数。准确度冻结验收仍只做一次；重复纯计时不改变模型、阈值、样本或参与选型。

主值=Σ每序列每重复 runtime_ms/(序列数×5)。源记录另报全部样本中位数、P95、标准差及每序列 5 次均值/波动，区分序列差异和重复噪声。P95 从原始样本合并计算，不平均序列 P95；冷启动另记。

汇总字段为 runtime_ms_per_sequence、runtime_p50_ms、runtime_p95_ms、runtime_std_ms；标准差采用总体标准差（ddof=0），分位数采用线性插值。重建器会与原始 CSV 对账。

## 必需记录

优先使用 02_model_records/formal/<evaluation_id>/<validation|test>/runtime.json，summary.runtime_ms_per_sequence 存主值。原始 CSV 字段：sequence_id,repeat,runtime_ms,input_events,processed_events,tail_events,dropped_events,output_events。

repeat=1–5，每序列每次恰好一行。计数非负整数，input=processed+tail+dropped、output≤processed。另保留 t0、最终 watermark、闭合窗数、有效输出前缀和原始事件哈希的覆盖日志，不能隐藏尾段。

填写 evaluation-record-v2 身份/comparison/provenance。measurement 另含 timer=monotonic_wall、synchronization（具体等待/拷回方法）、state_reset_per_sequence=true、repeats=5、warmup={input_role,iterations}；平台和板端字段遵守 C 总协议。metric_context.C07 的 definition_id=C07-v2、raw_artifact=runtime_per_sequence.csv，CSV 必须列入 artifacts 并哈希。

重建器核验全集、5 次重复、样本/覆盖恒等式和主均值；缺协议/平台/证据为 unverified，不自动补实测标签或协议。窗口 CPU benchmark、论文 runtime、8 s 输入时长不能填 C07。

C01 是纯核时间，C02 是含等窗/排队的事件响应，C03 是吞吐/实时性，C07 是预载完整序列模型耗时，不能相加或无条件取倒数互换。不同输出覆盖和因果性必须随结果披露。
