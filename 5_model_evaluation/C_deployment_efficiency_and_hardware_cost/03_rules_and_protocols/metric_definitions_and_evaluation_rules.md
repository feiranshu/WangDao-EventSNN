# C 类部署测量标准 v2

协议 deployment-measurement-v2。按模型、权重、平台、后端、线程、精度、时钟、传输、batch 分 evaluation_id。保留真实采样及哈希，不以算法计数推算实测。metric_context.<编号> 的 definition_id 固定 `<编号>-v2`，raw_artifact 必须在 provenance.artifacts 中。

## 时间与实时性

| 指标 | 固定边界及统计 |
| --- | --- |
| C01 mean/P50/P95/P99 | 指定 PL 核开始执行至完成，记录 kernel_name、sample_unit、timer_resolution_ns；host MMIO/XRT 提交等待不冒充纯核时间。 |
| C02 mean/P50/P95/P99 | 每事件从采集/设备接收到最终应用输出，包含等窗、排队、传输、模型和必需后处理；sample_unit=event，timestamp_domain 及同步校准方法明确，includes_window_wait/queue=true。无输出也记录 C09。 |
| C03a sequence/s | 完整输入序列完成数/测量墙钟秒数，注明并发、流水线、预填充及排空。 |
| C03b event/s | 实际处理事件数/同一墙钟秒数；输入率另列 offered_event_rate，丢弃不算处理成功。 |
| C03c RTF | 总处理耗时/总输入观测时长，越小越好；不平均逐序列比率。≤1 仍需检查尾延迟/积压。 |
| C03d 超时率 | 超过冻结 deadline_ms 的应输出事件数/全部应输出事件数；到期未输出计超时，适用范围与有效输出前缀一致。 |
| C07 runtime | 完整原始序列的模型流水线运行时间，见独立 runtime 协议。 |

100 ms 等窗+2 ms 运算，对窗首事件约 102 ms 响应。C01/C07 不含采集等待，不能代替 C02。C03 context 必含 offered_event_rate、deadline_ms、run_mode（live/real_time_replay/as_fast_as_possible），不同模式分开比。持续吞吐测试的负载、时长、P95/P99、超时率及队列增长上限预先冻结。

分位数使用线性插值，从合并原始样本计算，不平均各序列 P95。报告样本单位、数量及缺失数；样本不足以稳定估计 P99 时说明。冷启动单报。

## 功率、能量、资源

C04a 空闲功率、C04b 运行功率是指定区间时间加权平均；C04c=运行功率−同硬件/时钟/外设的空闲基线。context：power_domain、sensor、sampling_hz、idle_configuration。整板、PS、PL、墙上等不同供电边界不直接比；保留温度、散热、时钟、校准日志。

C05a=Σ∫P_run(t)dt/序列数；C05b=Σ∫[P_run(t)−P_idle(t)]dt/序列数。context 增加 integration_interval，保留时间戳和功率轨迹，默认时间梯形积分。不得不核对权重就用平均功率×平均耗时；覆盖、采集等待和积分区间一致。扣基线产生小负值保留符号并报告噪声/不确定性，不裁成 0，也不宣称负能耗收益。

C06a LUT、C06b FF、C06c BRAM36K 等效量、C06d DSP 使用实际器件 post-route 报告，并列容量/利用率。BRAM36K=BRAM36+BRAM18/2，可为半整数；LUTRAM 单列不重复加总。C06e 是实际运行 MHz，并列目标约束、WNS/TNS、保持检查和配置证据。context：device_part、report_stage=post_route、timing_met=true；未闭合时序的目标频率不算合格。

## 新增指标

| 编号 | 定义 |
| --- | --- |
| C08a host_peak_rss_bytes | 测量期间主机/PS 进程 RSS 峰值，含权重、状态、输入输出、工作区。 |
| C08b device_peak_allocation_bytes | 同期设备缓冲/DDR/CMA 分配峰值；GPU 区分 allocated/reserved，FPGA 片上 BRAM 归 C06。 |
| C09a processing_coverage | Σprocessed_events/Σinput_events。 |
| C09b output_coverage | Σoutput_events/Σinput_events，按有对齐原生输出的独立事件计。 |
| C09c dropped_event_rate | Σdropped_events/Σinput_events，未闭合 tail_events 单列。 |
| C09d max_backlog_events | 全程待处理队列最大事件数，零积压也需观察证据。 |
| C10a exact_agreement | 对齐输出与冻结软件金标准逐元素完全相同的比例。 |
| C10b max_abs_error | 对齐输出的最大绝对误差。 |
| C10c rmse | sqrt(Σ误差²/对应元素数)。 |
| C10d prediction_agreement | 相同冻结阈值下标签一致率。 |
| C10e overflow_rate | 溢出操作数/受监测运算数，声明检测位置与分母。 |
| C10f clipping_rate | 饱和/截断操作数/受监测运算数，与溢出/分类错误分开。 |

C08 context：memory_domain、sampling_method、includes_baseline。C09：coverage_counts_file、queue_observation_method。逐序列 JSON 含 sequence_id、input_events、processed_events、tail_events、dropped_events、output_events、max_backlog_events；input=processed+tail+dropped、output≤processed，输入数与冻结索引一致。正常输出背景分数的路由拒绝事件属于已处理。

C10 context：reference_evaluation_id、reference_weight_sha256、tensor_stage、denominator、tolerance；注明数值格式/尺度，只比较同一阶段。exact-int8 要求对应整数输出逐位相同；近似路径使用预先冻结容差，同时报告原始误差、标签一致率、同范围 A 指标差值。零溢出须有计数证据，成本下降不能抵消精度损失。

## 记录

summary.json 使用总标准 v2 身份/comparison/provenance。measurement 必含 host_model、backend、precision、software_versions、threads、batch_size；板端另含 transport=MMIO/XRT、bitstream_sha256、clock_mhz。每项都有 metric_context 和原始证据。

C07 优先用单独 runtime.json，避免与同前缀 summary.json 混用范围。不可测保持 pending；不适用说明原因；不得填虚构 0。不同 scope、设备或边界分别比较。
