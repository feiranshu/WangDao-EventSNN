# Pd/Fa 工程口径 v2

协议 `evuav-localization-half-open-v2`。统一实现为 `implementation/software/wangdao/localization_v2.py`；旧 metrics.py、phase1_metrics.py 及历史数值保留。

## 输入与公式

使用冻结原始事件、对齐概率和阈值。时间为原始 float64 毫秒，不先取整数，不使用旧 ev_loc 整型时间。t0=首事件时间；t_end=t0+floor((t_last−t0)/100 ms)×100 ms；评分 [t0,t_end)。每 50 ms 左闭右开：t0 计入，内部边界归右侧窗，t_end 不计。

- Pd 分母：每个非零目标身份在每个有其事件的窗口计一次。目标事件被正确预测的比例≥0.0001 时计一次检出。保留这一宽松阈值；目标事件数≤10000 时，一个命中就可能达标，不代表框定位准确率。
- Pd=检出的目标窗数/全部目标窗数。无目标则 null，不返回 1，不纳入宏平均，同时报告有效数量。
- Fa 分子：每窗背景误检事件投影为 346×260 布尔占用图，计 8 连通域；同像素重复事件不会因 uint8 溢出而消失。
- Fa 分母：完整 50 ms 窗口数×346×260，包含无事件的完整窗。单位 component/pixel/window，不是分类 FPR 或虚警/秒。
- 跨序列先相加整数分子/分母再求比率。保留 targets、correct_targets、false_components、frame_denominator、scored_localization_events、excluded_events。

当前 Test 的 1850 个闭合 100 ms 窗口对应 3700 个 50 ms 窗口曝光，旧 Fa 分母是 3676。不能只修分母后声称 v2：时间量化、边界和溢出也可能改变分子，必须从冻结预测重算。

## 离线重算

在第 5 板块目录执行：

```powershell
python ../1_H2_model_improvement/implementation/software/scripts/recompute_localization_v2.py frozen_predictions_manifest.json --evaluation-root . --output A_model_accuracy/02_model_records/formal/<evaluation_id>/<validation|test>/event_level/localization_v2.json
```

Manifest 顶层为 record、sequences。record 按总协议填写身份、comparison 与 provenance；scope_id=`first-event-complete-100ms-prefix-v1`，numeric_path、threshold、causality 与真实模型相符。sequences 每项含 sequence_id、events_file、probabilities_file、probabilities_sha256，路径相对 Manifest。

事件 NPZ 的 ev 含 t/x/y/label/name；预测 NPZ 含一维 probabilities，长度为完整序列或恰好闭合前缀，不能仅提供候选事件概率。脚本核对索引事件哈希、预测哈希和序列全集，生成逐序列计数与曝光；不加载模型、不选阈值、不覆盖已有结果。预测文件与权重/配置的归属由冻结导出清单建立，不能凭文件名推断。

v2 记录额外含 localization_counts_file，指向已哈希的逐序列 JSON；每项含 sequence_id、source_event_sha256、prediction_sha256、start_ms、end_ms、counts。旧 event_level/summary.json 原样保留。

## EV-SpSegNet 服务器迁移

20261008 包仍按完整序列及旧 Pd/Fa 评测。升级时保留官方模型预测路径，在输出后统一裁剪评分前缀，调用本 v2 定位器及冻结轨迹适配器；B 同样统计对应前缀。使用未来信息的模型标 offline_noncausal。

回传 event、localization、track、complexity、state_memory 的 v2 元数据、逐序列统计和证据哈希。导入器在复制前检查全部记录，旧包仅修改 protocol_id 不足以合格。保留旧包追溯，不自动重跑服务器任务。
