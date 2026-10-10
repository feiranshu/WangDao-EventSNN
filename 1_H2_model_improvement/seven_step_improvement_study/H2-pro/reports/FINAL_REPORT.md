> 版本更新（2026-10-10）：用户已将 H2-pro 晋级为现行正式最终模型，H2-plus/A4-2 转为 Step7 中间模型。下文“原部署入口不变”“研究教师候选”等为晋级前的实验交接状态，不能作为当前部署说明。

# H2-pro Stage 1 因果 Refiner 交接（2026-10-10）

## 1. 已完成与未完成

在不修改 `implementation` 冻结模型的前提下，新建 `H2-pro` 独立工程。`src/stream_state.py` 提供 100 ms 水位线流式推理；`src/causal_feature_builder.py` 重放原有四尺度计数及另外 35 列；`src/motion_coherence.py` 实现 C1 Proposal 运动和 C2 LMEC。`run_stage1.py` 使用固定的 36→64→32→4 Refiner、同一 A4-2 初始化、同一 seed42、同一 10 轮预算完成 C0/C1/C2 的本地训练；第 0/1/2/5/8/10 轮均保存 float、exact-int8、Train-dev 曲线及逐序列结果。三组训练轮次和评估累计约 390 秒，不含数据缓存生成。

按 Train-dev 选定并冻结 `H2-pro-Causal-ANN-v1`：C1、epoch10、exact-int8、阈值 0.35。C2-LMEC 有可分性，但未通过保护门槛。Validation/Test 在冻结后各运行一次。未测硬件 SRAM、ASIC 延迟/能耗，也未转换为 SNN；原 A4-2 部署入口不变。

## 2. 因果性证据

旧第 3 列使用全序列最后时间戳，是非因果列。新第 3 列只取前两个已闭合 100 ms Proposal 热图，经 ±1 网格/窗口的固定 3×3 速度搜索，计算 `uint8(rint(255·max_v min(H[w−1,c−v],H[w−2,c−2v])))`。C0 置零；C2 从前两个已完整 25 ms 桶生成局部活动一致性分数。推理仅接收 `x/y/t/p`，拒绝迟到和乱序事件；标签与目标 ID 仅在训练及评测中读取。

`CausalPipeline.reset(t0)` 后持续 `push(events, watermark)`，只在窗口右边界到达时发布；空窗口推进热图历史。末尾不足 100 ms 的事件由 `finalize()` 报告数量而不发布。前 100 ms 原始事件缓存及两张 Proposal 热图组成有界时间历史；全局窗口号不在 8 秒截断。12 项 H2-pro 测试通过，覆盖未来追加不变性、随机分片一致性、旧 Proposal 一致性、跨窗口 2/5/20/50 ms 计数、边界/空窗/>8 秒、其他 35 列一致性及初始化输出等价。初始 C0/C1/C2 的 float 和 int8 概率最大差均为 0。

## 3. 算法对照

所有新结果为 **同一完整 100 ms 窗口前缀**。旧历史 Test IoU 86.8931% 使用整序列口径，不应与本表混比。高密度定义为整序列事件率 ≥20,000 events/s。

| 角色及模型 | 阈值 | IoU | AP | Precision | Recall | Pd | Fa | 高密度 Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Train-dev 旧 A4-2 | .40 | .786008 | .931270 | .909973 | .852284 | .843531 | 1.2922e-5 | .566967 |
| Train-dev C0 epoch10 | .35 | .786955 | .931767 | .906752 | .856250 | .844032 | 1.3239e-5 | .578845 |
| Train-dev C1 epoch10（冻结） | .35 | .792615 | .934002 | .913259 | .857143 | .844534 | 1.2635e-5 | .582326 |
| Train-dev C2 epoch10（诊断点） | .40 | .787663 | .931674 | .914321 | .850433 | .838516 | 1.2089e-5 | .562769 |
| Validation 旧 A4-2 | .40 | .866927 | .961438 | .945192 | .912814 | .901786 | 7.416e-6 | .460502 |
| Validation C1 冻结 | .35 | .878318 | .963589 | .949168 | .921671 | .902423 | 7.056e-6 | .518782 |
| Test 旧 A4-2 | .40 | .868796 | .969564 | .918665 | .941192 | .936775 | 1.110e-5 | .763609 |
| Test C1 冻结 | .35 | .871383 | .970822 | .919882 | .942947 | .936195 | 1.099e-5 | .768746 |

固定阈值 .40 的 Train-dev IoU：C0 .787869、C1 .792743、C2 .787663；对应高密度 Recall 分别 .560311/.563076/.562769。阈值不能只按 IoU 选择：v1 原规则为 C1 选出 .45，但其高密度 Recall .547102，低于同口径旧 A4-2 的 .566967。查看这项 Train-dev 结果后制定并记录 `selection_protocol_v2.json`，在 **读取 Validation/Test 前** 加上相对旧 A4-2 的绝对保护线，最终选 .35。这是明确的开发集后验修订，不得宣称为实验前预注册规则。

Train-dev C1 对 C0 的 IoU 差 +.005660，密度分层序列级配对 Bootstrap 95% 区间 [.000934,.009727]；高密度 Recall 差 +.003481，区间 [−.001916,.005450] 跨零。C1 对旧 A4-2 的 Train-dev IoU 差 +.006607，区间 [.004488,.008499]。Validation IoU 差 +.011391，区间 [.002100,.020153]；Test 差 +.002587，区间 [−.000721,.006286]，因此 **不能声称 Test 上存在稳定 IoU 增益**。Validation 19/24 序列改善，Test 12/24 序列改善。C2 在 v2 下无合格工作点，不晋级。

## 4. 运动连续性

评测时按目标 ID 与 100 ms 窗口构成 object-window；窗口至少有 3 个目标事件才计入，任何该目标事件过阈值即检出。`coverage=检出窗口/合格窗口`；相邻合格窗口由检出转漏检记一次断裂。目标 ID 未进入在线特征。

Train-dev 旧 A4-2/C0/C1 的 coverage 为 .875374/.874377/.873380，断裂数 38/37/39。Validation 旧/C1 为 .924249/.922979，断裂 52/48；Test 为 .959428/.958269，断裂均 44。C1 在 Test 总体 IoU 上略升，但目标窗口覆盖率略降，不能概括成连续性全面改善。C2 的 Train-main 稀疏正例 LMEC 中位数为 0，提示稀疏小目标失配风险；目标相邻 100 ms 位移中位数约 4.05 px。逐序列定位见 `acceptance_validation.json`、`acceptance_test.json` 与 `acceptance_uncertainty.json`。

## 5. 计算与存储

Refiner 仍为 4480 权重、4480 dense MAC/候选事件。C1 增加两张 34×44 float32 Proposal 热图（共 11,968 字节）、每窗 9 个速度假设的热图查找/最小值/最大值计算；原四尺度计数继续保存前一窗原始事件，时间跨度有界但事件数没有固定上限。Train-main 两条序列、158 个闭合窗口的 Python CPU 测量：C0 均值 43.09 ms、p95 178.08 ms；C1 均值 93.68 ms、p95 185.53 ms；C2 均值 72.94 ms、p95 155.18 ms。最大候选数 2447/窗。C1 的 p95 已超过 100 ms，故**未证实单线程实现能持续跟上 10 Hz**。100 ms 窗口采集等待与上述计算时间是两部分，尚未实测并行排队与硬件 PPA。

## 6. 最终选择

C1 满足因果、模型宽度、冻结 Proposal、Train-dev v2 绝对保护线，并已作为**研究用因果 ANN 教师候选**冻结。LMEC 不晋级；原 A4-2 的历史科研判定和部署状态不变。浮点 C1 在 Train-dev .35 的 IoU .791899，exact-int8 为 .792615；候选概率浮点/int8 平均绝对误差 .001410、p95 .004490，最大单点误差 .476728，后者是需要继续排查的量化尾部风险。冻结选择使用 exact-int8 结果。

Validation/Test 既往在原研究中已被查看，故这是锁定方案后的回顾性验收，不能称为完全未接触的独立 Test。Test 的 bootstrap 区间跨零，现阶段不建议用“稳定精度提升”替换原部署模型。教师可用于下一阶段 ANN→SNN 研究，但应保留这一不确定性。

## 7. 复现入口

本机使用 `D:\software\anaconda\envs\evuav\python.exe`；依赖见 `requirements.txt`。从 `H2-pro` 目录执行：

```powershell
$py = 'D:\software\anaconda\envs\evuav\python.exe'
& $py run_stage1.py preflight
& $py -m unittest discover -s tests -v
& $py run_stage1.py prepare --role train_main --cache cache
& $py run_stage1.py prepare --role train_dev --cache cache
& $py run_stage1.py train --cache cache --output results/new_seed42 --device cpu
& $py evaluate_historical.py --cache cache --output reports/new_historical_train_dev.json
& $py select_stage1_v2.py results/new_seed42 --historical reports/new_historical_train_dev.json
```

输出目录为追加式，复现时改用新的目录名。现有完整训练在 `results/formal_seed42_local_20261010/`；`train_dev_selection.json` 保留 v1 选择，`train_dev_selection_v2.json` 保留修订后的选择。冻结模型在 `frozen/H2-pro-Causal-ANN-v1/`；`run_stage1.load_trained_float()` 加载 float 教师，`src.stream_state.CausalPipeline(...,initial_refiner=False)` 加载 exact-int8 教师。不要重复运行 `acceptance_once.py validation/test` 作为调参循环。

关键 SHA-256：索引 `7968011347d9d6291a94e183b4268a84549301003523bca7b4b185a5ca36a397`；特征/训练协议 `57d11d912d45a16bc8b4edf1f467d56537d071835a3ca2256e150eb60e6835b3`；v2 选择协议 `ea6a94e59fd88dfdf174f338c05f28479e7b144c2a83fefdc1a0615854b39dd6`；冻结 float `dd4b81fb5cef1b5e5f120afbc5e30c9b7d3b0e5f48cd275a0caf0534e2112c91`；冻结 exact-int8 `6e9221829459825413c58604a40da3582137a555aaa508d84d9da8325aa5368d`；验收锁 `e0cc8154e20b4475a80adf63d0ecf7d891721518d3db2865edf6d5f688fc06d9`。源 Proposal/旧模型权重哈希在 `model_manifest.json`。

## 8. 阻塞与后续建议

当前没有阻断本地训练和软件验收的环境问题；无需服务器训练包。尚未实测持续 10 Hz 延迟、最坏事件缓冲上限与硬件成本；Test IoU 改善的序列级区间跨零，量化误差存在少数大离群点。下一阶段若使用此教师，优先做实时计算优化、量化尾部样本审计和独立场景复验；这些应作为新协议开展，不改动本次冻结结果。
