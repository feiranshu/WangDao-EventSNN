# S0 冻结契约（2026-10-11）

本文件是第二代实验的只读基线，机器核对结果见 [`audit.json`](audit.json) 和 [`train_dev_errors.json`](train_dev_errors.json)。上游正式模型、数据及评测文件未改动。

## 项目与来源

根目录 0 保存原始 H2；1 保存七步 ANN 改进、H2-pro 冻结模型及训练证据；2 为 KV260 上板准备；3 的 `old-version/` 保存第一代 SNN；4 为部署；5 为正式 A/B/C 评测。`archive/`、`project_proposal/`、`tmp/` 是本地历史/临时材料，不是本阶段正式入口。第二代以 `3_ANN_to_SNN/PLAN.md` 为准，旧计划的 S0-P/S0-R 阶段编号不沿用。

| 冻结项 | 核定值 | 主要证据 |
| --- | --- | --- |
| 正式 ANN | H2-pro-Causal-ANN-v1，C1-proposal-motion，seed42，epoch10 | `1_H2_model_improvement/implementation/results/final_H2-pro/deployment_decision.json` |
| 数值与阈值 | exact-int8 replay，0.35；研究用 float 教师另存 | 冻结模型 `model_manifest.json`；`audit.json` |
| 权重 SHA-256 | exact-int8 `6e9221829459825413c58604a40da3582137a555aaa508d84d9da8325aa5368d`；float `dd4b81fb5cef1b5e5f120afbc5e30c9b7d3b0e5f48cd275a0caf0534e2112c91` | `audit.json` 实算 |
| 数据索引 SHA-256 | `7968011347d9d6291a94e183b4268a84549301003523bca7b4b185a5ca36a397` | `1_H2_model_improvement/implementation/data/index.json`；`audit.json` 实算 |
| 数据角色 | Train-main 79；Train-dev 10；HW-calibration 10；Validation 24；Test 24，互不重叠 | `audit.json` |
| 输入 | 346×260，事件 `x/y/t/p`；训练/评测的标签及 ID 不进入推理；候选特征为 `uint8[36]` | `src/stream_state.py` 与冻结协议 |
| 原 Refiner | 36→64→32→4；四原型，目标/背景各二，类内取最大后计算目标概率 | `model_manifest.json`、`run_stage1.py` |
| 时间与发布 | 独立序列首事件为 `t0`；100 ms 完整闭合窗口；水位线达到右边界才发布；末尾不完整窗口丢弃 | `protocol.json`、`src/stream_state.py` |

源文件路径以上游项目根目录为准，SHA 及准确路径由 `audit.json` 提供。冻结目录内 `model_manifest.json` 的 `acceptance_status: not_yet_evaluated` 是验收前快照；当前正式身份由后续 `deployment_decision.json` 确定。`acceptance_lock.json` 内部分旧绝对路径亦为历史快照；S0 按其内容哈希核对当前 `seven_step_improvement_study/H2-pro/` 源文件。

## 输入与输出行为

冻结 H2-pro 使用 `reset(t0)`、`push(events, watermark)`、`flush_completed_windows()`、`finalize()`。事件必须按时间排序，坐标及极性合法，`push` 不接受迟到、乱序或尚未越过水位线的事件。窗口内 Proposal 构造 88×68 的填充细格计数并形成 34×44 热图，阈值 0.51 后扩展候选；Refiner 仅处理候选。36 维特征中，第 3 列为过去两个**已闭合** Proposal 热图的 C1 运动一致性；其余列含当前已闭合窗口的统计。因此它是闭合窗口因果协议，窗口闭合前不可发布任何回放结果。

窗口输出含原始事件索引、候选掩码、全事件概率、预测与右边界；非候选事件概率为 0。空窗口仍推进 Proposal 历史；`finalize()` 可推进水位线并返回完整窗口及未发布尾部事件数。每条独立序列重新 `reset`，不得跨序列继承状态。第二代新增模型沿用计划的统一接口，并对每条事件输出补充传感器事件时间、可发布时间、候选标志、概率、预测类别及协议/成本元数据；闭合窗口回放的 SNN 仍标为协议 A，不得标成严格逐事件协议 B。

## 历史精度与错误

以下 Validation/Test 数值仅作**已冻结历史锚点**，不用于第二代结构或阈值选择。均为同一首事件锚定的完整 100 ms 前缀、exact-int8、阈值 0.35。完整来源在 `audit.json`。

| 角色 | Event IoU | AP | Precision | Recall | 高密度 Recall |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.878318 | 0.963589 | 0.949168 | 0.921671 | 0.518782 |
| Test | 0.871383 | 0.970822 | 0.919882 | 0.942947 | 0.768746 |

Train-dev 的 H2-pro 闭合前缀有 1,221,626 个事件、30,254 个目标事件、438,966 个候选。Proposal 目标召回 `29,776/30,254 = 98.4200%`，漏检 478；总漏检 4,322，其中候选内 Refiner 漏检 3,844，条件召回 `25,932/29,776 = 87.0903%`。这些数由冻结 C1 候选缓存与 Train-dev 选择记录交叉核对，见 `train_dev_errors.json`。旧 Step1 H2 报告也有 Proposal 漏检 478，但其总目标数、工作点和误差不同；不能把旧 H2 的分解当成 H2-pro 的证据。现有拆分支持先分析 Refiner，不证明空间记忆或背景机制有效。

## 历史 B03/B04 成本边界

| 角色 | 闭合窗 | 候选事件 | B03 稠密 MAC/序列 | B04 有效 MAC/序列 | B04 AC/序列 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 1,864 | 448,120 | 284,066,346.67 | 85,532,971.54 | 0 |
| Test | 1,850 | 643,002 | 318,939,040.00 | 111,750,456.96 | 0 |

B03/B04 只计 Proposal 与条件 Refiner 的正式 exact-int8 数值路径。B04 有效 MAC 基于跳零规则，AC 为 0（ANN）；不包括特征构造、连通域/路由、状态/内存读写、CPU 时间或能耗。第二代 SNN 的多值输入 MAC、二值脉冲 AC、标量状态操作和队列/状态字节须分开记录。旧正式 B 值不可直接当成 PyTorch 实际执行次数或硬件收益。冻结 H2-pro 模型参数 12,160，tensor payload 12,912 B，Proposal 两张历史热图 11,968 B；这些也不涵盖可变长事件缓存。

## 复核命令

从项目根目录运行：

```powershell
& 'D:\software\anaconda\envs\evuav\python.exe' '3_ANN_to_SNN\s0\audit.py'
& 'D:\software\anaconda\envs\evuav\python.exe' '3_ANN_to_SNN\s0\train_dev_errors.py'
```

第一条仅检查元数据、哈希、索引分区及正式 A/B 记录；第二条仅读取 Train-dev 缓存与开发集选择记录。两者均不训练、也不读取 Validation/Test 原始事件文件。
