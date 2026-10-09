# 第二代 SNN 模型 KV260 软件侧改进计划

## 1. 目标

在不修改 PL 工程、不重新生成 bitstream 的前提下，复用第一代 S1 exact AXI-Lite 实现，补齐可复现的分段计时、结果持久化、完整模型时间、面积归档和功耗输入接口，并通过软件缓冲降低明显的文件 I/O 开销。

## 2. 七项改进的边界

| 项目 | 本代是否实施 | 原因 |
|---|---|---|
| RTL cycle counter | 否 | 需要改 RTL 和 bitstream |
| C 驱动分段计时/JSON | 是 | 纯 PS 软件改动 |
| SD/PS/MMIO 分段 | 是 | 纯 PS 软件改动，但需标注页缓存 |
| AXI DMA/Stream | 否 | 需要 PL IP、地址和驱动变化 |
| 双缓冲 | 部分 | 仅做文件缓冲/顺序预读；单 PL input buffer 阻止计算与 AXI 输入安全重叠 |
| H2 迁入 PL | 否 | 需要新硬件实现 |
| 相机直连 PL | 否 | 需要新接口和硬件数据通路 |

## 3. 已实施

- Proposal 驱动采用 `CLOCK_MONOTONIC_RAW`，逐 patch 记录存储读取、PS 打包/AXI 写、PS 观测 start→done、AXI 输出读、页缓存写和 patch 总时间。
- 每序列 capture 执行 `fflush + fsync`，单独记录最终持久化时间。
- 输入/输出使用 1 MiB stdio 缓冲；输入调用 `posix_fadvise` 的 SEQUENTIAL/WILLNEED 提示。
- `run_batch.ps1` 自动下载并校验每序列 runtime JSON。
- PS H2 改为先在内存生成 prediction 数组，再批量写入并 fsync，同时输出输入、计算、输出和总时间。
- `board_full_run.sh` 输出 Proposal、H2 和完整序列三类 runtime JSON。
- `collect_kv260_runtime.py` 汇总时间、吞吐、第一代同 bitstream 的面积/时序以及可选功耗记录。
- `measure_kv260_power.py` 在板端采集 INA260 VCC_SOM idle/active 样本并生成可直接汇总的 CSV/JSON；结果明确不是 PL 分轨实测。
- 功耗模板强制区分 VCC_SOM 总功耗、动态代理和直接 PL 分轨测量。

## 4. 待板测验证

1. 在 KV260 Ubuntu 上以 `-Wall -Wextra` 编译两个 C 程序。
2. 单 patch 与 1 条 HW-calibration 逐字对齐。
3. 核对 runtime JSON 的 patch 数、文件大小和单调时钟结果。
4. 分别执行 warm-cache 与受控 cold-cache 测量；记录 `findmnt`。
5. 完成 10 条 HW-calibration 后汇总 `runtime.json`。
6. 采集 VCC_SOM idle/active 功耗；没有分轨条件时保持 `pl_power_w=null`。
7. 流程冻结后再运行 Validation，最后才运行 Test。

## 5. 下一硬件代建议

下一次允许重建 bitstream 时，按优先级增加 PL cycle counter、DMA/AXI-Stream、ping-pong input/output buffer，再评估 H2 部分迁移和事件相机直连。每项都应与本代 AXI-Lite基线使用相同数据、时钟、指标和功耗协议比较。
