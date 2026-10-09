# 指标、测量与评测

来源：`# 第一代 SNN 模型 KV260 部署：问题复盘、已验证修复与标准流程`。目录重构时按主题拆分，技术口径未改变。本文沿用第一代已实测 PL 基线；第二代仅包含 PS/软件优化，未重新生成 bitstream。

## 16. 下一步：正式模型指标

当前结果已经在 PC：

```text
experiments/board_runs/train_dev10/predictions
```

而且板端结果已验证与 PC H2 golden bit-exact。

所以下一步只需要：

```text
10 条 board predictions
        +
Train-dev GT
        ↓
统一 evaluation pipeline
        ↓
Event IoU
Precision
Recall
F1
Accuracy
Proposal Recall
PD
FA
```

注意：

> 不要另写一套新的指标定义。必须继续使用第一代项目原有、与 H2 / H2-plus 对比时相同的统一 metrics 实现，保证指标口径一致。

---

## 迁移后的统一测量输出

每次正式板端运行至少保存以下字段，原始样本放入 `experiments/board_runs/<run_name>/raw/runtime/`，汇总放入该 run 的 `evaluation/`；不得只保留终端截图。

| 指标 | 边界与推荐计时点 | 单位 |
|---|---|---|
| `storage_to_pl_observed_ns` | 每个 patch 从读取 SD/page cache 到最后一个输入字写入 PL 寄存器；另存 `sd_or_page_cache_read_ns` 与 `ps_pack_and_axi_write_ns` | ns |
| `pl_start_to_done_observed_ns` | 写 start 到 PS 观察到 done；逐 patch 保存并汇总总和、均值、P50/P95。包含 AXI 访问、轮询和调度开销，不等于纯 PL 周期 | ns |
| `pl_to_storage_component_total_ns` | AXI 输出读取、page-cache 写和 capture 最终 `fsync` 的组成总和 | ns |
| `full_model_total_ns` | Proposal PL 进程、capture `fsync`、PS H2 和 prediction `fsync` 的完整单序列包络 | ns |
| `VCC_SOM_total` 功耗 | INA260 运行前 idle 与完整命令 active 样本，保存均值、中位数、P95、最小/最大值及 active-idle 动态代理 | W |
| 单位任务能耗 | `active/dynamic mean power × monotonic command duration`，同时输出总量、每序列和每 patch；范围仍为整个 SOM | J |
| B07 `area` | 从匹配当前 bitstream SHA256 的实现报告读取 LUT、FF、BRAM、DSP、时钟、器件和 Vivado 版本 | count / % |

时间统一使用 `CLOCK_MONOTONIC_RAW`（C）或 Python 单调时钟。功耗工具位于 `kv260_deployment/measurement/`，实现面积证据位于第一代 `verification_evidence/implementation/`；第二代引用 `verification_evidence/inherited_first_gen_pl_baseline/implementation/`，因为未重新综合。KV260 当前暴露的 INA260 节点是 `VCC_SOM`，因此 active-idle 只能称为 PL 工作负载动态功耗代理；若要“PL-only 功耗”，必须增加可隔离的 PL 电源轨或外部仪表，不能靠软件改名。

## B_model_complexity（A01—A06）

`prepare_kv260_batch.py` 在生成完全相同的板端输入时，同时记录 Proposal-SNN 的参数、存储、状态、全部时间步的稠密突触乘积、逐层脉冲数和有效 AC。`prepare_ps_h2_batch.py` 记录 Refiner 静态复杂度与由真实候选数决定的稠密 MAC；板端 `wangdao_ps_h2.c` 再从实际激活记录有效 MAC、理论工作缓冲和进程最大 RSS。A03 中理论状态/工作内存与进程 RSS 分开，A05 是“非零输入或脉冲连接非零权重”的算法活动量，不表示当前稠密 RTL/C 循环已跳过这些运算，也不能直接换算能耗。

## 正式运行与汇总

准备包后，在板上执行 `sh board_full_run_logged.sh`。该入口默认先采 10 秒 idle 功耗，再用功耗采样包围完整批次，最后做 bit-exact 核验并打包结果。若 INA260 节点不唯一，先用 `measure_kv260_power.py --power-node ...` 确认正确的 `power1_input`，再设置 `WANGDAO_POWER_NODE=/sys/class/hwmon/.../power1_input` 后运行；不得跳过功耗后仍把 C 类结果标成 complete。

回收到 PC 后运行：

```powershell
python .\host_tools\result_evaluation\collect_kv260_runtime.py `
  --manifest <run>\manifests\s1_manifest.json `
  --runtime-dir <run>\raw\runtime `
  --power-json <run>\raw\runtime\power_summary.json `
  --implementation <run>\manifests\hardware_resource_manifest.json `
  --output <run>\evaluation\kv260_BC_metrics.json
```

输出的顶层 `B_model_complexity` 对应 A01—A06；`C_deployment_efficiency_and_hardware_cost` 对应 B01—B05、B07。缺少新 manifest、分段 runtime、功耗或实现证据时汇总器会保留空值/不完整状态，不接受用 0 占位。

## 模型效果评测

板端 prediction 下载后使用 `host_tools/result_evaluation/score_kv260_batch.py` 和项目既有评测口径计算逐序列及总体指标。至少保留 Event Precision/Recall/F1、Event IoU、定位误差、误报率，以及仓库评测器可计算的 PR 曲线、HOTA/跟踪与 ANN↔SNN 差异。Train-dev 结果只保存在本代 `experiments/board_runs/train_dev10/evaluation/`，不写入项目根级 `5_model_evaluation`。

所有结论必须区分三类状态：历史继承证据、结构迁移后主机侧复验、开发板重新实测。没有重新上板时，不得把前两类写成“第二代板端验证通过”。
