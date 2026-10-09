# SNN 模型上板与 KV260 部署

本目录只保存模型上板所需的冻结模型快照、PL 设计、KV260 固件、部署工具、板端结果和可追溯验证证据。模型训练与结构研究仍以项目其他阶段目录为准。

## 选择哪一代

| 目录 | 用途 | PL 状态 |
|---|---|---|
| `01_first_generation_baseline/` | 第一代完整基线、首次成功上板链路和 Train-dev 板端结果 | 已验证的 S1 exact AXI-Lite bitstream |
| `02_second_generation_ps_software_optimized/` | PS/PC 软件计时、功耗采集和数据通路优化 | 复用第一代 bitstream，不代表重新综合 |

两代目录完整独立，不通过链接共享 RTL、权重或固件。唯一外部依赖是各代 `dataset/EV-UAV-dataset` Junction。

## 每代目录

- `docs/`：架构、部署、指标、开发计划和迁移记录。
- `dataset/`：数据划分索引和数据集 Junction。
- `model_runtime/`：冻结 Python 模型、权重和来源证据。
- `host_tools/`：PC 数据准备、RTL 验证、部署包生成和结果评测工具。
- `pl_design/`：RTL、测试平台、黄金向量和 Vivado 工程入口。
- `kv260_deployment/`：固件、PC 自动化、板端运行、冒烟与测量工具。
- `verification_evidence/`：正式仿真、实现、联调和运行时证据。
- `experiments/`：准备包、板端运行结果和可再生成工作区。
- 根级 `migration_records/`：迁移前后 SHA256 清单和结构迁移验收记录。

两代历史文档、旧备份和旧生成包统一保存在根级本地`../archive/4_model_deployment/`，不作为当前运行入口，也不上传 GitHub。

## 常用入口

进入所选代际目录后执行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& .\kv260_deployment\pc_automation\check_deployment_prereqs.ps1 `
  -Python 'D:\software\anaconda\envs\snn_env\python.exe'
& .\kv260_deployment\pc_automation\prepare_train_dev10.ps1 `
  -Python 'D:\software\anaconda\envs\snn_env\python.exe'
```

完整操作、双 SSH 保活、结果回收和评测方法见各代 `docs/kv260_deployment_guide.md` 与 `docs/metrics_and_evaluation.md`。

## 后续所有上板模型必须遵守的 B/C 指标合同

从 2026-10-08 起，软件筛选通过的正式模型在目录 5 必须完成 A+B；实际部署到开发板的正式配置必须使用独立 `evaluation_id` 并完成 A+C，不强制重复 B。候选模型先使用目录 3 的 S01/S02 做优化、消融和选型。缺少任一强制来源时只能标记为 `incomplete`，不得用 0、估算值或 Vivado vectorless power 冒充板端实测；正式横向比较按小指标和数据划分输出 CSV。

必须保留并区分三类证据：

- **算法派生**：B01 参数量、B02 int8 权重与 float32 每输出通道尺度存储、B03 全部 Proposal 与实际 Refiner 调用的稠密 MAC、B04 非零输入/脉冲连接非零权重的有效 MAC/AC。B04 必须分项保存 MAC、AC，并用 `MAC+AC` 总量对比 B03；输入相关项覆盖所有时间步并按数据划分记录。
- **KV260 实测**：C01 PS 观测的 PL start-to-done，C02 包含读入、传输、PL、PS 后处理与落盘的端到端时间，C03 指定输入规模和持续条件的吞吐/实时因子，C04 INA260 `VCC_SOM` 空闲与运行功耗，C05 与同一命令时段对应的总/动态能耗及每序列、每 patch 能耗。
- **实现静态证据**：C06 LUT、FF、BRAM、DSP、时钟、器件和 Vivado 版本，必须来自与当前 bitstream 对应且带 SHA256 的实现报告。该面积不是板端动态读数；`VCC_SOM` 是整块 SOM，不得写成 PL-only 功耗。

正式运行入口为包内 `board_full_run_logged.sh`。它默认调用 `run_measured_full.sh`，生成：

- `runtime/<sequence>.pl.runtime.json`：SD/page-cache 读、PS 打包/AXI 写、PS 观测 PL 内核、AXI 读回、page-cache 写与最终 `fsync`；
- `runtime/<sequence>.h2.runtime.json`：PS H2 分段时间、参数/存储、稠密与有效 MAC、理论工作内存和进程最大 RSS；
- `runtime/<sequence>.full.runtime.json`：基于单调时钟的完整流水线时间；
- `runtime/power_samples.csv` 与 `runtime/power_summary.json`：原始功耗样本、空闲/运行功耗和对应能耗；
- `s1_manifest.json`、`ps_manifest.json`、`hardware_resource_manifest.json`：Proposal 复杂度、完整模型复杂度来源和硬件资源快照。

结果回收到 PC 后，在对应代际根目录运行：

```powershell
python .\host_tools\result_evaluation\collect_kv260_runtime.py `
  --manifest .\experiments\board_runs\<run_name>\manifests\s1_manifest.json `
  --runtime-dir .\experiments\board_runs\<run_name>\raw\runtime `
  --power-json .\experiments\board_runs\<run_name>\raw\runtime\power_summary.json `
  --implementation .\experiments\board_runs\<run_name>\manifests\hardware_resource_manifest.json `
  --output .\experiments\board_runs\<run_name>\evaluation\kv260_BC_metrics.json
```

板端正式记录必须能导入 `C_deployment_efficiency_and_hardware_cost/02_model_records/`，并按 Common、Validation、Test 分开。Train-dev 结果只作为部署调试证据；进入目录 5 正式横向对比时，仍应使用冻结 Validation/Test，并保持平台、输入流、边界和功耗口径一致。

## 迁移状态

目录结构和离线路径已完成迁移；冻结固件、权重和已有 Train-dev 原始结果按 SHA256 核验。目录重构后尚未重新连接 KV260，因此不得把离线迁移验证表述成新的板端实测。

两代数据集目录均为 Junction，唯一数据实体保存在 `E:\experiment\FPGA上的SNN实现\anti-EUV\EV-UAV-dataset`；D 盘不保留数据集副本。详细验收结论见 `migration_records/2026-10-08/validation_summary.md`。

