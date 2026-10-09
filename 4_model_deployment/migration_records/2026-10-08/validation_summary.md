# 4_model_deployment 目录迁移验收记录

日期：2026-10-08

结论：**结构迁移验证完成；未声明开发板重新验证完成。** 本次没有重新综合 PL、没有重新生成或修改 bitstream/DTBO/权重，也没有改写既有板端 capture、prediction 或评测数值。

## 目录与数据

- 活动代际目录仅有 `01_first_generation_baseline` 和 `02_second_generation_ps_software_optimized`。
- 两代根目录均不存在旧的 `software`、`hardware`、`results`、`tmp` 顶层目录。
- 活动目录中没有 `__pycache__`、`.pyc` 或散落 `.bak`；历史 `.bak` 已集中迁入根级本地`archive/4_model_deployment/`对应代际的`legacy_backups/`。
- 两代 `dataset/EV-UAV-dataset` 均为 Junction，目标为 `E:\experiment\FPGA上的SNN实现\anti-EUV\EV-UAV-dataset`。
- E 盘数据集共 19,061 个文件；D 盘迁移期间产生的 10.54 GB 安全副本已按用户明确授权删除，因此数据实体只保留在 E 盘。

## 不可变制品

- 迁移前后对照 72 项关键文件：71 项字节数和 SHA256 完全一致，0 项缺失。
- 唯一预期变化是 `experiments/board_runs/train_dev10/evaluation/evaluate_board_results.py`，其仓库路径解析已改为新目录。
- 固件、权重、`dataset/index.json`、10 条 Train-dev capture、10 条 prediction、评测 JSON/CSV/PNG/NPZ 均与迁移前一致。
- 两代 DTBO：1,986 bytes，SHA256 `0a1c9eb846b6929129b315b5e3ff510a06e332a74744dc048a9bc46c47ca2706`。
- 详细记录见同目录的 `pre_migration_sha256.csv`、`post_migration_sha256.csv`、`checksum_verification.csv` 和 `checksum_summary.json`。

## 主机侧验证

- 第一代完整单元测试：87/87 通过。
- 第二代完整单元测试：87/87 通过。
- Python AST：第一代 42 个、第二代 43 个活动 Python 文件通过。
- 两代各 4 个主要 Python CLI 的 `--help` 通过。
- 两代全部活动 PowerShell 脚本语法解析通过。
- PC 环境预检与固件制品预检通过；使用 Python 3.9.25、NumPy 1.26.4、PyTorch 2.8.0+cpu。
- 跨盘单序列准备包验证通过：`train_022.npz`、1 个窗口、336 个 patch，输入与 golden 文件 SHA256 均与 manifest 一致。
- 当前 Windows 主机只有不可用的 WSL `bash.exe`，因此未在本机宣称完成 `sh -n`；板端 Shell 语法检查和固定双 patch smoke 应在 Linux/KV260 上执行。

## 既有 Train-dev 结果

- 10/10 PL capture 与 golden 逐字一致，PL mismatch words = 0。
- 10/10 PS prediction 与 frozen reference 逐字节一致，PS mismatch bytes = 0。
- Event IoU = 0.7388567337，Precision = 0.8965030408，Recall = 0.8077562869，F1 = 0.8498189867。
- Pd = 0.7714853453，FA×1e4 = 0.1392056000，AP = 0.8855861024，HOTA = 0.3216772530。
- 现有归档没有可靠的单调时钟样本或板级功耗样本，因此运行延迟、吞吐、运行功耗和能耗仍保持 `not_measured`，没有用文件时间戳或 Vivado 估算冒充实测。
- 资源证据来自匹配 bitstream 的实现报告：100 MHz，WNS 0.956 ns，LUT 21,531，寄存器 10,707，BRAM 0，DSP 0；2.854 W 是 Vivado vectorless 总片上估算，不是 PL 运行实测功耗。

## 仍需开发板执行

1. 在 Linux/KV260 上对 `kv260_deployment/board_runtime/*.sh` 执行 `sh -n`。
2. 运行固定双 patch smoke。
3. 若要补齐运行效率指标，按 `docs/metrics_and_evaluation.md` 采集 SD→PL、PL 总时间、PL→SD、整体时间、PL 功耗/能耗；面积继续引用与 bitstream 匹配的静态实现报告。

