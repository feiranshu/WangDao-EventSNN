# KV260 Train-dev10 板端结果整理与评测

本目录只对应 `train_dev10` 开发集，不写入根目录 `5_model_evaluation`，不能作为 Validation/Test 正式结果。

## 核心结论

- PL Proposal-SNN：10/10 条序列与 Python 定点 golden 逐字一致。
- PS H2：10/10 条序列与 Python fixed golden 逐字节一致。
- 事件数：1,237,183；目标事件：30,659；候选事件：479,130。
- Event IoU：0.738857；Precision：0.896503；Recall/官方兼容 ACC：0.807756；F1：0.849819。
- Pd：0.771485；Fa：1.392056e-05；Fa×10⁴：0.139206。
- Event AP：0.885586。该值由位级一致的 PL capture 在 PC 上重放冻结 fixed H2 概率得到，不是板端直接保存的概率。
- HOTA：0.321677；DetA：0.355828；AssA：0.291447；LocA：0.923956。

## 文件说明

- `summary.json`：完整总览与证据边界。
- `per_sequence.csv`：逐序列事件指标、Pd/Fa、HOTA 与一致性。
- `threshold_curve.csv`、`pr_curve.csv`：基于 PL capture + fixed H2 重放的阈值曲线和 PR 曲线。
- `hota_summary.json`：板端预测与 ANN 基线的冻结轨迹协议结果。
- `model_complexity.json`、`complexity_per_sequence.csv`：可从本次输入规模确定的 B 类复杂度。
- `deployment_metrics.json`：C 类可用面积/时序，以及无法事后补算的运行时字段。
- `ann_snn_difference.json`：同一 Train-dev10 上的 T=4 SNN 与冻结 ANN 基线差值。
- `artifact_checksums.csv`：原始 capture/prediction 与 manifest 的 SHA-256。
- `replay_probabilities/`：板端 PL capture 重放概率和 ANN 基线概率，用于复核 PR/AP。
- `pr_curve.png`、`threshold_metrics.png`：图形化结果。
- `evaluate_board_results.py`：复现脚本。

## 不能从当前归档测量的指标

本次归档没有 `runtime.json`、板端单调时钟采样、功耗采样或外部仪表记录，因此 B01 内核延迟、B02 端到端延迟、B03 吞吐/实时性、B04 运行功耗、B05 单任务能耗均标为未测。文件修改时间不具备明确测量边界，未被用于推算性能。

B07 面积/资源来自与本次 exact bitstream 对应的 Vivado implemented-design 报告：LUT 21,531，寄存器 10,707，BRAM 0，DSP 0。其中 2.854 W 是 Vivado vectorless 估计，不是板级实测功耗。

## 口径说明

工作点 Event IoU/Precision/Recall/F1/Pd/Fa 直接读取板端二值 prediction；PR/AP 使用与板端 PL 输出逐字一致的 capture，在 PC 上执行冻结 fixed H2 重放。密度分层阈值为 20,000 events/s。轨迹采用 `evuav-observable-event-track-hota-v1` 冻结适配器（3 px、20 px/50 ms、最大缺失 20 窗）。ANN→SNN 差值同时包含 Proposal 转换和定点部署影响，不能拆成独立量化损失或单独转换损失。
