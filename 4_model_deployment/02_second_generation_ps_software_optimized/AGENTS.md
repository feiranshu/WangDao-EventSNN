# 第二代 SNN PS/软件优化协作规则

## 范围

本目录是完整独立的第二代部署副本。当前只允许不改变已验证 PL bitstream 的 PS/PC 软件、测量、归档和数据通路改进。第一代 `../01_first_generation_baseline/` 是只读基线。

模型训练和结构研究以 `../../3_ANN_to_SNN/implementation/` 为准；`model_runtime/` 与 `dataset/index.json` 是冻结部署快照。

## 硬件与证据约束

- 复用第一代 `kv260_deployment/firmware/artifacts/wangdao_pl_flat.bit.bin`。
- 当前 PL 是 S1 exact、固定调度、AXI-Lite 单 patch 接口。
- 不得声称已有纯 PL cycle counter、DMA、PL 双缓冲、PL H2 或事件相机直连。
- `pl_start_to_done_observed_ns` 包含 PS、AXI、调度与轮询，不是纯 PL 周期。
- VCC_SOM 是整 SOM 功耗；没有分轨测量时 `pl_power_w` 必须为空。
- 第一代复制证据只能放在 `verification_evidence/inherited_first_gen_pl_baseline/`，不得写成第二代重新生成。

## 文件职责

- `pl_design/` 与 `kv260_deployment/firmware/`：继承的冻结 PL 基线。
- `kv260_deployment/board_runtime/`：带分段计时的板端 Proposal 与 H2 程序。
- `kv260_deployment/measurement/`：板端功耗测量工具。
- `host_tools/result_evaluation/`：运行时、面积和功耗汇总。
- `verification_evidence/runtime_measurements/`：第二代实际新增证据。
- `experiments/`：准备包、板端运行和可再生成工作区。

## 验证

修改板端 C 或脚本后至少执行静态检查、帮助入口和小文件行为测试。真实 `/dev/mem`、`xmutil`、功耗与 SD 冷缓存结果只能在 KV260 上验证。不得覆盖冻结权重和正式证据。只有用户明确要求时才创建 Git 提交。

