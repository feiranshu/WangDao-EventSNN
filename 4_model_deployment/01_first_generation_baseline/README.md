# 第一代 SNN 基线部署

本目录是第一代冻结 SNN 的完整、独立 KV260 部署基线，包含已验证 PL 固件、模型快照、部署工具、验证证据和首次 Train-dev 板端结果。

## 文档入口

- [架构与数据通路](docs/architecture_and_dataflow.md)
- [KV260 部署指南](docs/kv260_deployment_guide.md)
- [指标、测量与评测](docs/metrics_and_evaluation.md)
- [开发计划](docs/development_plan.md)
- [目录迁移记录](docs/migration_manifest.md)

## 关键目录

- `model_runtime/`：冻结 Python 模型及权重。
- `pl_design/`：第一代 RTL、测试向量和 Vivado 工程。
- `kv260_deployment/firmware/artifacts/`：板卡验证过的 bitstream、DTBO、XSA 和制品清单。
- `kv260_deployment/pc_automation/`：Windows 预检、上传和批跑入口。
- `kv260_deployment/board_runtime/`：KV260 板端 Shell 与 C 程序。
- `experiments/board_runs/train_dev10/`：首次 Train-dev 板端原始结果与本地评测。
- `verification_evidence/`：RTL、综合实现和主机联调正式证据。

## 部署前检查

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& .\kv260_deployment\pc_automation\check_deployment_prereqs.ps1 `
  -Python 'D:\software\anaconda\envs\snn_env\python.exe'
```

只有出现 `WANGDAO_ARTIFACT_PREFLIGHT_PASS` 和 `WANGDAO_PC_PREREQUISITES_PASS` 才继续部署。当前 DTBO 必须为 1986 bytes，SHA256 为 `0a1c9eb846b6929129b315b5e3ff510a06e332a74744dc048a9bc46c47ca2706`。

## 证据边界

目录重构未修改 PL、权重、数据划分或评测口径。现有板端结果是迁移前完成的真实结果；重构后的脚本已经离线验证，但尚未重新连接开发板执行。

