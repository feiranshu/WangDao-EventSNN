# 第二代 SNN PS/软件优化部署

本目录是第二代完整、独立部署副本。改进范围是 PC/PS 数据通路、分段计时、功耗采集和结果归档；PL 继续复用第一代已验证 bitstream，不代表重新综合。

## 文档入口

- [架构与数据通路](docs/architecture_and_dataflow.md)
- [KV260 部署指南](docs/kv260_deployment_guide.md)
- [指标、测量与评测](docs/metrics_and_evaluation.md)
- [开发计划](docs/development_plan.md)
- [目录迁移记录](docs/migration_manifest.md)

## 关键目录

- `model_runtime/`：第二代部署所需的独立冻结模型快照。
- `pl_design/` 与 `kv260_deployment/firmware/`：继承的第一代 PL 基线。
- `kv260_deployment/measurement/`：SOM 功耗采集及配置。
- `host_tools/result_evaluation/collect_kv260_runtime.py`：运行时间、实现面积和可选功耗汇总。
- `verification_evidence/inherited_first_gen_pl_baseline/`：继承证据，不应描述成第二代重新生成。
- `verification_evidence/runtime_measurements/`：第二代实际新增的计时与测量证据。

## 部署前检查

```powershell
Set-ExecutionPolicy -Scope Process Bypass
& .\kv260_deployment\pc_automation\check_deployment_prereqs.ps1 `
  -Python 'D:\software\anaconda\envs\snn_env\python.exe'
```

只有出现 `WANGDAO_ARTIFACT_PREFLIGHT_PASS` 和 `WANGDAO_PC_PREREQUISITES_PASS` 才继续部署。第二代不得将 PS 观测时间称为纯 PL cycle，也不得把 VCC_SOM 功耗称为 PL 单独功耗。

## 证据边界

目录重构没有修改或重新生成 bitstream、DTBO 和冻结权重。重构后的工具完成离线验证后仍需下一次 KV260 运行才能形成新的板端证据。

