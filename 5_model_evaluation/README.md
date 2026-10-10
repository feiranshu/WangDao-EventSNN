# 模型评测统一入口

目录 5 只保留三类正式评测：A 模型准确度、B 模型复杂度、C 部署效率与硬件成本。横向比较采用“一个小指标一个目录、一个数据划分一个 CSV”，不维护集中式总表、图表或工作簿。

## 准入规则

- 软件正式模型：A Validation/Test 与 B Common/Validation/Test 必测。
- 板端正式模型：A Validation/Test 与 C Common/Validation/Test 必测。
- 软件模型不强制 C，板端模型不强制 B；同一模型的软件与板端配置使用不同 `evaluation_id`。
- C07 Runtime 记录完整序列模型推理运行时间（`ms/sequence`），同时接受软件与板端实测；目前作为软件补充项，不改变软件 A+B 准入规则。定义及待测表位于 [C07_runtime](C_deployment_efficiency_and_hardware_cost/01_metric_comparisons/C07_runtime/README.md)，软件覆盖率中的 C `not_required` 不代表已经测过 runtime。
- EV-SpSegNet 是软件正式对比模型，不再作为 `external_reference` 排除。论文能够直接支持的 Test A02/A04/A06/A07 和 B01 先标记为 `paper_reported`；其余项以及同口径复测由服务器 A+B 包完成。`paper_reported` 不等同于本项目协议下的 `complete`。
- EMS-YOLO 是软件正式 SNN 对比模型。现有实验报告的框级 `mAP@0.5` 与 A01 Event AP 口径不同，不直接入表；A/B 由冻结权重的 CUDA 服务器包同口径实测。
- 板端 A 优先采用板端输出；只有逐项 bit-exact 且留有等价性报告时，才可引用软件 A 结果。

## 当前模型状态

- H2、H2-plus：A01–A22 与 B01–B04c 均完整。
- C07 Runtime：已为注册的软件正式模型建立 Validation/Test 待测表；尚无同口径实测值，空白保持 `pending`。
- EV-SpSegNet：已正式注册；论文值已录入，完整性暂为 `incomplete`，等待官方 CUDA 权重同口径实测回传。
- EMS-YOLO：已正式注册；冻结 FP32 SNN 权重和历史阈值 0.407，完整性暂为 `incomplete`，等待 Validation/Test A+B 服务器实测回传。该阈值曾由 Validation F1 峰值选择，因此 Test 是主要无偏横向结果。

## 查找顺序

1. 在 `model_registry.csv` 查稳定 ID、显示名、轨道、角色、权重哈希和源清单。
2. 在各类 `01_metric_comparisons/` 按指标和数据划分查看正式对比。
3. 在 `02_model_records/` 复核每个模型的源结果；定义与复现协议在 `03_rules_and_protocols/`。
4. 用 `evaluation_coverage.csv` 查看强制项是否齐全；`paper_reported` 与 `pending` 都不会通过正式完整性检查。

CSV 统一字段为 `evaluation_id,model_name,evaluation_track,model_role,platform_id,value,unit,status,evidence_type,protocol_id,source_file,notes`。CSV 是唯一正式对比数据，README 不重复抄录实测数值。
