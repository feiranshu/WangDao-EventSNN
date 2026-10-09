# WangDao_EventSNN

本项目围绕事件相机目标检测模型的 ANN 改进、SNN 转换、近似计算和 FPGA 部署展开。当前主线以原始两级 MLP/H2 为冻结基线，先完成 ANN 改进与统一评测，再转换为 SNN，并面向 KV260 开发板开展硬件实现。

## 当前进展

| 编号 | 目录 | 状态 | 作用 |
| --- | --- | --- | --- |
| 0 | `0_mlp_original_project/` | 已完成 | 保存最初的两级 MLP/H2 参考工程。 |
| 1 | `1_H2_model_improvement/` | 已完成 | 保存 H2 ANN 七步改进、最终 H2-plus 和完整研究证据。 |
| 2 | `2_KV260_bring_up_tests/` | 进行中 | KV260 环境、启动流程和基础上板测试。 |
| 3 | `3_ANN_to_SNN/` | 第一代已完成，后续进行中 | SNN 转换、定点、PoT/LOA、模型优化/消融/选型指标和软件黄金模型。 |
| 4 | `4_model_deployment/` | 进行中 | FPGA 部署、仿真、综合和板级验证。 |
| 5 | `5_model_evaluation/` | H2 系列 A/B 已完成，对比模型补测中，C 待上板 | 只按准确度、模型复杂度、部署效率与硬件成本三类正式评测入选模型。 |

## 当前模型结论

- H2 是冻结 ANN 基线，也是后续 SNN 研究的教师模型。
- H2-plus 是最终入选改进模型，历史实验 ID 为 `A4-2`，正式版本为 seed42 epoch8。
- 当前正式比较采用 `exact_int8_replay`；H2 阈值 0.30，H2-plus 阈值 0.40。Validation/Test 只在配置冻结后报告，不用于重新选择模型或阈值。
- EV-SpSegNet 与 EMS-YOLO 已注册为软件正式对比模型，目前等待冻结权重的同口径 CUDA 服务器补测；论文值与历史框级 mAP 不冒充本项目完整实测。
- 已完成的准确度与复杂度属于软件评测，不等同于 KV260 板级实测。

## 统一模型评测体系

目录 5 最终只保留 A/B/C 三类。H2 与 H2-plus 已完成正式软件评测；EV-SpSegNet 与 EMS-YOLO 已进入正式软件对比注册表，但覆盖率仍为 `incomplete`。跨模型比较统一使用 CSV，不再维护图表或工作簿。

| 类别 | 评测内容 | 当前状态 |
| --- | --- | --- |
| [A_model_accuracy](5_model_evaluation/A_model_accuracy/README.md) | 原有事件级、EV-UAV 兼容和可观测事件轨迹指标。 | H2/H2-plus 的 Validation、Test 已完成；EV-SpSegNet 与 EMS-YOLO 待同口径补测。 |
| [B_model_complexity](5_model_evaluation/B_model_complexity/README.md) | B01 参数量、B02 静态存储、B03 稠密 MAC、B04 有效 MAC/AC。 | H2/H2-plus 已完成；EV-SpSegNet 与 EMS-YOLO 待冻结模型补测。 |
| [C_deployment_efficiency_and_hardware_cost](5_model_evaluation/C_deployment_efficiency_and_hardware_cost/README.md) | C01—C06：延迟、吞吐、功耗、能耗和硬件资源/面积。 | 等待同平台、同输入条件的正式上板实测。 |

原 D 类正式取消，其 C01/C02 已迁入 [3_ANN_to_SNN/model_selection_metrics](3_ANN_to_SNN/model_selection_metrics/README.md)，重编号为 S01/S02，用于模型优化、消融和选型。中间模型、旧图表、旧 XLSX 和历史入口统一保存在本地根归档区 `archive/5_model_evaluation/`，不再在各子目录内设置 `archive/`。

A/B/C 正式目录统一采用 `01_metric_comparisons/`、`02_model_records/`和`03_rules_and_protocols/`。每个小指标按 Common 或 Validation/Test 分文件比较；模型身份与顺序由 `model_registry.csv` 固定，强制项由 `evaluation_coverage.csv` 自动检查。软件正式模型必测 A+B，板端正式模型必测 A+C；历史布局和旧服务器包只保存在根级本地`archive/`，详细规则见 [`5_model_evaluation/README.md`](5_model_evaluation/README.md)。

## 本地归档规则

全项目只保留一个根级 `archive/`。从子目录退出正式主线、但仍需留存复核的材料，按原来源路径归入 `archive/<原父目录路径>/`；任何编号子目录下都不再保留独立 `archive/`。完整来源清单记录在本地 `archive/README.md`。

根级 `archive/` 只用于本机历史复核，不是正式模型、正式指标或部署入口。

## GitHub 发布范围

本仓库发布编号 0—5 的六个正式工程目录。目录 2 的本地第三方仓库克隆及工具缓存不上传，目录 4 的外部数据集 Junction 和 Vivado 可再生成目录不上传；KV260 启动镜像、冻结固件和二进制部署产物通过 Git LFS 管理。

根级 `archive/`、`tmp/`、`project_proposal/` 是永久本地目录，任何后续 Git 提交、GitHub 上传或同步操作都不得包含；该规则不因 0—5 各阶段状态变化而取消。外部数据集、缓存、训练 checkpoint、日志和工具生成目录继续按各子工程 `.gitignore` 排除。

## 大文件说明

部分 ZIP/NPZ 研究结果、KV260 启动镜像、冻结固件和二进制部署产物使用 Git LFS。首次克隆前安装 Git LFS 并执行 `git lfs install`；如未自动下载大文件，在仓库目录运行 `git lfs pull`。

## 使用说明

本地项目根目录统一为 `D:\WangDao_EventSNN`。各子工程入口、数据边界和验证方法以对应 `README.md` 与 `AGENTS.md` 为准。Train-main、Train-dev、HW-calibration、Validation 和 Test 必须严格分离，不得用 Test 重新选择模型、阈值或量化参数。
