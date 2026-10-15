# WangDao_EventSNN

本项目围绕事件相机目标检测模型的 ANN 改进、SNN 转换、近似计算和 FPGA 部署展开。原始两级 MLP/H2 保留为历史基线；H2-pro 是后续转换使用的冻结 ANN。各子项目的实现状态以对应入口文档为准。

## 当前进展

| 编号 | 目录 | 状态 | 作用 |
| --- | --- | --- | --- |
| 0 | `0_mlp_original_project/` | 已完成 | 保存最初的两级 MLP/H2 参考工程。 |
| 1 | `1_H2_model_improvement/` | 已完成 | 保存 H2 ANN 七步研究、H2-pro 因果最终版和完整证据。 |
| 2 | `2_KV260_bring_up_tests/` | 进行中 | KV260 环境、启动流程和基础上板测试。 |
| 3 | `3_ANN_to_SNN/` | [统一实现状态](3_ANN_to_SNN/implementation/README.md) | SNN 转换、定点、PoT/LOA、模型优化/消融/选型指标和软件黄金模型。 |
| 4 | `4_model_deployment/` | 进行中 | FPGA 部署、仿真、综合和板级验证。 |
| 5 | `5_model_evaluation/` | v2 标准已修订；Pd/Fa 重算、状态内存清单及部署测量待补齐 | 按准确度、模型复杂度、部署效率与硬件成本三类评测；历史数值保留。 |

## 当前模型结论

- H2 是第一代 SNN 的历史教师；H2-pro 转换的当前状态、版本结果与有效计划统一见 [ANN→SNN 实现说明](3_ANN_to_SNN/implementation/README.md)。
- H2-pro 是最终入选改进模型：C1 历史 Proposal 运动特征替换非因果全序列时间维度，seed42 epoch10，exact-int8，阈值 0.35。H2-plus/A4-2 已归入七步研究 Step7 中间模型。
- 正式 H2/H2-pro A/B 比较使用同一闭合 100 ms 窗口前缀。H2 阈值 0.30；H2 旧评分由全序列结果裁剪，仍含非因果时间输入。Validation/Test 不用于重新选择模型或阈值。
- EV-SpSegNet 已注册为软件正式对比模型，等待同口径 CUDA 服务器补测；论文值不冒充本项目完整实测。
- 已完成的准确度与复杂度属于软件评测，不等同于 KV260 板级实测。

## 统一模型评测体系

目录 5 保留 A/B/C 三类，现行规则见 [评价标准 v2](5_model_evaluation/EVALUATION_STANDARD_V2.md)。H2/H2-pro 原软件结果保留；新版 Pd/Fa 和 B05 尚待补齐，因此当前准入为 `incomplete`。EV-SpSegNet 仍待同协议补测，跨模型比较使用 CSV。

| 类别 | 评测内容 | 当前状态 |
| --- | --- | --- |
| [A_model_accuracy](5_model_evaluation/A_model_accuracy/README.md) | A01–A22，及配对不确定性。 | 旧 Pd/Fa 为 legacy_only；统一 v2 重算及 EV-SpSegNet 同前缀补测待完成。 |
| [B_model_complexity](5_model_evaluation/B_model_complexity/README.md) | B01/B02 存储参数、B03 结构连接、B04 理论非零运算、B05 持久状态。 | H2/H2-pro 原 B01–B04 保留；B05 完整清单待补。 |
| [C_deployment_efficiency_and_hardware_cost](5_model_evaluation/C_deployment_efficiency_and_hardware_cost/README.md) | C01–C10：延迟、吞吐、功耗、能耗、资源、runtime、内存、覆盖/积压、数值一致性。 | 协议与待测位置已建立，未生成虚构实测。 |

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
