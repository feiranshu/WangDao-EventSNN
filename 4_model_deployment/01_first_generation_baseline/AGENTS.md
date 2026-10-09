# 第一代 SNN 基线协作规则

## 范围

本目录只负责第一代冻结 SNN 的 PL 设计、KV260 固件、板端运行和验证。模型训练、结构研究和正式软件模型结果以 `../../3_ANN_to_SNN/implementation/` 为准。

`model_runtime/wangdao/`、`model_runtime/weights/` 和 `dataset/index.json` 是只读部署快照，不得在这里独立改变模型后假定上游会自动同步。

## 证据边界

- Python 黄金模型、RTL 行为仿真、综合实现、板级实测分开记录。
- 单 patch 冒烟、少量窗口回放和完整数据集验收分开表述。
- Vivado 功耗估计不是板级功耗，软件周期或操作计数不是能耗。
- Test 只在模型和部署流程全部冻结后运行，不用于调试。

## 文件职责

- `pl_design/`：RTL、测试平台、黄金向量和 Vivado 工程。
- `kv260_deployment/`：固件、PC 自动化、板端程序、冒烟与测量。
- `host_tools/`：数据准备、RTL 联调、部署包生成和结果评分。
- `verification_evidence/`：正式模型、仿真、综合与联调证据。
- `experiments/prepared_packages/`：可再生成的板端包。
- `experiments/board_runs/`：板端原始输出及对应评测。
- `experiments/workspace/`：可删除、可再生成工作区。
- `../../archive/4_model_deployment/01_first_generation_baseline/`：根级本地历史材料，不作为当前运行入口且不得上传 GitHub。

## 验证

修改 RTL、接口、向量或板端工具后运行对应单元测试和行为回归；只有跨模块冻结时才做完整综合或板测。不得覆盖冻结权重、正式证据或首次 Test 结果。只有用户明确要求时才创建 Git 提交。

