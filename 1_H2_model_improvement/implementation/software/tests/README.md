# 测试目录

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `test_a4_2_runtime.py` | 最终A4-2结构、元数据、阈值和部署决策一致性 |
| `test_core.py` | 基础数据、特征和量化内核测试 |
| `test_h2_research.py` | H2诊断与历史特征研究函数测试 |
| `test_h2_release.py` | H2发布产物测试 |
| `test_h2_students.py` | 学生模型路径测试 |
| `test_causal.py` | 因果模型与训练辅助测试 |
| `test_phase1_common.py` | 阶段一数据边界与公共函数测试 |
| `test_phase1_release.py` | 阶段一冻结发布测试 |
| `evuav_eval_33e2aff.py` | 固定版本的EV-UAV评测参考实现 |

最终部署验收只需运行`test_a4_2_runtime.py`；全量历史研究回归再运行整个目录。两类命令见上级`README_SERVER.md`。`__pycache__`和`.pyc`不属于工程文件。

