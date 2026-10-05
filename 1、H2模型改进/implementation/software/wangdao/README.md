# `wangdao` Python包

## 文件与子目录

| 文件或目录 | 作用 |
| --- | --- |
| `a4_2_runtime.py` | 最终A4-2权重校验、36维特征拼接和exact-int8推理 |
| `data.py` | EV-UAV结构化事件验证与NPZ加载 |
| `h2_research.py` | 冻结H2 Proposal、候选特征和4维因果历史特征 |
| `h2_refiner.py` | H2 Refiner特征与训练辅助 |
| `phase1_common.py` | 冻结数据角色、索引检查和阶段一公共逻辑 |
| `phase1_metrics.py` | 官方兼容事件与定位指标 |
| `metrics.py` | 通用评测汇总函数 |
| `runtime.py` | 设备、随机种子和环境记录工具 |
| `h2/` | 原H2表示、特征、NumPy/PyTorch网络与导出内核；详见`h2/README.md` |
| `causal.py`、`causal_training.py` | 历史因果实验框架，保留用于复现 |
| `h2_release.py` | H2发布清单和一致性逻辑 |
| `h2_students.py` | 学生模型实验，不是最终A4-2路径 |
| `__init__.py` | Python包标记 |

最终推理从`A42Pipeline`开始。它复用冻结H2 Proposal生成候选与原32维特征，再追加4维历史特征并调用最终int8 Refiner。

