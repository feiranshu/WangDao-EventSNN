# H2底层模块

| 文件 | 作用 |
| --- | --- |
| `constants.py` | 传感器、窗口、网络维度、阈值和导出版本常量 |
| `representation.py` | 事件窗口、patch表示、Proposal热图和候选路由 |
| `features.py` | 原H2事件特征提取 |
| `numpy_model.py` | exact-int8 NumPy MLP、H2权重加载和基线流水线 |
| `torch_model.py` | 训练用PyTorch MLP及原型logit处理 |
| `export.py` | PyTorch权重量化和H2 NPZ导出 |
| `losses.py` | 训练损失函数，非最终推理依赖 |
| `__init__.py` | H2子包导出 |

现行 H2-pro 继续使用这里的 Proposal 和 `quantized_mlp` 数值内核，以 36 维 C1 因果 Refiner 替换原 32 维 H2 Refiner；A4-2 为 Step7 历史中间模型。

