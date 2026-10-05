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

最终A4-2继续使用这里的Proposal和`quantized_mlp`数值内核，但用36维A4-2 Refiner替换原32维H2 Refiner。

