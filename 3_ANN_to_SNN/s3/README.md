# S3 训练与状态分解

当前状态：**S3-A 已完成；S3-B 的研究 ANN 已完成，持久状态 SNN 待服务器训练结果；S3-C/D/E 按证据条件执行。** 本目录独立保存 S3 协议、只读诊断、统一训练实现、测试、结果与服务器上传包。S2 的源码与检查点只作冻结依赖，不改写。

- [`PROTOCOL.md`](PROTOCOL.md)：S3-A/B/C 的数据角色、目标函数、预算、阈值与条件分支。
- [`REPORT_A.md`](REPORT_A.md)：S3-A 的同 FP 预算、逐层状态、读出、推理干预和输入关联结果。
- [`diagnose.py`](diagnose.py)：冻结 S2 epoch5 的只读回放；完整结果在 [`results/GEN2-S3-001-A-revision1/`](results/GEN2-S3-001-A-revision1/)。
- [`runtime.py`](runtime.py)、[`train.py`](train.py)：设备常驻 SNN 状态、与 ANN 对齐的 8 窗口更新组和统一加权 CE；本地完整 ANN 历史及检查点在 [`runs/ann/`](runs/ann/)。
- [`preflight.py`](preflight.py)、[`test_runtime.py`](test_runtime.py)：哈希/角色隔离、CPU 与目标设备数值门槛、S2 等价性。
- [`README_SERVER.md`](README_SERVER.md)：服务器单行命令、数据链接、中断恢复与回传说明。
- [`upload/GEN2_S3_server_20261012.zip`](upload/GEN2_S3_server_20261012.zip)：Linux 上传包；同目录有 `.zip.sha256` 和 `.manifest.json`。原始数据集不在包内。

研究 ANN 按 S3 统一训练口径完成 seed42、79/10 序列、10 epoch，每轮 2,219,864 个训练候选和 766 次优化更新。第 10 轮 Train-dev AP **0.877494**、预设网格选择阈值 0.60、Event IoU **0.679662**；第 10 轮仍较第 9 轮改善。它是研究对照，不替代正式 H2-pro 锚点。CPU/CUDA 数值一致性必须在目标 4090 环境的前置检查通过，不能以本地 CPU 测试代替。

**S3 尚未完成。** 收到服务器 `GEN2_S3_B_results_20261012.zip` 后，先核对 10 epoch SNN 的训练健康、PR/AP、同 FP 预算、成本和状态，再决定是否启动已准备的 S3-C 窗口重置对照，以及 S3-D/E 的入口。Validation/Test/HW-calibration 原始事件尚未读取。
