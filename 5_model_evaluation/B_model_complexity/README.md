# B 模型复杂度

B 是软件正式模型的必测项。B01/B02 是模型固有属性，只保存 `common.csv`；B03、B04a、B04b、B04c 分别保存 Validation/Test。

- B01：严格加载正式权重后的可训练参数量。
- B02：推理 `state_dict` 中参数与持久 buffer 的 tensor payload 字节数，不是 checkpoint 压缩文件大小。
- B03：实际被调用的输出位置按完整结构连接计算的稠密 MAC 基准。
- B04a：利用正式数值路径稀疏性后实际执行的 MAC。
- B04b：SNN 的 AC；普通 ANN 单列为 0。
- B04c：`B04a + B04b`，这是与 B03 稠密 MAC 正式比较的唯一口径。

B03 与 B04a/B04b 必须覆盖完全相同的序列、事件和算子边界。H2/H2-plus 的源结果已经完整。EV-SpSegNet 论文只给出约 4.0M 参数，先标记为 `paper_reported`；B02–B04 以及精确 B01 由官方 checkpoint 的 CUDA 服务器实测补齐，论文运行时间不能替代任何 B 项。

EMS-YOLO 按 FP32 SNN 数值路径和完整序列 250 ms/100 ms 滑窗统计：对实际执行的 Conv2d/Linear，非零模拟输入连接计入 B04a，二值脉冲输入连接计入 B04b；B04c 始终为两者之和。NMS、数据加载、事件帧生成和框到事件适配器不进入网络算子复杂度。现有实验报告没有同口径 B01–B04c，必须由冻结 checkpoint 实测。
