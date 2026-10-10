# B 模型复杂度

B01 参数量、B02 模型 tensor payload 字节数为 Common；B03 稠密 MAC、B04a 有效 MAC、B04b AC、B04c MAC+AC 分 Validation/Test。H2 与 H2-pro 已在同一闭合窗口前缀上完成。B03/B04 仅覆盖 Proposal 和条件 Refiner 的 exact-int8 计算，未覆盖特征构建、路由、后处理、延迟及内存工作区。普通 ANN 的 B04b 为 0。

H2-plus 旧记录已归入 Step7 中间模型。EV-SpSegNet 的论文参数量只标记 `paper_reported`，其余 B 项待官方 checkpoint 的同口径 CUDA 测量。正式比较只见 `01_metric_comparisons/` CSV。
