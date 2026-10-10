# B 类指标定义与评测规则

本目录评估算法本身的参数、静态存储和运算成本，不等同于板级速度、功耗、能耗或 FPGA 资源。H2/H2-pro 使用与 A 相同的冻结 exact-INT8 Proposal+Refiner，共享 Proposal 只计一次；EV-SpSegNet 使用严格加载官方 checkpoint 的浮点 CUDA 数值路径。H2-plus 已退出正式注册。

| 编号 | 指标 | 统一口径 |
| --- | --- | --- |
| B01 | 参数量 | 严格加载正式权重后的可训练参数元素数；包含零权重。 |
| B02 | 静态存储 | 推理 `state_dict` 参数与持久 buffer 的 tensor payload；不使用 NPZ/ZIP/checkpoint 文件大小。 |
| B03 | 稠密 MAC | 同一正式前向调用中按完整结构连接计算的 MAC；稀疏卷积按实际活跃输出位置乘完整卷积核与通道连接。 |
| B04 | 有效 MAC/AC | 正式数值路径实际利用稀疏性后的 MAC 与 SNN 脉冲 AC 分开统计；普通 ANN 的 AC 按定义为 0。 |

## B04 合并比较口径

有效 MAC 与有效 AC 必须保留为两个独立字段，便于分析 ANN/SNN 运算组成；但与 B03 比较时必须先统一统计范围，再使用：

`B04 有效运算总量（op/sequence） = 有效 MAC（MAC/sequence） + 有效 AC（AC/sequence）`

正式主表同时给出三个 B04 子项：有效 MAC、有效 AC、有效运算总量。跨模型结论使用“有效运算总量 / 稠密 MAC”，不得用单独的 MAC 或 AC 代表全部有效运算。这里采用操作等价计数，每次 MAC 与每次 AC 各计 1 次；它不是能耗模型，不表示两者的焦耳成本相同。

参数量和静态存储是模型通用属性，只出现一次；B03/B04 按 Validation、Test 分开。H2/H2-pro 各划分 24 条序列，采用同一首事件锚定完整 100 ms 窗口前缀；每条闭合窗口数按序列记录。B03/B04 只计 Proposal 和条件 Refiner exact-int8 MAC，不含特征构建、后处理、CPU 延迟或内存工作区。

旧编号映射为 A01→B01、A02→B02、A04→B03、A05→B04。旧 A03 工作/状态内存和旧 A06 辅助算子不再属于正式主表，历史值随旧文件归档。EV-SpSegNet 是正式软件对比模型：论文约 4.0M 参数仅标记为 `paper_reported`；精确 B01/B02 与 Validation/Test 的 B03/B04 由服务器包实测，不能由论文运行时间或当前源码静态推算替代。
