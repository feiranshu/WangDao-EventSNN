# B、模型复杂度

项目正式模型仍只有 H2 与 H2-plus，均使用冻结的 exact INT8 Proposal+Refiner 权重；H2-plus 复用 H2 Proposal。按用户明确要求，`EV-UAV/` 另存 EV-UAV 原论文 EV-SpSegNet 的外部参照，它不因此成为本项目正式入选模型。`Charts and pictures/` 保存正式模型及论文参照的对比图。

## 主表指标与边界

| 编号 | 指标 | 本目录口径 |
| --- | --- | --- |
| A01 | 参数量 | H2/H2-plus 为 int8 矩阵中的权重元素数，包含零权重；共享 Proposal 只计一次。EV-SpSegNet 使用原论文 Table 2 的舍入报告值 4.0M，并明确标成论文值。 |
| A02 | 静态存储 | 冻结权重 int8 字节 + 每输出通道的 float32 scale 字节；不使用 ZIP/NPZ 文件长度，元数据及软件包装另列。 |
| A03 | 工作/状态内存 | 当前两个 ANN 没有 SNN 膜电位等持久神经元状态。提供输入相关的核心张量预算：patch、热图、基础事件特征、Refiner 特征、H2-plus 历史特征和输出概率的字节和。它不是同时存活的峰值。另在各划分事件最多的一条序列测 Python/NumPy 追踪堆峰值（不含原始输入与已加载权重）；它也不是 CPU RSS 或硬件 SRAM，不能与板测内存等同。 |
| A04 | 稠密 MAC | `80 窗 × 336 patch × Proposal 每调用 MAC + 实际候选事件数 × Refiner 每调用 MAC`；按一条 8 秒序列及每秒原始输入分别汇总。 |
| A05 | 有效 MAC/AC | 在每层的真实输入激活上，统计非零输入激活连接到非零权重的 MAC；当前 ANN 的 AC 为 0。该数字仅代表稀疏执行的潜在运算量，当前 NumPy exact INT8 回放仍执行稠密矩阵乘法。 |
| A06 | 神经元更新/其他运算 | 当前 ANN 的 SNN 神经元更新为 0；分别列出隐藏层量化变换元素数、Proposal sigmoid 输出元素数、Refiner softmax 的候选事件数、H2-plus 历史特征值数。不同类型不相加为一个“总 OPS”。 |

冻结源：`../../1、H2模型改进/implementation/software/weights/H2_D1_seed42.npz` 与 `A4-2_seed42_epoch008_int8.npz`，脚本核对其 SHA-256。输入只取 `implementation/data/index.json` 指定的 24 条 Validation 和 24 条 Test 完整序列，事件数和候选数与 A 中对应的 exact INT8 `per_sequence.csv` 逐条核对。每条序列包含 80 个 100 ms Proposal 窗；这与 A 中 Pd 的 50 ms 评分窗是不同用途。

`生成模型复杂度评测.py` 负责重算 H2/H2-plus 并生成三模型柱状图；`模型复杂度完整结果.json` 保存两个正式模型的完整逐序列数值。`EV-UAV/summary.json` 保存论文值、官方仓库当前代码核对值及缺失项。三模型图只在 A01 使用 EV-SpSegNet 的论文数据；A02—A04 明确标“原论文未报告”，不作推算。原有工作簿仍只服务两个正式模型的逐序列核对。

算法复杂度结果不是 KV260 板测功耗、速度或资源。实际上板后的 B01—B05/B07 指标在 C 目录记录。
