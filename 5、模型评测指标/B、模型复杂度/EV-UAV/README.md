# EV-UAV 原模型：复杂度参照

此目录保存 EV-UAV 原论文基线 **EV-SpSegNet** 的外部参照，不表示它成为本项目的正式入选模型。

## 可用于正式对比的数值

ICCV 2025 原论文 Table 2 只报告了 `Params = 4.0M`，因此 B 类主表只在 A01 参数量中加入这一项。论文还报告 `Infer. = 35.9 ms`，但测试硬件与本项目尚未完成的同条件板测不同；它属于 C 类部署指标，不加入 B 的算法复杂度图。

论文没有报告 A02 精确静态张量载荷、A03 工作内存、A04 稠密 MAC、A05 有效 MAC 或 A06 其它操作量。这些位置保持“未报告”，不由 4.0M 参数、论文延迟或当前仓库代码反推。EV-SpSegNet 是 ANN，因此“持久 SNN 神经元状态”“SNN AC”“SNN 神经元更新”属于不适用项，按统一表记 0。

## 官方仓库版本差异

官方仓库当前 HEAD 为 `33e2affb51a204676536e119803ab8494e2eb0a9`，配置为 `width=12`、`input_channel=4`。按该提交的 `evspsegnet.py` 与 `basemodel.py` 逐层展开，可训练参数是 **943,749**，FP32 参数张量为 3,774,996 byte。计数包含 `GDConv` 中未显式关闭的卷积 bias。这个结果与论文舍入报告的 4.0M 不一致，所以它只作为代码核对项，不替代论文原模型数值，也不混入三模型正式柱状图。

`核对EVUAV复杂度.py` 可复算当前仓库代码的参数量；`summary.json` 保存论文值、代码核对值、来源和缺失项。

来源：

- Nuo Chen et al., *Event-based Tiny Object Detection: A Benchmark Dataset and Baseline*, ICCV 2025, Table 2: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Event-based_Tiny_Object_Detection_A_Benchmark_Dataset_and_Baseline_ICCV_2025_paper.html
- 官方仓库：https://github.com/ChenYichen9527/EV-UAV
