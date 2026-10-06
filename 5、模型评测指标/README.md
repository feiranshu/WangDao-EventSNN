# 模型评测指标：统一标准

本目录只让**正式入选模型**进入 A—D 四类正式评测。目前的正式模型是 H2 与 H2-plus；历史实验 ID `A4-2` 在对外结果中写作 H2-plus。`附：中间产物模型/`只保存研究过程和未入选模型；除非用户明确指定模型及所需指标，不对其中任何模型运行或更新 A—D 正式评测，也不将其混入正式对比图。

## 总目录

| 目录 | 作用 | 正式指标及当前状态 |
| --- | --- | --- |
| [`A、模型准确度/`](A、模型准确度/README.md) | 在冻结的数据和评测协议下衡量目标检测效果；保留正式模型各自的结果与对比图。 | Event AP、Event IoU、Precision、目标事件 Recall（EV-UAV 的 ACC）、F1、Pd、Fa、PR 曲线以及高密度结果。H2/H2-plus 已完成 Validation、Test。 |
| [`B、模型复杂度/`](B、模型复杂度/README.md) | 衡量网络算法本身的参数、存储、计算和运行内存需求；软件阶段可评测。 | A01 参数量、A02 静态存储、A03 工作/状态内存、A04 稠密 MAC、A05 有效 MAC/AC、A06 神经元更新和其他关键运算。H2/H2-plus 已完成 Validation、Test；EV-SpSegNet 仅作为明确指定的论文外部参照。 |
| [`C、部署效率和硬件成本/`](C、部署效率和硬件成本/README.md) | 衡量实际部署后的速度、功耗与资源。 | B01 模型延迟、B02 端到端延迟、B03 吞吐/实时性、B04 运行功耗、B05 单位任务能耗、B07 硬件资源/面积。当前没有完整的同条件上板实测，不填正式结果。 |
| [`D、ANN和SNN 差异、可靠性和研发成本/`](D、ANN和SNN%20差异、可靠性和研发成本/README.md) | 衡量转换或近似计算是否以效果换取成本收益；目录名保留现有结构，实际只采用已筛选的主表指标。 | C01 时间步—效果—成本曲线；C02 ANN→SNN、量化、PoT/LOA 各自的效果损失。等待正式 SNN 转换及可比结果，不填正式结果。 |
| `附：中间产物模型/` | 保存未入选模型、消融、对照和历史诊断证据。 | 默认不参与上面四类评测；仅在用户明确要求时单独计算。 |

## 正式模型进入目录的规则

1. 只有经研究决策明确接受为正式模型，才在 A—D 的结果和对比图中新增一列或一个模型目录。模型的权重、数值格式、阈值、数据索引和评测脚本版本必须可追溯；中间产物不自动晋级。
   用户明确要求加入的论文基线可作为“外部参照”出现在对比图中，但必须与项目正式模型分开标注；论文未报告项保持缺失，不推算，也不改变正式模型名单。
2. A 与 B 对同一正式模型使用冻结部署配置；同一指标要统一数据集、观测窗口、模型组成和统计单位。现有 H2/H2-plus 的正式比较采用 `exact_int8_replay`，A 的指标协议为 `evuav-33e2aff-official-compatible-v1+step3-density-20000`。H2 的正式工作点阈值为 0.30；H2-plus 为 seed 42、epoch 8、阈值 0.40。
3. Train-main 训练，Train-dev 选择，HW-calibration 选择硬件相关设置；Validation 和 Test 只作冻结后的独立报告，不用 Test 反向选择模型、阈值、量化或近似参数。正式结果按 Validation/Test 分开展示。
4. A 用目标相关指标判断效果；背景真负例很多时，常规总体准确率容易被抬高，不作为主结果。EV-UAV 兼容口径的 ACC 对应目标事件 Recall。Event AP 是事件级 PR 面积，不等于边界框检测 mAP。Pd 不含背景真负例，但其 50 ms 窗口判定较宽松，须结合 Recall、AP、Fa 阅读。
5. B 将 Proposal、条件触发的 Refiner 和全部已执行的时间步计入。参数与静态存储是模型属性；稠密计算量是结构上界；有效 MAC/AC 按输入和非零连接统计，是稀疏硬件可利用的操作量，不代表普通 CPU/GPU 已跳过这些运算。工作内存的理论口径与实际峰值必须分开标注。
6. C 以同一平台、同一输入流和明确的测量边界报告模型内核与完整系统。板级功耗、PL 分轨功耗、综合估计分别命名；不能互相代替。D 需要冻结 ANN 教师、转换模型、时间步、读出和数值格式，逐项报告转换、量化、近似带来的差值。
7. 每项结果都给出单位、统计对象、数据划分、软件/板测/估算状态和源文件。缺少实测时标“未测/不适用”，不能填 0 或以理论运算量推断真实功耗。比较模型优劣时并列呈现效果和资源，不使用未约定权重的单一总分。

## 当前进度

| 正式模型 | A 准确度 | B 复杂度 | C 部署效率/硬件成本 | D ANN/SNN 差异 |
| --- | --- | --- | --- | --- |
| H2 | 已完成 | 已完成 | 未正式评测 | 不适用当前 ANN 版本 |
| H2-plus | 已完成 | 已完成 | 未正式评测 | 不适用当前 ANN 版本 |

## 指标依据

- [NeuroBench，Nature Communications 2025](https://www.nature.com/articles/s41467-025-56739-4)：算法层的模型体积、连接/激活稀疏性、SynOps，以及系统层的正确性、速度和能耗。
- [MLPerf Tiny Benchmark，NeurIPS 2021](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/da4fb5c6e93e74d3df8527599fa62642-Abstract-round1.html)：资源受限设备的准确性、延迟和能耗联合评价。
- [Efficient Processing of Deep Neural Networks，Sze 等](https://arxiv.org/abs/1703.09039)：网络计算、数据搬运与硬件成本。
- [SpikedAttention，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7c9341ad0263428b5057d92f4d88dfa0-Abstract-Conference.html)：SNN 能耗核算需纳入膜电位更新和数据搬运。
- [Optimal ANN-SNN Conversion，ICLR 2022](https://yuzhaofei.github.io/papers/22-ICLR-Optimal%20ANN-SNN%20Conversion%20for%20High-accuracy%20and%20Ultra-low-latency%20Spiking%20Neural%20Networks.pdf)：时间步、转换效果和延迟的权衡。
