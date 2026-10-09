# A 类指标定义与评测规则

## 正式配置

- H2：冻结 exact-INT8，阈值 0.30。
- H2-plus：A4-2、seed 42、epoch 8、冻结 exact-INT8，阈值 0.40。
- EV-SpSegNet：官方浮点 CUDA 数值路径，阈值 0.90；权重哈希在服务器回传导入后冻结。
- 事件级协议：`evuav-33e2aff-official-compatible-v1+step3-density-20000`。
- Validation 与 Test 只报告冻结配置，不参与调参；总体 Accuracy 受背景真负例影响，不作为主结论。

## 事件级指标

| 指标 | 含义 | 趋势 |
| --- | --- | --- |
| Event AP | 事件级 Precision–Recall 曲线面积，不使用背景 TN；不等同于边界框 COCO mAP。 | 越高越好 |
| Event IoU | `TP/(TP+FP+FN)`。 | 越高越好 |
| Precision | `TP/(TP+FP)`。 | 越高越好 |
| ACC / Target-event Recall | `TP/(TP+FN)`；本项目与 EV-UAV 兼容的 ACC 口径。 | 越高越好 |
| F1 | Precision 与 Recall 的调和平均。 | 越高越好 |
| Pd | 50 ms 目标窗检出率，用于和 EV-UAV 原论文对齐；判定较宽松，需结合 AP、Recall 和 Fa。 | 越高越好 |
| Fa | 按官方兼容实现统计的虚警率。 | 越低越好 |

序列宏平均、高密度和普通密度 Recall 用于定位模型在不同事件密度下的差异。所有正式汇总按 Validation、Test 分开呈现。

## 可观测事件轨迹 HOTA

轨迹协议为 `evuav-observable-event-track-hota-v1`，评估对象是模型与共享冻结目标聚类/关联适配器的组合。它不使用背景 TN。

1. 每条序列按 50 ms 左闭右开窗口划分，边界事件只统计一次。
2. GT 使用同一非零 `name` 的目标事件最小包围框，单像素框具有非零面积；`name` 是真实轨迹身份。
3. 预测端只使用事件坐标、时间和模型概率；阈值以上事件按空间邻近聚类，保留单事件候选。
4. 预测目标通过恒速位置预测、中心距离门限和一对一最优分配获得轨迹 ID；静默期间只保留内部状态，不补造预测框。
5. 聚类半径、关联距离和最大缺失窗口只用 Train-dev 理想目标事件输出选择一次，冻结值分别为 3 px、20 px/50 ms、20 个窗口。完整 27 组结果见 `protocols_and_sources/target_track_protocol.json`。

在 IoU 门限 $\alpha$ 下：

$$
DetA_\alpha=\frac{TP}{TP+FN+FP},\qquad
HOTA_\alpha=\sqrt{DetA_\alpha AssA_\alpha}
$$

正式 HOTA 对 0.05 至 0.95、步长 0.05 的 19 个门限取平均。DetA/DetRe/DetPr 衡量目标检测，AssA/AssRe/AssPr 衡量轨迹身份保持，LocA 衡量匹配目标定位质量；最长连续漏检 P95 按 IoU≥0.5 的连续可观测 GT 区间统计，越低越好。

GT 静默窗口不补框、不计漏检；该窗口出现预测仍计 FP。因此它是“可观测事件轨迹”评测，不能证明物理目标在静默期间仍被持续跟踪。

## EV-SpSegNet 论文值与正式补测

EV-SpSegNet 是正式软件对比模型。论文表 2公开的 Test Event IoU=0.5518、ACC=0.6502、Pd=0.7753、Fa=0.000163 标记为 `paper_reported`，不计为本项目同口径完整性通过。论文未报告 Validation、Event AP、其余事件指标和轨迹指标，必须等待 Linux+NVIDIA CUDA 官方浮点实测，禁止由论文聚合值推算。

依据：HOTA（IJCV，<https://doi.org/10.1007/s11263-020-01375-2>）、TrackEval（<https://github.com/JonathonLuiten/TrackEval>）与 EV-UAV（ICCV 2025，<https://arxiv.org/abs/2506.23575>）。
