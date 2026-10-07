# A、模型准确度

本目录只保存正式模型的冻结准确度结果，以及明确标注的 EV-SpSegNet 外部参照。找文件遵循一句话：**看结论去 01，查原数去 02，查口径去 03**。

```text
A、模型准确度/
├─ README.md
├─ 01、对比图表/
│  ├─ 指标汇总.csv
│  ├─ Validation/             # 01—06 六类图
│  └─ Test/                   # 与 Validation 同名同序
├─ 02、评测数据/
│  ├─ H2/{Validation,Test}/{事件级,目标轨迹级}/
│  ├─ H2-plus/{Validation,Test}/{事件级,目标轨迹级}/
│  └─ EV-SpSegNet/{论文参考.json,实测状态.json}
└─ 03、评测说明/
   ├─ 指标定义与评测规则.md
   ├─ 复现方法.md
   ├─ 历史资料索引.md
   └─ 协议与来源/目标轨迹协议.json
```

## 快速入口

| 想查什么 | 去哪里 |
| --- | --- |
| 三模型对比结论 | `01、对比图表/Validation/` 或 `Test/`；机器可读总表为 `指标汇总.csv` |
| H2/H2-plus 原始正式数值 | `02、评测数据/<模型>/<数据集>/` |
| 指标公式、边界和 EV-UAV 对比规则 | `03、评测说明/指标定义与评测规则.md` |
| 如何重算 | `03、评测说明/复现方法.md` |
| Train-dev、float、seed 稳定性等历史资料 | `03、评测说明/历史资料索引.md` 指向研究工程 |

## 当前状态

- H2：exact-INT8，阈值 0.30，Validation/Test 事件级和目标轨迹级均完成。
- H2-plus：A4-2 seed42 epoch8 exact-INT8，阈值 0.40，两类正式评测均完成。
- EV-SpSegNet：论文 Test ACC/Pd 可作外部参照；Validation、Event AP/mAP 未报告，目标轨迹等待 CUDA 实测，缺失值统一为 N/A。
- 事件级主结果包括 Event AP、IoU、Precision、ACC/目标事件 Recall、F1、Pd、Fa 及密度分层；目标级结果包括 HOTA、DetA、AssA、LocA、检测/关联分项和最长连续漏检 P95。

本目录不保存生成脚本、权重、缓存、Train-dev 调参结果、float 对照或 seed 稳定性实验，也不保存 SVG。正式生成代码统一位于 `1、H2模型改进/implementation/software/scripts/`。
