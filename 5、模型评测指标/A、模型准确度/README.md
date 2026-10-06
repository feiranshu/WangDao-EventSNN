# A、模型准确度

本目录保存**正式模型**的效果评测。直属的三个大目录保持固定：`0、原始H2模型/`、`1、H2-plus模型/` 和 `Charts and pictures/`。前两个目录保存各自的数值和可复核源数据；第三个目录只放面向比较的图表及数据。`生成目标事件图表.py` 从两个正式模型冻结的 `summary.json` 生成目标事件图和表。

| 位置 | 数据内容 | 阅读入口 |
| --- | --- | --- |
| `0、原始H2模型/统一int8评测/` | 与 H2-plus 同协议的 Train-dev、Validation、Test exact INT8 正式基线，含 summary、逐序列和 PR/阈值曲线数据。 | `0、原始H2模型/统一int8评测/README.md` |
| `0、原始H2模型/历史Step1结果/` | 原始 H2 的早期 Train-dev 诊断结果，不用于独立 Validation/Test 正式对比。 | `0、原始H2模型/README.md` |
| `1、H2-plus模型/float/`、`int8/` | seed 42 Train-dev 冻结结果；正式数值路径为 int8。 | `1、H2-plus模型/README.md` |
| `1、H2-plus模型/validation/`、`test/` | 独立 Validation/Test 的 float 与 int8 结果。 | 同上 |
| `1、H2-plus模型/stability/`、`与H2对比/` | 正式模型的其他 seed 稳定性复现及与 H2 的汇总。 | 同上 |
| `Charts and pictures/PR曲线对比/` | H2 与 H2-plus 在 Validation/Test 的 PR 图。 | PNG 图 |
| `Charts and pictures/指标对比表/` | AP、IoU、Precision、Recall、F1、Pd、Fa 和高密度指标。 | CSV + PNG |
| `Charts and pictures/目标事件指标/` | 不依赖背景 TN 的目标事件 Recall、漏检率、Event AP；EV-UAV 论文 ACC/Pd 与两正式模型的对比。 | 指标说明、CSV + PNG |

正式协议：`exact_int8_replay` 和 `evuav-33e2aff-official-compatible-v1+step3-density-20000`。H2 的工作点阈值为 0.30，H2-plus 为 0.40。Validation/Test 独立报告，不重新选阈值。总体 Accuracy 容易受大量背景真负例影响，目标事件主图主要使用 Event AP、Recall/论文 ACC，并辅以 Pd、Fa。原论文未报告的 EV-SpSegNet Validation 数据留空。

Windows 将 `.svg` 矢量图关联到 Microsoft Edge；按本项目文件约定，A 不再保存 SVG，正式图片保留 PNG，数值保留 CSV/JSON/NPZ。仅正式模型进入本目录；中间产物位于上一级的 `附：中间产物模型/`。
