# Validation 统一评测

此目录与原始 `mlp到SNN` 隔离。所有原网络代码和冻结权重均以字节一致的快照使用，不改网络、不训练、不调阈值、不运行 Test。官方模型推理与官方指标代码核验是两件不同的事情。

## 当前范围

- 评测模型：H2 ANN、A1-H2 定点 ANN、S1-P Proposal-SNN + A1-H2 Refiner。
- 数据来源：用户指定的 `E:/experiment/FPGA上的SNN实现/anti-EUV/EV-UAV-dataset/val`。
- 仅复制原索引指定的 24 条 Validation，1,424,330 个事件，65,506 个目标事件。24 个文件 SHA-256 全部与原索引一致；`ev`、`ev_loc`、`evs_norm` 的事件数量、坐标、标签、目标身份和顺序逐条核对。
- 本机仅检测到 Intel Iris Xe 核显；用户确认没有官方预训练权重或已配置的 NVIDIA GPU 环境。因此 EV-SpSegNet 的真实 Validation 推理尚未完成，不能宣称已证明优于官方网络。
- 官方指标参考固定为提交 `33e2affb51a204676536e119803ab8494e2eb0a9`，保留下载的源文件及哈希。

本轮完整评测已完成，耗时 191.65 秒（含三个模型推理、两套指标、参考核验和保存预测，不是单模型推理延迟）。正式结果见 `results/validation_alignment.json`；数据核验见 `results/input_audit.json`。单序列冒烟结果位于 `results/runs/smoke.json`，不能当作完整评测结果。

## 本轮结果（2026-09-02）

以下全部是同一份 Validation 的 CPU 软件结果，不是 FPGA 或 Test 结果。官方 ACC 为目标事件召回率。

| 模型 | Event IoU | 官方 ACC / Recall | F1 | 统一 Pd | 统一 Fa（×10⁻⁴） |
|---|---:|---:|---:|---:|---:|
| H2 ANN | 86.1630% | 91.0772% | 92.5673% | 90.1092% | 0.0783931 |
| A1-H2 定点 ANN | 86.1582% | 91.0787% | 92.5645% | 90.1092% | 0.0784819 |
| S1 混合 SNN | 83.6273% | 88.2896% | 91.0838% | 85.9723% | 0.0782449 |
| 官方 EV-SpSegNet | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |

S1 相对 H2 的 Event IoU 下降 2.5357 个百分点；这是本地冻结流水线的对比，不是相对官方网络的结果。

统一口径前后，H2 的 Pd 从 90.0944% 变为 90.1092%，Fa 从 7.9163352×10⁻⁶ 变为 7.8393050×10⁻⁶；S1 的 Pd 从 85.9811% 变为 85.9723%，Fa 从 7.9341114×10⁻⁶ 变为 7.8244915×10⁻⁶。Event IoU 没有变化。

已执行的核验：

- 12 项单元测试全部通过，包括 100 张随机图的独立连通域对照、边界、阈值、微平均和官方参考差分。
- 三个模型的 TP/FP/FN/TN、候选数与历史结果逐项一致；历史口径 Pd/Fa 也复现。
- 统一定位整数计数与官方 CPU 适配参考一致，Pd/Fa 数值差为 0；IoU/ACC 最大浮点差约 2.90×10⁻⁸，来自 float32/float64 舍入。
- 全部 1,424,330 个事件参与事件级指标；官方严格时间边界下 1,386,745 个事件参与定位统计，37,585 个边界事件仅不参与 Pd/Fa，没有从 IoU 中删除。
- 三个模型均无误报图 uint8 计数溢出。
- 22 个原始快照来源文件的哈希均未改变；72 份预测包的事件索引、源数据哈希和重新计算的事件计数全部通过校验。

结论：H2 的 86.1630% Event IoU 在这份 Validation 上可复现，不是把总体 Accuracy 当成 IoU 的结果。与官方网络的胜负关系仍未知，跨模型任务状态保留为 `partial`。

## 统一协议

| 项目 | 固定规则 |
|---|---|
| 评测集 | 索引中固定的 24 条 Validation；不读取 Test 内容 |
| 事件支持集 | 所有原始事件，不随机抽样，不裁剪为候选事件 |
| 模型输入 | 各自冻结的输入表征，不为了评测改变模型；本地模型推理接口不接收 label/name 字段 |
| Proposal 拒绝事件 | 概率为 0，仍计入 FN/TN 等总数 |
| Event IoU | TP / (TP + FP + FN)，只评目标类，不是两类 mIoU |
| 官方 ACC | TP / (TP + FN)，即目标事件 Recall；总体 Accuracy 单列 |
| Precision / F1 | TP / (TP + FP)；2TP / (2TP + FP + FN) |
| 汇总方式 | 先对全部序列累加整数计数，再计算比例，不平均逐序列 IoU |
| 事件阈值 | 保留预先冻结阈值：本地三模型均为 0.2800000011920929；官方默认 0.9；不在 Validation 上搜索 |
| Pd/Fa 时间戳 | `ev_loc` 按官方数据加载器转 int64，再转 float32；不是 H2 原始小数毫秒时间戳 |
| 定位时间窗 | 50 ms，严格 `t > i*50` 且 `t < (i+1)*50`，不减去起始时间 |
| Pd 判据 | 每个时间窗每个非零目标身份计一次，正确目标事件占比 ≥ 0.0001 |
| Fa 判据 | 每窗将误报背景事件映射到 346×260 的 uint8 图，8 邻域连通域只统计一次 |
| Fa 分母 | 累计 `floor((max(t)-min(t))/50)` × 346 × 260；原值和 ×10⁴ 数值都保存 |
| 零分母 | 新评测器输出 null（未定义），不默认为完美精度；真实 Validation 的主指标不存在此情形 |

阈值数值相同不等于校准程度相同；统一协议固定的是“采用各模型预先冻结的工作点”，并非在 Validation 上为某个模型寻找优势阈值。此处也不把不同模型改成同一种输入特征，避免改变被评测模型本身。

## 与原实现的差异及核验边界

1. 原项目 `accuracy` 包含大量背景，不能对应官方 `ACC`；统一结果明确同时列出两者。
2. H2 历史 Pd/Fa 使用原始小数时间戳，官方使用整数毫秒坐标；官方严格开区间会排除恰在 50 ms 边界上的事件。统一结果保留历史 Pd/Fa、官方口径 Pd/Fa 和差值，Event IoU 不受此定位口径变化影响。
3. 用户 E 盘旧版官方 `utils/eval.py` 把连通域统计放在逐个误报事件的循环内，造成重复计数；固定版本已将其移到循环外。旧目录不改动，也不使用该旧 Fa 作为统一标准。
4. 为忠实对齐固定官方代码，保留其 uint8 计数溢出行为与末段时间窗/分母规则，并记录实际是否出现 uint8 溢出。将来若修正这些规则，必须使用新的协议版本、对所有模型同时重评，不能悄悄替换。
5. `reference_check.py` 执行下载的官方评测类：仅将指标张量 `.cuda()` 改为 `.cpu()`，并在缺少 OpenCV 时用独立并查集实现其所需的 8 邻域连通域数量。主评测器使用另一个 flood-fill 实现；两者有随机图和边界测试。**这不是原生 CUDA/OpenCV 环境的完整复现，更不是 EV-SpSegNet 的推理。**
6. 官方 event 指标以 float32 报告；整数计数必须一致，IoU/ACC 浮点差异容限 1e-7。Pd/Fa 的目标数、命中数、连通域数、分母必须精确一致。
7. 同时回归本地模型历史 TP/FP/FN/TN、候选事件数、历史 Pd/Fa；任一失败即停止，不生成成功的正式结果。

## 目录

- `baseline/`：本项目原网络、权重、索引和历史结果的冻结副本；运行不依赖原目录。
- `data/EV-UAV-dataset/val/`：仅 Validation 数据副本，Git 忽略。
- `official_reference/`：固定提交的官方指标/输入协议源文件，非完整官方网络工程。
- `prepare.py`：输入一致性核验、只读快照、官方参考文件下载。
- `metrics.py`：唯一统一评测器。
- `reference_check.py`：官方指标的 CPU 适配差分核验。
- `adapters.py`：H2/A1/S1 推理适配，不改冻结网络。
- `evaluate_validation.py`：固定 Validation 清单和阈值，生成逐序列、总体结果和事件概率。
- `test_metrics.py`：合成边界、指标公式、连通域和官方参考差分测试。
- `results/`：正式审计和统一评测 JSON；`results/runs/` 保存冒烟和逐事件概率。

## 运行

依赖本机已有 NumPy 和 PyTorch；本轮未安装或修改任何 Python/CUDA/WSL 环境。使用的解释器：

```powershell
$validationPython = 'C:/Users/sfr08/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
Set-Location 'D:/WangDao_EventSNN/H2-SNN-KV260/validation_compare'
& $validationPython -B -m unittest discover -s . -p 'test_*.py' -v
& $validationPython -B evaluate_validation.py --smoke
& $validationPython -B evaluate_validation.py
```

推理前验证冻结文件哈希；脚本不接受 Test 选项。为保护证据，不覆盖已有结果或预测目录；上述推理命令仅适用于尚未生成对应产物时，重复运行会拒绝覆盖。需要独立重复实验时，应使用新的实验目录，不删除首轮记录。

## 官方网络后续接入

1. 从[官方仓库](https://github.com/ChenYichen9527/EV-UAV)获取公开预训练权重：[百度网盘，提取码 ztp4](https://pan.baidu.com/s/1e6a_Ool5WZ3cBMPvoJvWbg?pwd=ztp4) 或 [Google Drive](https://drive.google.com/file/d/1nNZsckiN0qp2oo1uX40tU6oz3mUcrSHq/view)。记录实际文件名和 SHA-256；不得用随机初始化网络代替。
2. 准备支持 CUDA 的 NVIDIA GPU 环境，依官方说明安装相互兼容的 PyTorch、spconv 和 HAIS_OP。当前目录没有完成这一安装，不应直接照搬本机 CPU 环境运行官方网络。
3. 在新的官方工程副本中使用相同固定提交，保留原模型结构和预处理，只将数据入口限制为本目录审计清单中的 `val`。batch size=1、eval 模式、无随机采样，禁止直接运行默认访问 Test 的官方 `test.py`。
4. 对每个序列执行官方 `preds = preds[p2v_map]`，恢复为原事件顺序。输出必须与完整事件数相同，不能只保存体素概率或保留的正例。
5. 使用本目录 `metrics.py` 和 `reference_check.py` 评分，固定官方阈值 0.9；匹配源文件哈希、事件索引、标签与身份，再与本轮结果合并。全部预测都应由相同评测器评分；官方模型未完成前比较状态保持 partial。

不需要为了完成本阶段而现在购买 FPGA。官方参考推理需要 GPU 环境，H2/S1 的当前 CPU 评测不需要 GPU。

## 论文分数的使用

[论文第 5.1 节](https://arxiv.org/html/2506.23575v1#S5.SS1)明确在 Validation 选择最佳模型，然后在 Test 评测。表 2 的 55.18% IoU 可作为“论文报告 Test 结果”引用，不能填入“官方网络本次 Validation 实测”一栏，也不能与本次 Validation IoU 相减声称提升。

本轮验证的是用户提供数据与本地冻结索引的字节一致性，不等于已独立复核论文使用的数据发布版本、作者 checkpoint 训练过程、论文全部实验或官方 GPU 执行环境。
