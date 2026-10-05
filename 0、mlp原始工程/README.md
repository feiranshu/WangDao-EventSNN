# 望道项目

复旦大学 FDUROP 望道项目工作仓库。项目研究 EV-UAV 事件流目标检测中的 SNN、定点量化、近似计算与 FPGA 实现。

## 当前状态

截至 2026-09-06：

- 当前主线改为第一阶段 H2 系精度优先优化：先固定 S1 前端适配 Refiner，再研究 25 ms 因果 ANN、共享空间编码和双时间尺度记忆。第二阶段再转 SNN/近似计算，第三阶段做 Verilog/KV260。旧 S1/S3 和 RTL 均保留，不覆盖。
- 已实现 B1/B2 训练、C0—C4 因果模型及训练/评测、S1 误差归因与独立计时、完整 Train-dev 选点和阶段冻结入口。65 项软件测试通过；B1/B2、C0/C2/C4 的真实 Train 小样本冒烟已完成。**尚无正式多轮训练增益，也没有 GPU/KV260 能耗实测。**

- 数据、H2 ANN 基线、浮点 SNN、定点黄金模型和 PoT 消融均已完成。
- FPGA 对照已经冻结为 S1 exact 与 S3 approximate 两种 Proposal-SNN，共用 A1-H2 Refiner。两者均使用 int8 权重、24 位累加器和 24 位 Q12 膜电位。
- S2 混合 PoT 精度可接受但硬件收益有限，只作为消融，不替换 S1。
- EV-UAV 论文指标口径已归档到 `results/evuav_metric_alignment.json`。论文 Test 只能作外部参考；当前 Recall 对应论文 ACC，Fa 展示值为仓库原始值乘 `10^4`。
- STC 受限实验已经结束。时空相关性诊断有效，但 `λ_STC=0/0.05/0.10/0.20` 的完整 Train-dev 选择均未通过门槛；未导出 STC 权重，也未运行 HW-calibration、Validation 或 Test，详见 `results/stc_calibration.json`。
- S3 已完成并通过 Validation。冻结配置在第一层、第三层突触加法树及三个推理膜更新位置使用 2 位 LOA；Event IoU 为 83.2342%，相对 S1 下降 0.3931 个百分点，零饱和。
- 板卡无关 RTL 已覆盖 24 位 exact/LOA 加法器、固定平衡加法树、单步 LIF 和冻结第一层单输出通道。Vivado 2019.1 行为仿真共对齐 470 条 Python 黄金向量，其中 48 条来自真实 HW-calibration patch，覆盖第一层全部三种缩放。尚无 KV260 工程、目标器件综合、板级结果或事件相机直连实现。
- Test 共 24 条序列，其 S1 结果此前已在服务器运行并向研发公开，不能再称为未接触盲测。本阶段不读取 Test，不重跑官方模型；旧 Test 结果只作历史证据。新方案只在 Train-dev 选择、冻结后一次 Validation。

新成员或 AI 按以下顺序开始：

1. 阅读 `AGENTS.md`，了解协作、实验记录和 Git 规则。
2. 阅读本文件，建立数据、模型、代码和结果的对应关系。
3. 阅读 `PLAN.md`，确认当前阶段的实施协议、验收门槛和后续路线。
4. 执行 `git status --short`，保留他人未提交改动；再阅读当前任务涉及的代码。

当前仓库是唯一正式工作区。外部 Gazelle、EMS-YOLO、旧 EV-UAV 和 FPGA 工程只可作只读参考，不是运行时依赖，也不适用它们的 `AGENTS.md`。

原始立项申请书归档在 `materials/proposal/望道项目申请书_可阅读版.md`。它解释了 SNN、PoT、近似加法、动态精度和 FPGA 的研究动机，但其中的拟实施方案与前期结果不能替代当前仓库证据。申请书报告的 `mAP=0.446` 目前不可复现；申请书中的历史数据尺寸描述也不覆盖当前数据审计得到的 346×260。

## 仓库结构

```text
AGENTS.md   长期协作规则
README.md   新成员入口、当前事实和复现入口
PLAN.md     技术路线、当前任务和验收标准
data/       完整 EV-UAV 数据集与唯一正式划分
materials/  立项申请书等本地材料，含个人信息，不提交 Git
software/   Python 模块、脚本、测试和冻结权重
hardware/   RTL、测试平台、黄金向量与可重复仿真入口
results/    可追溯的正式 JSON 结果
tmp/        未分类临时内容，Git 全部忽略
```

训练 checkpoint、探索日志和冒烟结果只放 `results/runs/`，该目录不提交。正式数据只从 `data/EV-UAV-dataset/` 读取。

新增正式实验 JSON 使用统一顶层摘要：`schema_version`、`status`、`experiment_id`、`stage`、`data_role`、`primary_metrics`、`decision` 和 `conclusion`。详细逐序列结果仍保留在阶段字段中；历史 JSON 不为格式统一而重写。

## 环境

项目契约见 `pyproject.toml`：Python 3.9 及以上、NumPy 1.24 及以上、PyTorch 2.0 及以上。

本次第一阶段本机验证使用 `C:\python\python.exe`：Python 3.14.2、NumPy 2.4.1、PyTorch 2.11.0+cpu，无 CUDA。服务器仍优先使用已有 `evuav` Python 3.10 / PyTorch 2.5.1 环境，不要求更换成此本机版本。本机未安装 pyright，本次不将静态类型检查记为通过。

完成现有正式实验的参考环境为：

| 项目 | 版本或位置 |
|---|---|
| Python | 3.10.20 |
| NumPy | 2.2.6 |
| PyTorch | 2.5.1+cu121 |
| CUDA | 12.1 |
| Windows 解释器 | `C:\Anaconda\envs\gazelle-train\python.exe` |
| RTL 单元仿真 | Vivado Simulator 2019.1，路径 `C:\Xilinx\Vivado\2019.1` |
| KV260 目标工具 | Vivado ML Standard 2025.2.1，待下载安装 |

环境名是历史遗留名称，本项目不依赖 Gazelle、光芯片 SDK 或 Python 3.6。所有命令从仓库根目录执行。已有环境可安装当前包：

```powershell
python -m pip install -e .
```

Vivado 2019.1 只用于当前板卡无关 RTL 的编译和行为仿真，不支持 K26/KV260 的正式器件与 Board Flow。KV260 暂定为目标板但尚未购买；目标版本选用 Vivado ML Standard 2025.2.1，并与 2019.1 并行安装。2025.2.1 尚未在本机验证，正式综合前必须确认 K26 器件和 KV260 Board Flow 可用；在此之前不卸载 2019.1。安装时使用 Windows Web Installer 的 `Download and Install`，只选择 Vivado、Kria/Zynq UltraScale+ MPSoC 器件支持及需要的 cable driver，不安装只能编程和调试的 Lab Edition。任何大型环境安装仍应先按 `AGENTS.md` 停下并向用户给出完整方案。

## 数据契约

`data/index.json` 是唯一正式划分，共 147 条序列：

| 角色 | 数量 | 用途 |
|---|---:|---|
| `train_main` | 79 | 训练模型 |
| `train_dev` | 10 | 调参、早停和模型选择 |
| `hw_calibration` | 10 | 定点、近似计算和硬件校准 |
| `validation` | 24 | 方案冻结后的阶段评测 |
| `test` | 24 | 已有历史结果，本阶段禁用；独立泛化需另采新数据 |

每个 NPZ 的事件数组键为 `ev`，事件字段如下：

| 字段 | 含义 |
|---|---|
| `x`, `y` | 事件坐标，传感器尺寸为 346×260 |
| `t` | 单调递增时间戳，当前表征按毫秒处理 |
| `p` | 二值极性 |
| `label` | 二值事件标签 |
| `name` | 目标身份，`0` 为背景，正数为目标 |

NPZ 还包含 `ev_loc` 和 `evs_norm`。旧模型推理使用 `ev`，按 8 秒、80 个 100 ms 窗口处理。新因果模型只接受 `x/y/t/p` 四列，以 25 ms 半开块发布；标签与目标 ID 只用于训练和离线评分。`data/index.json` 保持不变，第一阶段读取时校验对应文件 SHA-256。本机数据入口是指向已有数据的目录联接，不随软件包上传。

只有数据发生变化时才重新运行完整审计，因为该步骤会遍历数据并更新哈希：

```powershell
python software/scripts/audit_data.py
```

## 历史冻结 S1 算法闭环

```text
EV-UAV 事件流
  → 100 ms 窗口，SNN 方案再切成 T=4 个微时间片
  → 336 个重叠 patch，每个 patch 生成 128 维 uint8 Proposal 特征
  → S1-P：128→48→24→16 的定点 Proposal-SNN
  → 候选路由
  → A1-H2：32→64→32→4 的定点 ANN Refiner
  → 事件概率、连通域、定位指标
```

FPGA 第一版采用 PS 预处理与后处理、PL 执行核心网络。原始 AER 直连事件相机、自适应窗口和复杂异步缓存不是第一版闭环的前置条件。

## 模型、权重与结果

下表 Event IoU 和 F1 均为 24 条 Validation 序列上的历史事件级指标，不是新因果模型结果。旧 S1 后来已有单独的服务器 Test 记录。

| 阶段 | 结构与结论 | Event IoU | F1 | 冻结权重 | 正式结果 |
|---|---|---:|---:|---|---|
| H2 | 全 ANN 基线，也是蒸馏教师 | 86.1630% | 92.5673% | `software/weights/H2_D1_seed42.npz` | `results/baseline_h2.json` |
| S0-P | 浮点 Proposal-SNN + H2 Refiner；冻结方案 | 83.6892% | 91.1204% | `software/weights/S0P_T4_POS15_KD50_S42.npz` | `results/s0p_validation_balanced.json` |
| S0-R | H2 Proposal + 浮点 Refiner-SNN；单级对照 | 84.1438% | 91.3892% | `software/weights/S0R_T4_POS10_KD50_S42.npz` | `results/s0r_validation.json` |
| S0-Full | 直接串联两个 SNN 后 Train-dev IoU 71.89%，停止继续 | 未运行 | 未运行 | 无 | 探索结果未提交 Git |
| S1 | 定点 Proposal-SNN + 定点 H2 Refiner；FPGA 主线 | 83.6273% | 91.0838% | `software/weights/S1P_INT8_ACC24_MEM24_Q12.npz` | `results/fixed_calibration.json`、`results/s1_validation.json` |
| S2 | 中间层 PoT、首末层 int8；消融 | 82.5451% | 90.4380% | `software/weights/S2P_POT_MIXED_EMIN4_EMAX3_ACC24_MEM24_Q12.npz` | `results/pot_calibration.json`、`results/s2_validation.json` |
| S3 | 第一、三层突触树与全部推理膜更新使用 2 位 LOA；近似主线 | 83.2342% | 90.8501% | `software/weights/S3P_LOA_SYN13_MEMALL_K2_ACC24_MEM24_Q12.npz` | `results/approx_exact_alignment.json`、`results/approx_calibration.json`、`results/approx_combination.json`、`results/s3_validation.json` |

补充结论：

- S0-P Proposal Recall 为 97.7956%，候选事件比例为 40.8780%。`results/s0p_validation_accuracy.json` 是另一训练目标的对照，不是主线模型。
- HW-calibration 计算代理显示，S0-P Proposal 的事件驱动突触运算为 H2 稠密 Proposal MAC 的 26.45%；包含相同 H2 Refiner 后为完整 H2 的 53.46%。这不是功耗或能效实测，详见 `results/compute_profile.json`。
- S1 相对浮点 S0-P 仅下降 0.0619 个百分点，且 HW-calibration 与 Validation 均无饱和。A1 定点 H2 Validation IoU 为 86.1582%。
- S2 相对 S1 下降 1.0822 个百分点。PoT 仅覆盖 Proposal 的 15% 权重，估算权重存储减少 5.625%，候选比例从 43.02% 升到 50.41%，第一层 uint8 乘法仍存在，因此不替换 S1。
- STC 诊断选出的 `1×1×3` 邻域 AUROC 为 0.8838，但正式训练没有转化为端到端收益。相对冻结 S0-P，STC05/10/20 的 Train-dev Event IoU 分别下降 1.0604/0.8292/1.3603 个百分点；Pd 虽提升 1.4874/1.4378/2.2310 个百分点，仍因 IoU 超过允许损失而全部停止。`λ_STC=0` 对照也下降 0.6506 个百分点，未满足公平对照门槛。
- S3 的 `k=0` 全量回归覆盖 10 条 HW-calibration 序列和 4,300,800 个 logits，输出及统计与 S1 逐位一致。S3 相对 S1 的 Validation Proposal Recall 提升 0.4824 个百分点，Pd 下降 0.0630 个百分点，候选比例增加 3.2159 个百分点。16,907,304,960 是 24 条 Validation 上的软件近似加法计数；冻结位置覆盖 Proposal 目标加法位置的 85.31%，不代表已经获得同等比例的 FPGA 功耗、面积或延迟收益。
- 队友报告的 `mAP=0.446` 没有代码、数据划分、权重或评测记录，当前不可复现，不能作为仓库基线或对比结论。

## 关键代码入口

| 路径 | 职责 |
|---|---|
| `software/wangdao/data.py` | 按索引定位和校验数据 |
| `software/wangdao/h2/` | H2 常量、事件表征、特征、NumPy/PyTorch 模型和导出 |
| `software/wangdao/snn.py` | 浮点 LIF、Proposal-SNN 与权重导入导出 |
| `software/wangdao/s0p.py` | S0-P 数据集、蒸馏和训练损失 |
| `software/wangdao/stc.py` | STC 连续块、可微热图、相关性诊断与辅助损失 |
| `software/wangdao/s0r.py` | S0-R 数据构造、模型和损失 |
| `software/wangdao/fixed.py` | A1、S1、S2 和 S3 的位精确定点推理、量化、LOA 与固定加法树 |
| `software/wangdao/metrics.py` | 事件级与定位指标 |
| `software/scripts/train_s0p.py`、`train_s0r.py` | S0 训练与参数扫描 |
| `software/scripts/analyze_stc.py`、`select_stc.py` | Train-dev STC 诊断与端到端候选选择 |
| `software/scripts/evaluate.py` | H2 Validation 基线评测 |
| `software/scripts/evaluate_s0p.py`、`evaluate_s0r.py`、`evaluate_s0full.py` | 浮点 SNN 评测 |
| `software/scripts/profile_compute.py` | ANN/SNN 计算代理统计 |
| `software/scripts/evaluate_fixed.py`、`evaluate_s1.py` | S1 校准与冻结评测 |
| `software/scripts/evaluate_pot.py`、`evaluate_s2.py` | S2 校准与冻结评测 |
| `software/scripts/evaluate_approx.py` | S3 的 `k=0` 精确对齐、HW-calibration 单因素扫描与组合选择 |
| `software/scripts/evaluate_s3.py` | 冻结 S3 与 S1 的唯一 Validation 对照 |
| `software/scripts/generate_rtl_vectors.py` | 生成并回读校验板卡无关 RTL 黄金向量 |
| `hardware/rtl/` | 24 位 exact/LOA 算术、单步 LIF 和第一层单输出参考数据通路 |
| `hardware/tb/`、`hardware/run_simulation.ps1` | 自检测试平台与 Vivado 2019.1 批量仿真入口 |
| `software/tests/` | 表征、SNN、定点与导出回归测试 |

## 第一阶段：服务器执行教程

先完成 B1/B2 机制验证和 C0，再按 C1→C2→C3 的顺序追加消融。C4 仅是条件精修研究实现，须在强 Refiner 有效之后启动；默认置信分数尚未校准，不可称为最优效率模型。复杂教师、蒸馏、INT8 QAT/整数导出、GPU 能耗采样仍待精度实验决定后实施，当前包不包含这些阶段的完成证据。

### 1. 上传、解压、配置环境

上传工作区生成的 `phase1_optimization_20260906.zip` 到服务器 `/export01/frshu/experiment/SNN-model/`。包内是 `software/`（含原冻结权重）、`data/index.json`、`pyproject.toml` 和三份文档；不含数据集、训练缓存、冒烟 checkpoint 或个人材料。不要覆盖原来的 `validation_compare`。

以下命令都在服务器 Bash 中执行，每个代码行是完整命令；不要复制提示符，也不要把命令里的网址改成 Markdown 链接。若同名目录已存在，请先检查，使用新的目录名，不删除已有结果。

```bash
cd /export01/frshu/experiment/SNN-model
unzip phase1_optimization_20260906.zip -d phase1_optimization
cd phase1_optimization
ln -s ../../EV-UAV-dataset data/EV-UAV-dataset
conda activate evuav
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check
nvidia-smi
python -c "import sys,numpy,torch; print(sys.version); print('NumPy',numpy.__version__,'Torch',torch.__version__,'CUDA',torch.version.cuda); print('GPU usable:',torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
find -L data/EV-UAV-dataset/train -maxdepth 1 -name 'train_*.npz' | wc -l
python -B -m unittest discover -s software/tests -v
```

Train 文件数应为 99，程序进一步按原索引划分为 79/10/10 并校验内容。`find -L` 用于跟随数据链接。如果 GPU 不可用，先把 `nvidia-smi` 和 Python 检查输出发回，不直接启动正式训练。已有 evuav 无需重复安装 torch；没有该环境时可新建 `conda create -n evuav python=3.10 pip setuptools -y`，再激活，使用一整行 `python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121` 和一整行 `python -m pip install numpy==2.2.6`。CUDA 包是否适配以服务器驱动和实际可用检查为准，不改系统驱动。

### 2. 先跑冒烟，确认本机到服务器的闭环

```bash
python -B software/scripts/diagnose_s1.py --smoke --cross-context --output results/runs/server_s1_diagnosis_smoke.json
python -B software/scripts/train_refiner_adaptation.py --variant B1 --smoke --device cuda --output-dir results/runs/server_B1_smoke
python -B software/scripts/train_causal.py --variant C0 --smoke --device cuda --output-dir results/runs/server_C0_smoke
python -B software/scripts/evaluate_causal.py --checkpoint results/runs/server_C0_smoke/epoch_001.pt --smoke --device cuda --output results/runs/server_C0_smoke_eval.json
```

B 冒烟使用 1 条 Train-main（最多 8192 个候选训练样本）、1 条完整 Train-dev；C 冒烟使用 1 条序列前 8 块、1 轮。当前 Train-dev 第一段没有目标事件，因此出现 `IoU=null` 正常，不应写成 0 或 100%。冒烟报告应为 `status=complete`、`decision=smoke_only`，选点应拒绝冻结。

### 3. 正式 B1/B2 与 C0（长任务，手动启动）

在一块 GPU 上顺序运行，不同时启动多份训练。建议 `tmux new -s phase1`，执行下方命令；按 Ctrl+B 后 D 退出连接而保持运行，`tmux attach -t phase1` 返回。每条命令完成后检查对应 `summary.json` 再启动下一条。

```bash
python -u -B software/scripts/diagnose_s1.py --run --cross-context --output results/runs/S1_train_dev_diagnosis.json
python -u -B software/scripts/train_refiner_adaptation.py --variant B1 --run --device cuda --epochs 20 --seed 42 --output-dir results/runs/B1_seed42
python -u -B software/scripts/train_refiner_adaptation.py --variant B2 --run --device cuda --epochs 20 --seed 42 --output-dir results/runs/B2_seed42
python -u -B software/scripts/train_causal.py --variant C0 --run --device cuda --epochs 20 --seed 42 --output-dir results/runs/C0_seed42
```

默认 B1 学习率 1e-4，B2/C0—C3 为 1e-3；AdamW、余弦退火、最多 20 轮。B0 为冻结整数 S1，B1 初始浮点 Refiner另列，避免混淆数值路径变化与训练改善。B1/B2 的最终阈值评测已集成到训练入口。

首轮完成时读取 `history[0]` 的耗时、数据等待和缓存统计，估算全部 GPU 小时；短冒烟不能直接线性估计完整数据耗时。C 模型默认单序列 TBPTT=8，缓存 256 MiB，并预取下一条序列。B 模型使用带数据/权重标识的候选磁盘缓存。日志不含真实功耗。

使用摘要定位最佳 C0 checkpoint，再进行完整 Train-dev 阈值评测：

```bash
c0_ckpt=$(python -c "import json,pathlib; p=pathlib.Path('results/runs/C0_seed42'); print(p/json.loads((p/'summary.json').read_text())['best_checkpoint'])")
python -u -B software/scripts/evaluate_causal.py --checkpoint "$c0_ckpt" --device cuda --output results/runs/C0_seed42_eval.json
```

### 4. 逐项增加共享编码、记忆和强 Refiner

只在前一组结果已看清楚后运行下一组，不把全部结构自动排队。下面以 C1 为例，其余把 `C1` 同时改为 `C2` 或 `C3`：

```bash
python -u -B software/scripts/train_causal.py --variant C1 --run --device cuda --epochs 20 --seed 42 --output-dir results/runs/C1_seed42
candidate_ckpt=$(python -c "import json,pathlib; p=pathlib.Path('results/runs/C1_seed42'); print(p/json.loads((p/'summary.json').read_text())['best_checkpoint'])")
python -u -B software/scripts/evaluate_causal.py --checkpoint "$candidate_ckpt" --baseline results/runs/C0_seed42_eval.json --device cuda --output results/runs/C1_seed42_eval.json
```

模型含义：C0=因果 MLP；C1=共享空间 CNN；C2=C1+双衰减记忆；C3=C2+所有候选走强 Refiner；C4=双头条件精修。无记忆对照使用 C2 的 `--no-memory`，其余训练参数不变；容量对照使用 C0 的 `--mlp-hidden`，先根据 `resources.parameter_count` 匹配候选参数量，不能任意选宽度就称为等容量。

C4 必须从完成的 C3 初始化（先按 C3 评估理想路由空间，再决定是否开展）：

```bash
c3_ckpt=$(python -c "import json,pathlib; p=pathlib.Path('results/runs/C3_seed42'); print(p/json.loads((p/'summary.json').read_text())['best_checkpoint'])")
python -u -B software/scripts/train_causal.py --variant C4 --initialize-from "$c3_ckpt" --learning-rate 0.0001 --run --device cuda --epochs 20 --seed 42 --output-dir results/runs/C4_seed42
```

当前 C4 的不确定性、稀疏事件和历史冲突规则是研究初值；温度校准/路由阈值选择和量化路径尚未实施，不据默认 C4 结果认定最终部署方案。输出筛选阈值与路由用阈值分别记录，不可在 Test 上调。

正式配置最多 12 种，由 `results/runs/phase1_registry.json` 计数；改变 seed 不重复计数。不可删除登记文件来绕过预算。保留两个最终候选补 seed43/44，必须使用相同预算与采样；默认导出 seed42，不挑最好的种子。C4 每个种子对应同种子的 C3 前驱。

### 5. 阅读结果、冻结及一次阶段 Validation

`evaluate_causal.py` 报告包含 `evaluation.curve`（完整事件指标）、`event_pr_curve`、逐序列指标、候选/深层比例、排队回放和状态占用；连续分数及原始索引保存在同名 `_predictions/`。`selection` 先要求相对 C0 的 Pd 下降≤0.5个百分点、Fa 增加≤10%，再最大化 IoU。0.28/0.9是额外参考工作点，不是网络结构增益。Recall=目标事件召回，Pd=官方50 ms定位协议下目标实例检出率，两者不同；Fa列显示时乘1e4。

等正式消融、多种子和阈值全部冻结后，才执行以下命令。例中 C2 只是路径示例，必须换成实际通过筛选的候选：

```bash
python -B software/scripts/freeze_phase1.py freeze --evaluation results/runs/C2_seed42_eval.json --output results/runs/phase1_accuracy_frozen.json
python -u -B software/scripts/freeze_phase1.py validate --manifest results/runs/phase1_accuracy_frozen.json --device cuda --output results/phase1_accuracy_validation.json
```

冻结会检查十条完整 Train-dev 的文件名、事件数、协议、索引及 checkpoint 哈希；冒烟不能进入 Validation。Validation 启动记录会保留以防静默重跑。中断后不要删除该记录或另开冻结文件绕过保护，应先确认已经使用的数据与失败位置。所有训练入口拒绝覆盖已有目录，目前不支持自动断点续训；保留失败目录，必要时以新目录重启并登记原因。

完成首批正式训练后需要交回：B1/B2/C0 的 `summary.json`、`S1_train_dev_diagnosis.json`、`C0_seed42_eval.json` 和 GPU 型号/显存输出。无需上传原数据集或候选缓存。之后才根据误差与耗时决定 C1—C4、教师或 QAT 的资源投入。

## 后续硬件路线（本阶段暂停）

S3 已冻结，不再扫描近似位数、插入位置或阈值，也不重复运行其 Validation。板卡无关算术、LIF 和第一层单输出参考数据通路已有黄金向量；完成第一阶段及 SNN 转换后再恢复层级流水、三层网络与目标器件综合。

NumPy 是 S3 权威黄金模型；CUDA 后端只加速相同的整数位级规则。软件结果不得表述为 FPGA 资源、功耗或时延结果。

STC 训练入口保留用于复现。它默认后台预取下一条 Train-main 序列，并跨 epoch 缓存 Train-dev 不可变表征；四个候选仍共享相同数据。两轮等价性冒烟确认启用与禁用优化时全部训练指标和四份最终权重完全一致。每轮 `history.json` 的 `performance` 会记录数据等待比例、缓存命中和内存占用；本次正式 20 轮训练在该统计加入前启动，因此不能用其历史记录反推完整训练的 CPU 等待比例。

目标板暂定为 AMD Kria KV260 Vision AI Starter Kit（`SK-KV260-G`），目前尚未购买。Vivado ML Standard 2025.2.1 已选为待安装的目标工具；PS–PL 数据通路和事件相机物理接口仍未冻结。S3 是 FPGA 无关的软件黄金模型；当前 2019.1 仿真不构成目标器件综合或板级证据。

## 最小检查与复现命令

文档改动无需运行模型。修改算法或公共接口后至少执行：

```powershell
python -m unittest discover -s software/tests -v
pyright
```

重新生成第一批 RTL 向量并执行板卡无关行为仿真：

```powershell
& "C:\Anaconda\envs\gazelle-train\python.exe" software\scripts\generate_rtl_vectors.py
& .\hardware\run_simulation.ps1
```

仿真结果保存到 `results/rtl_unit_simulation.json`。该结果只证明 RTL 行为与 Python 黄金向量一致，不是 KV260 综合或 FPGA 实测。

复现三个核心冻结结果：

```powershell
python software/scripts/evaluate.py
python software/scripts/evaluate_s1.py --experiment-id S1-FIXED-validation-v1 --output results/s1_validation.json
python software/scripts/evaluate_s2.py --experiment-id S2-POT-validation-v1 --output results/s2_validation.json
python software/scripts/evaluate_s3.py --device cuda --experiment-id S3-LOA-validation-v1 --output results/s3_validation.json
```

这些命令会覆盖同名正式 JSON，只有明确需要复现时才运行。训练、实验记录、长任务授权和 Git 提交规则以 `AGENTS.md` 为准。
