# H2-S1 模型改进研究：七步实施状态

本目录实现 `H2-S1_模型改进研究计划_v2.0`。目标顺序为准确率优先、复杂度其次；研发阶段只使用 Train-main 与 Train-dev。服务器的逐条命令和回传清单见 [`README_SERVER.md`](README_SERVER.md)。

## 1. 当前结论

步骤1已经在全部10条Train-dev上完成。1,237,183个事件中共有4,421个FN，其中冻结Proposal造成478个，Refiner造成3,943个；后端占89.19%，因此A1原结构后端适配是第一优先。解除门控会把Pd从84.30%提高到86.54%，但IoU降至69.91%、Fa明显增加，不支持直接取消门控。

步骤2—4的训练代码和本地真实数据闭环已经完成，但正式训练依赖服务器。现在没有任何新模型可以宣称超过H2。步骤5在线分支依赖主精度结论，暂不启动。步骤7的冻结和一次性Validation/Test入口已实现，但没有运行任何受保护数据。

## 2. 七步完成状态

| 步骤 | 本地已完成 | 尚未完成及依赖 |
| --- | --- | --- |
| 1. H2错误归因 | 完整10条Train-dev；门控/后端FN守恒；真实全事件分类；阈值曲线、分层、失败窗和逐序列结果 | 服务器可复跑作环境一致性确认，不是后续阻塞项 |
| 2. 不加层的H2改进 | A1原结构适配；A2全部正样本加困难/随机负样本；冻结Proposal、候选缓存、正负损失和公平选点 | 79条Train-main、10条Train-dev、最多20轮，先跑seed42 |
| 3. 最小新信息 | A3四个前序窗特征；同维同参A3-null；未来泄漏和初始同构测试 | A3证据目前较弱，等待A1/A2正式剩余错误后决定是否训练 |
| 4. 显式减层与蒸馏 | S-R1/S-R2、直接训练与KD、二类logit KL、同初始化、复杂度统计、版本化浮点导出；S-R1两条真实数据冒烟通过 | 必须先确定正式A*；S-R2只在S-R1精度不足时启动 |
| 5. 可选在线化 | 原C0—C4入口仍在；补齐严格no-memory同参数对照和显式无门控诊断 | 主精度结论稳定且确有在线需求后才可正式训练；当前停止 |
| 6. 统一选点与缺口修正 | H2/候选统一Pd/Fa约束；H2候选缓存；解除门控真分类；浅层版本化导出；no-memory保留融合层；因果整序列无门控输出 | 服务器结果产生后才能形成公平模型表和实际时延结论 |
| 7. 冻结后一次验收 | 多种子门槛、哈希和完整Train-dev检查、模型/阈值冻结、一次性运行标记、Validation后才能Test、序列级配对区间 | 需先有A*的seed42/43/44；可选S*也需三种子。当前没有运行Validation/Test |

## 3. 已固定的本地证据

H2完整Train-dev：

| 工作点 | IoU | Pd | Fa×1e4 |
| --- | ---: | ---: | ---: |
| H2-F，阈值0.28 | 77.9478% | 84.3020% | 0.14518 |
| H2-R，公平阈值0.30 | 78.0533% | 84.0537% | 0.14074 |
| 真正解除门控，阈值0.28 | 69.9070% | 86.5375% | 0.35533 |

`train_089.npz`和`train_092.npz`合计贡献3,756/3,943个Refiner FN以及455/478个Proposal FN。正式实验必须报告逐序列变化，不能只给micro汇总。

步骤2/3一轮冒烟只用于验证程序：A1、A2、A3、A3-null均能训练、保存、回读和公平选点。步骤4的S-R1直接训练及KD也完成一轮冒烟，并生成`student_float32_v1.npz`。冒烟教师明确标为`engineering_smoke_only_unqualified_h2_reconstruction`，结果不能用于研究结论。

## 4. 模型复杂度口径

| 模型 | Refiner | 串行权重层 | 矩阵权重总数 | 相对H2 |
| --- | --- | ---: | ---: | ---: |
| H2/A1/A2 | 32→64→32→4 | 6 | 11,904 | 基线 |
| S-R1，32维输入 | 32→32→4 | 5 | 8,832 | -25.81% |
| S-R2，32维输入 | 32→64→4 | 5 | 9,984 | -16.13% |

若A*使用36维A3/A3-null输入，脚本会按实际维度重新计算，不能照搬上表百分比。这里统计矩阵权重，不等同于实际延迟、显存或能耗。

## 5. 新增和主要入口

```text
software/wangdao/h2_research.py
    步骤1—3：冻结H2、历史特征、采样和公平选点。

software/scripts/diagnose_h2.py
software/scripts/train_h2_refiner.py
    步骤1—3的诊断与训练入口。

software/wangdao/h2_students.py
software/scripts/train_h2_student.py
    步骤4：S-R1/S-R2、直接/KD训练、复杂度和版本化导出。

software/wangdao/causal.py
software/wangdao/causal_training.py
software/scripts/evaluate_causal.py
    步骤5/6：因果契约、同参no-memory和显式无门控诊断。

software/wangdao/h2_release.py
software/scripts/release_h2_family.py
    步骤7：三种子冻结、一次性Validation/Test与配对区间。
```

## 6. 数据与选择边界

- Train-main用于训练；Train-dev用于checkpoint与阈值；HW-calibration只用于后续数值校准。
- 冻结前禁止读取Validation/Test。冻结入口要求完整、非smoke的10条Train-dev证据。
- A*必须有seed42/43/44同配方结果。冻结门槛为：seed42相对H2-R至少+0.5个百分点，三种子平均增益为正，至少两个种子为正。
- S*必须保持高于H2，且seed42相对A*损失不超过0.3个百分点。
- H2-F定义可行区间；所有候选使用同一阈值网格和Pd/Fa约束。单纯调阈值的收益只能记为工作点收益。
- A3必须优于同维同参A3-null，才能把收益归因于历史信息。
- S0和S-KD使用相同随机种子、输入、候选、采样、训练预算；KD温度2、权重0.2是待验证起点。
- Test命令必须提供与冻结清单匹配的已完成Validation结果。Validation或Test启动标记已经存在时，程序拒绝静默重跑。

## 7. 验证记录与服务器复核

后续正式运行统一使用服务器上已有的 Conda 环境 `evuav`，不再创建项目内 `.venv`。本次82项回归是在 Windows 本地环境完成的工程验证；它不替代服务器复核，也不作为后续正式实验环境。服务器上传、环境记录和复核命令以 `README_SERVER.md` 为准。

```bash
conda activate evuav
python -m pip install -e . --no-deps --no-build-isolation
python -B -m unittest discover -s software/tests -v
```

本地验证结果：**82/82通过**。新增检查覆盖：

- S-R1/S-R2预注册复杂度；
- 直接/KD相同初始化；
- 教师无梯度、KL方向和温度平方；
- 浅层checkpoint和NPZ往返；
- no-memory保留融合层与参数量并忽略历史状态；
- force-deep显式返回每个事件的无门控概率；
- A*三种子一致性和开发门槛；
- Test在读取文件前要求匹配Validation；
- 95%区间按序列重采样，不把相关事件当独立样本。

本地结果位于`results/runs/local_*`。`data/EV-UAV-dataset`是指向现有数据集的目录联接，原始数据没有复制进本目录。

## 8. 当前停止点

现在应在服务器完成A1/A2正式训练并返回摘要。只有结果支持时才进入A3；只有选出A*后才运行S-R1；S-R2、在线C0—C4和受保护评测都继续受条件门控制。完整服务器顺序、命令和回传文件见`README_SERVER.md`。

## 9. 完整目录与逐文件职责

下面的目录从外层`1、H2_S1模型改进研究计划/`开始，覆盖当前保留的文件夹和文件。`archive/`是历史快照；当前研究、服务器运行和后续修改均以`implementation/`及外层当前版计划为准。`results/runs/`中的内容是本地诊断、冒烟或可再生中间产物，不能当成服务器正式训练结果。

```text
1、H2_S1模型改进研究计划/
├─ H2-S1_模型改进研究计划_v2.0_可编辑文本.md
│  └─ 当前研究计划的可编辑源文件；说明目标、七步路线、门槛、预算和证据边界。
├─ H2-S1_模型改进研究计划_v2.0.pdf
│  └─ 与可编辑文本对应的当前阅读版计划。
├─ H2_S1_steps1to7_20260923.zip
│  └─ 当前服务器上传入口；包内以implementation内容为根，原始数据集不打包。
├─ archive/
│  ├─ H2_S1_steps123_20260916.zip
│  │  └─ 只完成步骤1—3时期的旧上传快照。
│  ├─ H2_S1_steps1to7_20260923_pre_agents_refresh.zip
│  │  └─ 更新协作规则AGENTS前的七步实现快照。
│  ├─ H2_S1_steps1to7_20260923_pre_conda.zip
│  │  └─ 服务器环境改为统一使用evuav Conda环境前的快照。
│  ├─ H2_S1_steps1to7_20260923_pre_docs_20260924.zip
│  │  └─ 2026-09-24文档整理前的实现快照。
│  ├─ H2_S1_steps1to7_20260923_pre_final_audit.zip
│  │  └─ 最终交付审计前的实现快照。
│  ├─ H2_S1_steps1to7_20260923_unix_meta.zip
│  │  └─ 保留Unix元数据版本的历史打包快照。
│  ├─ H2-S1_模型改进研究计划_v2.0_可编辑文本_pre_20260924.md
│  │  └─ 2026-09-24修订前的v2.0可编辑计划。
│  ├─ H2-S1_模型改进研究计划_v2.0_pre_20260924.pdf
│  │  └─ 2026-09-24修订前的v2.0 PDF。
│  ├─ H2-S1_模型改进研究计划.pdf
│  │  └─ 被v2.0取代的早期研究计划。
│  └─ phase1_optimization_20260906.zip
│     └─ 第一阶段优化实现的早期归档包。
└─ implementation/
   ├─ AGENTS.md
   │  └─ 本目录的协作规则：证据等级、数据边界、验证、服务器环境、打包和Git约定。
   ├─ README.md
   │  └─ 当前状态唯一摘要：已完成事项、关键结论、入口、停止点和本目录说明。
   ├─ README_SERVER.md
   │  └─ 服务器正式运行的唯一逐条命令来源；规定阶段门、日志和回传清单。
   ├─ PLAN.md
   │  └─ 完整路线图和历史设计记录；当前状态与PLAN冲突时以README为准。
   ├─ pyproject.toml
   │  └─ Python包、最低版本、numpy/torch依赖、setuptools布局和Pyright检查范围。
   ├─ data/
   │  ├─ index.json
   │  │  └─ 唯一冻结的数据索引；记录147条序列的哈希、统计和Train-main/Train-dev/
   │  │     HW-calibration/Validation/Test角色，开发阶段不得重写。
   │  └─ EV-UAV-dataset/
   │     └─ 指向本机正式EV-UAV数据集的目录联接；服务器上需重建为实际数据路径，
   │        不复制进上传包。其train/val/test文件由index.json约束，不在此逐份登记。
   ├─ results/
   │  ├─ phase1_implementation_smoke.json
   │  │  └─ 第一阶段软件集成的历史冒烟记录；只证明短流程可运行，不证明精度提升。
   │  └─ runs/
   │     ├─ local_H2_diagnosis_full.json
   │     │  └─ 步骤1完整10条Train-dev错误归因、阈值曲线、逐序列指标和失败窗证据。
   │     ├─ local_H2_diagnosis_smoke.json
   │     │  └─ 单条train_002的步骤1短冒烟，验证诊断入口和输出结构。
   │     ├─ h2_candidate_cache_v2/
   │     │  ├─ train_000_186bb001c45b62be563414e23ea21934438c04f827acdcea3a6452fd4e47755b.npz
   │     │  ├─ train_000_6838890559568ba916a41283b91da14732b64b4d914b21d0d06377df1324eaa2.npz
   │     │  ├─ train_002_6eac6d9d6b5e0b901cc0307319609352823d6421e6e1df6594c562d5705c74c6.npz
   │     │  └─ train_002_86e05c40c0b950583779481a1f850575dc9aaeffb354bdce30e94e1b97071fdd.npz
   │     │     └─ 冻结H2 Proposal生成的候选/表征缓存；哈希区分配方，可删除重建，
   │     │        不属于正式结果，也不进入上传包。
   │     ├─ local_H2_diagnosis_full_predictions/
   │     │  ├─ train_002.npz
   │     │  ├─ train_020.npz
   │     │  ├─ train_034.npz
   │     │  ├─ train_039.npz
   │     │  ├─ train_041.npz
   │     │  ├─ train_050.npz
   │     │  ├─ train_060.npz
   │     │  ├─ train_066.npz
   │     │  ├─ train_089.npz
   │     │  └─ train_092.npz
   │     │     └─ 完整Train-dev诊断的逐事件预测；用于复核错误归因和逐序列差异。
   │     ├─ local_H2_diagnosis_smoke_predictions/
   │     │  └─ train_002.npz
   │     │     └─ 单序列诊断冒烟的逐事件预测。
   │     ├─ local_A1_smoke/
   │     │  ├─ summary.json
   │     │  │  └─ A1原结构Refiner适配的一轮单序列冒烟摘要。
   │     │  ├─ epoch_001.pt
   │     │  │  └─ A1冒烟checkpoint，只用于保存/回读验证。
   │     │  └─ dev_predictions/train_002.npz
   │     │     └─ A1冒烟在train_002上的逐事件开发集预测。
   │     ├─ local_A2_smoke/
   │     │  ├─ summary.json
   │     │  │  └─ A2正样本全保留、困难/随机负样本采样的一轮冒烟摘要。
   │     │  ├─ epoch_001.pt
   │     │  │  └─ A2冒烟checkpoint。
   │     │  └─ dev_predictions/train_002.npz
   │     │     └─ A2冒烟在train_002上的逐事件开发集预测。
   │     ├─ local_A3_smoke/
   │     │  ├─ summary.json
   │     │  │  └─ A3加入四个历史特征的一轮冒烟摘要。
   │     │  ├─ epoch_001.pt
   │     │  │  └─ A3冒烟checkpoint。
   │     │  └─ dev_predictions/train_002.npz
   │     │     └─ A3冒烟在train_002上的逐事件开发集预测。
   │     ├─ local_A3_null_smoke/
   │     │  ├─ summary.json
   │     │  │  └─ A3-null同维同参无历史对照的一轮冒烟摘要。
   │     │  ├─ epoch_001.pt
   │     │  │  └─ A3-null冒烟checkpoint。
   │     │  └─ dev_predictions/train_002.npz
   │     │     └─ A3-null冒烟在train_002上的逐事件开发集预测。
   │     ├─ local_SR1_direct_smoke/
   │     │  ├─ summary.json
   │     │  │  └─ S-R1直接训练的一轮冒烟摘要；教师为工程冒烟教师，不能作精度结论。
   │     │  ├─ epoch_001.pt
   │     │  │  └─ S-R1直接训练冒烟checkpoint。
   │     │  ├─ student_float32_v1.npz
   │     │  │  └─ 由冒烟checkpoint导出的版本化浮点学生权重，用于导出/回读测试。
   │     │  └─ dev_predictions/train_002.npz
   │     │     └─ S-R1直接训练冒烟的逐事件开发集预测。
   │     └─ local_SR1_kd_smoke/
   │        ├─ summary.json
   │        │  └─ S-R1知识蒸馏的一轮冒烟摘要；只验证KD数据流和损失实现。
   │        ├─ epoch_001.pt
   │        │  └─ S-R1 KD冒烟checkpoint。
   │        ├─ student_float32_v1.npz
   │        │  └─ 由KD冒烟checkpoint导出的版本化浮点学生权重。
   │        └─ dev_predictions/train_002.npz
   │           └─ S-R1 KD冒烟的逐事件开发集预测。
   └─ software/
      ├─ scripts/
      │  ├─ audit_data.py
      │  │  └─ 审计EV-UAV文件、角色和哈希；仅在正式建索引阶段维护data/index.json。
      │  ├─ evaluate.py
      │  │  └─ 在受控数据角色上评测冻结H2基线。
      │  ├─ diagnose_h2.py
      │  │  └─ 步骤1入口：在Train-dev分析H2 Proposal门控与Refiner漏检来源。
      │  ├─ train_h2_refiner.py
      │  │  └─ 步骤2/3入口：训练A1、A2、A3和A3-null，并做公平阈值选择。
      │  ├─ train_h2_student.py
      │  │  └─ 步骤4入口：训练S-R1/S-R2直接或KD学生，统计复杂度并导出权重。
      │  ├─ train_causal.py
      │  │  └─ 步骤5入口：训练C0—C4因果ANN；默认仅执行有界冒烟，不访问Test。
      │  ├─ evaluate_causal.py
      │  │  └─ 评测因果checkpoint、冻结开发工作点并可输出显式无门控诊断。
      │  ├─ release_h2_family.py
      │  │  └─ 步骤7入口：冻结A*/S*三种子并执行受保护的一次Validation/Test。
      │  ├─ diagnose_s1.py
      │  │  └─ 在Train-dev统计S1分阶段耗时和漏检归因，服务旧S1适配分支。
      │  ├─ train_refiner_adaptation.py
      │  │  └─ 使用冻结S1候选训练B1/B2浮点Refiner的旧离线适配入口。
      │  ├─ train_s0p.py
      │  │  └─ 训练S0-P（SNN Proposal + H2 Refiner）；默认只做训练前检查。
      │  ├─ export_s0p.py
      │  │  └─ 将选定S0-P checkpoint导出为可审计NPZ权重。
      │  ├─ evaluate_s0p.py
      │  │  └─ 在固定角色上评测冻结S0-P Proposal与H2 Refiner组合。
      │  ├─ train_s0r.py
      │  │  └─ 训练S0-R（H2 Proposal + SNN Refiner）；默认只做训练前检查。
      │  ├─ select_s0r.py
      │  │  └─ 在Train-dev联合选择S0-R checkpoint和事件阈值。
      │  ├─ export_s0r.py
      │  │  └─ 将选定S0-R checkpoint导出为可审计NPZ权重。
      │  ├─ evaluate_s0r.py
      │  │  └─ 评测H2 Proposal与事件独立S0-R Refiner-SNN，不开放Test。
      │  ├─ evaluate_s0full.py
      │  │  └─ 评测冻结S0-P Proposal与S0-R Refiner的直接串联方案。
      │  ├─ analyze_stc.py
      │  │  └─ 在Train-dev诊断Proposal时空相关性，不更新权重。
      │  ├─ select_stc.py
      │  │  └─ 端到端比较STC候选并为HW-calibration导出输入；该分支已负面停止。
      │  ├─ evaluate_fixed.py
      │  │  └─ 在HW-calibration扫描H2/S1定点位宽并冻结通过门槛的S1配置。
      │  ├─ evaluate_pot.py
      │  │  └─ 在HW-calibration比较全PoT与混合PoT并冻结S2配置。
      │  ├─ evaluate_s1.py
      │  │  └─ 在Validation固定比较A1-H2与S1-P + A1 Refiner。
      │  ├─ evaluate_s2.py
      │  │  └─ 在Validation固定比较S1与S2-PoT。
      │  ├─ evaluate_approx.py
      │  │  └─ 在HW-calibration扫描S3的LOA位置和低位数候选。
      │  ├─ evaluate_s3.py
      │  │  └─ 对冻结S3配置执行一次Validation并与S1比较。
      │  ├─ profile_compute.py
      │  │  └─ 统一统计H2 ANN与S0-P SNN的计算活动代理；不等同于能耗实测。
      │  ├─ generate_rtl_vectors.py
      │  │  └─ 生成并校验exact/LOA加法器、LIF和第一层通道的RTL黄金向量。
      │  └─ freeze_phase1.py
      │     └─ 旧第一阶段冻结入口：冻结完整Train-dev选择后允许一次Validation。
      ├─ tests/
      │  ├─ evuav_eval_33e2aff.py
      │  │  └─ 固定版本的EV-UAV官方式定位评测参考实现，供指标一致性测试调用。
      │  ├─ test_core.py
      │  │  └─ 数据读取、事件指标和H2基线的最小回归测试。
      │  ├─ test_phase1_common.py
      │  │  └─ 数据角色、统一选点和记录协议测试。
      │  ├─ test_phase1_release.py
      │  │  └─ 旧阶段冻结/一次性Validation保护的合成测试。
      │  ├─ test_refiner_adaptation.py
      │  │  └─ 验证S1离线快速路径与冻结整数参考一致。
      │  ├─ test_snn.py
      │  │  └─ 覆盖LIF、SNN训练、定点/PoT/LOA、STC、RTL向量和数据边界。
      │  ├─ test_causal.py
      │  │  └─ 覆盖因果性、no-memory同参对照和时间训练不变量。
      │  ├─ test_h2_research.py
      │  │  └─ 覆盖步骤1—3的H2诊断、采样、历史特征和公平选点。
      │  ├─ test_h2_students.py
      │  │  └─ 覆盖步骤4学生结构、初始化、KD、复杂度和导出契约。
      │  └─ test_h2_release.py
      │     └─ 覆盖步骤4—7多种子门槛、冻结哈希和Validation/Test保护。
      ├─ wangdao/
      │  ├─ __init__.py
      │  │  └─ wangdao Python包标识。
      │  ├─ data.py
      │  │  └─ EV-UAV事件数组校验和NPZ加载基础函数。
      │  ├─ metrics.py
      │  │  └─ 通用事件级指标和EV-UAV定位指标实现。
      │  ├─ phase1_common.py
      │  │  └─ 第一阶段数据边界、官方兼容评分、阈值选择和结果记录公共层。
      │  ├─ phase1_metrics.py
      │  │  └─ 统一的全事件micro计数与固定EV-UAV定位规则评估器。
      │  ├─ h2_research.py
      │  │  └─ 步骤1—3公共实现：冻结H2、错误诊断、A1/A2采样与A3特征。
      │  ├─ h2_students.py
      │  │  └─ 步骤4公共实现：浅层学生、logit蒸馏、复杂度和版本化产物。
      │  ├─ h2_release.py
      │  │  └─ 步骤7公共实现：候选冻结、哈希校验和受保护评测。
      │  ├─ causal.py
      │  │  └─ 无标签泄漏的25 ms流式表征和C0—C4因果ANN结构。
      │  ├─ causal_training.py
      │  │  └─ 因果ANN训练、整序列回放评测、缓存和同步软件计时。
      │  ├─ refiner_adaptation.py
      │  │  └─ 旧S1离线B0/B1/B2诊断和浮点Refiner适配实现。
      │  ├─ s0p.py
      │  │  └─ S0-P的数据构造、采样和训练目标。
      │  ├─ s0r.py
      │  │  └─ S0-R的H2候选数据入口与事件流训练目标。
      │  ├─ snn.py
      │  │  └─ 可训练LIF单元、Proposal-SNN、Refiner-SNN及相关时间切片逻辑。
      │  ├─ stc.py
      │  │  └─ 训练专用时空相关性标签、热图与辅助损失工具。
      │  ├─ fixed.py
      │  │  └─ 面向软件/RTL对齐的H2、S1、PoT和LOA位精确整数黄金模型。
      │  └─ h2/
      │     ├─ __init__.py
      │     │  └─ H2子包公共导出。
      │     ├─ constants.py
      │     │  └─ H2固定传感器几何、窗口、阈值和网络结构常量。
      │     ├─ representation.py
      │     │  └─ H2窗口patch、Proposal路由掩码和Refiner上下文构造。
      │     ├─ features.py
      │     │  └─ H2 Refiner使用的无标签28维事件特征。
      │     ├─ losses.py
      │     │  └─ H2 Proposal与D1 Refiner训练损失。
      │     ├─ torch_model.py
      │     │  └─ 与导出格式匹配的无偏置PyTorch QAT网络。
      │     ├─ numpy_model.py
      │     │  └─ 冻结H2 NPZ的纯NumPy加载与端到端推理。
      │     └─ export.py
      │        └─ 将修改后的H2 PyTorch QAT模型导出为紧凑NPZ。
      └─ weights/
         ├─ H2_D1_seed42.npz
         │  └─ 当前冻结H2 ANN基线权重；A1/A2训练和全部H2比较的起点。
         ├─ S0P_T4_POS15_KD50_S42.npz
         │  └─ 冻结S0-P浮点Proposal-SNN权重。
         ├─ S0R_T4_POS10_KD50_S42.npz
         │  └─ 冻结S0-R浮点Refiner-SNN权重。
         ├─ S1P_INT8_ACC24_MEM24_Q12.npz
         │  └─ 冻结S1定点Proposal-SNN权重与24位/Q12配置。
         ├─ S2P_POT_MIXED_EMIN4_EMAX3_ACC24_MEM24_Q12.npz
         │  └─ 冻结S2混合PoT Proposal配置与权重。
         └─ S3P_LOA_SYN13_MEMALL_K2_ACC24_MEM24_Q12.npz
            └─ 冻结S3 LOA配置：syn1/syn3与全部膜更新使用k=2近似低位。
```

### 9.1 运行后会新增的目录

服务器按`README_SERVER.md`执行后，预计出现以下内容；它们目前不存在，不能写成已经完成：

```text
results/server_logs/                 # 环境、单元测试和各正式训练日志
results/runs/A1_seed42/              # A1正式seed42结果；A1入围后才有seed43/44
results/runs/A2_seed42/              # A2正式seed42结果；A2入围后才有seed43/44
results/runs/A3*_seed*/              # 仅在历史型剩余错误支持时运行
results/runs/SR*_*/                  # 仅在A*确定后运行的浅层学生结果
results/h2_family_release_v1.json    # A*/可选S*三种子通过门槛后的冻结清单
results/h2_family_validation_v1.json # 冻结后唯一一次Validation结果
results/h2_family_test_v1.json       # 匹配Validation完成后的唯一一次Test结果
```
