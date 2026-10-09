# A4-2最终实现

本目录是项目的正式部署和复现入口。七步研究已经完成，最终采用A4-2 seed42、epoch 8、阈值0.40的exact-int8模型。原Step6/Step7保护门槛结论保持不变；最终接受部署是基于用户明确选择“整体IoU、Precision和误报控制优先”。

## 最终效果

在独立Test上，相对H2-R：AP提高0.1712个百分点、整体IoU提高0.9431个百分点、Precision提高1.2933个百分点、F1提高0.5427个百分点，总体Fa下降6.11%；高密度IoU提高3.0945个百分点、高密度Fa下降13.49%。整体IoU差值的bootstrap 95%区间为+0.0565至+2.2878个百分点。

已知取舍是高密度Recall下降1.2138个百分点。因此A4-2是精度和误报控制优先模型，不应描述为全部指标无条件优于H2。完整边界见[MODEL_CARD.md](MODEL_CARD.md)。

## 快速运行

```bash
python -m pip install -e .
```

```bash
python software/scripts/run_a4_2.py path/to/input.npz --output runs/a4_2_predictions.npz
```

输出NPZ包含每个事件的概率、布尔预测、Proposal候选下标、阈值和模型身份。脚本拒绝覆盖已有输出。

## 目录结构

```text
implementation/
├─ README.md                         最终模型说明、快速入口和目录导航
├─ README_SERVER.md                  环境安装、测试、推理与正式结果复核命令
├─ MODEL_CARD.md                     模型身份、选择理由、适用范围和已知限制
├─ PLAN.md                           Step1—7完整研究计划及最终结论
├─ AGENTS.md                         后续协作时必须遵守的数据与实验边界
├─ pyproject.toml                    `wangdao-h2-ann` 1.0.0包配置和依赖
├─ DELIVERY_MANIFEST.json            最终交付文件大小与SHA-256完整性清单
├─ data/
│  ├─ README.md                      数据目录、索引和GitHub边界
│  └─ index.json                     Train/Validation/Test冻结文件名与SHA-256
├─ software/
│  ├─ README.md                      运行代码总览及子目录职责
│  ├─ scripts/                       推理、评测、诊断、训练和发布命令
│  ├─ tests/                         单元测试与最终A4-2身份测试
│  ├─ wangdao/                       Python包：A4-2运行时、H2内核和研究工具
│  └─ weights/                       原H2 Proposal权重与最终A4-2权重
└─ results/
   ├─ README.md                      正式结果目录说明
   └─ final_A4-2/                    精简Validation/Test结果和部署决策
```

研究阶段代码与完整证据保存在`../seven_step_improvement_study/`；它们不是部署依赖。冗余旧包、冒烟权重和缓存已经删除，清理记录统一保存在根级本地目录`../../archive/1_H2_model_improvement/`。

