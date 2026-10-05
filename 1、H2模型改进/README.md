# H2 ANN模型改进工程

本目录是可直接上传GitHub的最终工程交付。项目从原始两级MLP H2出发，经过七步受控研究，最终选择历史特征增强模型A4-2作为实际部署版本；完整实验过程保存在`seven_step_improvement_study/`，正式运行入口、权重和独立评测结果位于`implementation/`。

## 最终模型

A4-2保留原H2 Proposal，并把Refiner输入从32维扩展到36维：原32维特征后追加来自前两个100 ms窗口的4维因果历史信息。正式配置固定为seed42、epoch 8、全局阈值0.40和exact-int8回放。

最终决策以整体IoU、Precision和误报控制为主要部署目标：

| Test指标 | H2-R | A4-2 exact-int8 | 变化 |
| --- | ---: | ---: | ---: |
| AP | 96.7597% | 96.9309% | +0.1712个百分点 |
| 整体IoU | 85.9500% | 86.8931% | **+0.9431个百分点** |
| Precision | 90.5758% | 91.8692% | **+1.2933个百分点** |
| F1 | 92.4442% | 92.9869% | +0.5427个百分点 |
| 总体Fa | 1.1579e-5 | 1.0871e-5 | **−6.11%** |
| 高密度IoU | 57.3230% | 60.4175% | **+3.0945个百分点** |
| 高密度Fa | 4.4394e-5 | 3.8405e-5 | **−13.49%** |

Test整体IoU差值的10,000次配对序列bootstrap 95%区间为+0.0565至+2.2878个百分点，下界大于0。已知取舍是Test高密度Recall下降1.2138个百分点；因此本工程把A4-2定义为“整体IoU、Precision和误报控制优先”的最终模型，而不是所有指标无条件占优的模型。

## 快速使用

安装工程：

```bash
python -m pip install -e implementation
```

对一个包含`ev`结构化数组的EV-UAV NPZ序列运行最终模型：

```bash
python implementation/software/scripts/run_a4_2.py path/to/input.npz --output runs/a4_2_predictions.npz
```

数据集不随仓库提交。完整环境、验证和服务器复现方式见[implementation/README_SERVER.md](implementation/README_SERVER.md)。

## 目录结构

```text
1、H2_S1模型改进研究计划/
├─ README.md                         GitHub首页、最终结论、快速入口和总目录
├─ H2_ANN_模型改进研究计划.md       七步研究路线、实验边界和最终决策记录
├─ .gitignore                        排除数据集、缓存和本地临时文件
├─ implementation/                   最终A4-2交付：运行代码、权重、测试和正式指标
│  ├─ README.md                      最终实现说明及implementation完整目录
│  ├─ README_SERVER.md               安装、测试、推理和结果复核命令
│  ├─ PLAN.md                        Step1—7完整研究计划与结论
│  ├─ MODEL_CARD.md                  A4-2模型卡、适用范围和已知限制
│  ├─ DELIVERY_MANIFEST.json         最终实现全部文件的大小与SHA-256清单
│  ├─ pyproject.toml                 Python包元数据与依赖
│  ├─ data/                          冻结数据索引；真实数据集不入库
│  ├─ software/                      Python包、脚本、测试和冻结权重
│  └─ results/final_A4-2/            精简后的正式Validation/Test交付结果
├─ seven_step_improvement_study/      Step1—7完整研究历程和原始实验产物
│  ├─ README.md                      七步状态、每步作用及目录导航
│  └─ step1/ … step7/                各步骤代码、协议、服务器包和结果
└─ archive/
   └─ README.md                      已删除冗余旧包、smoke和缓存后的清理记录
```

`seven_step_improvement_study/`是研究可追溯记录，不是部署入口；`archive/`已经清空冗余产物，仅保留清理说明。正式使用以`implementation/`为准。

