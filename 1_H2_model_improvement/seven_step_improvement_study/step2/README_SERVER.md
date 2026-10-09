# H2步骤二服务器包说明

步骤二正式运行已经完成。该包现在用于**复现或审计既有实验**，不再用于继续训练A2候选。正式结论是：A2-1、A2-2均无满足冻结Pd/Fa约束的工作点；A2-3因特征审查不通过而跳过；A2-4因STC适用性诊断不通过而跳过。七步研究现已全部完成，最终部署入口为`../../implementation/README.md`；不得使用Step2旧包启动后续步骤或替代A4-2。

## 既有正式运行信息

- 环境：Python 3.10.18、PyTorch 2.5.1+cu124、NumPy 2.2.6、NVIDIA GeForce RTX 4090。
- 数据：79条Train-main用于训练，10条Train-dev用于开发期评估。
- Validation/Test：均未使用。
- 正式目录：`runs/step2_formal_20261003`。
- A2-1、A2-2只运行seed42；没有模型通过，因此未运行seed43/44。

回传结果已经保存到本地`results/step2_formal_20261003/`并通过45个文件的SHA-256清单校验。现有结果不支持追加10轮：两组模型训练损失下降，但Train-dev AP从第5轮起持续下降。

## 复现步骤

解压并进入目录：

```bash
mkdir -p ~/wangdao_h2_step2 && cd ~/wangdao_h2_step2 && unzip H2_step2_server_20261003.zip && cd H2_step2_server_20261003
```

连接数据集：

```bash
ln -s /你的绝对路径/EV-UAV-dataset data/EV-UAV-dataset
```

检查环境、测试和数据：

```bash
conda activate evuav
```

```bash
python -m pip check
```

```bash
python -m unittest discover -s tests -v
```

```bash
python run_step2.py preflight --output runs/preflight.json --require-cuda
```

如确需从头复现实验，使用新输出目录，禁止覆盖既有正式结果：

```bash
python run_step2.py run-all --output runs/step2_reproduction --device cuda --run
```

每轮进度保存在模型目录的`training_progress.csv`；第5、10、15、20轮包含完整Train-dev评估。运行结束后收集结果：

```bash
python collect_step2_results.py runs/step2_reproduction --output runs/step2_reproduction.results.zip
```

## 不再执行的操作

- 不在A2-1或A2-2上追加10轮。
- 不绕过诊断门槛强行运行A2-3或A2-4。
- 不加载A2检查点开始步骤三。
- 不读取Validation或Test。

后续研究按照本地七步总计划进行。步骤3以后使用20,000 events/s序列密度边界；本包中的旧阈值和A2结果保持冻结。Step3没有候选晋级，Step4识别出A4-2历史特征候选，Step5已完成纠正选型和冻结。本A2复现包不继续扩展，也不用于Step6多seed复现。
