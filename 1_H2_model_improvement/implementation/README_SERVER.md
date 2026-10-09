# A4-2安装、测试与运行说明

以下命令均为单行，可直接复制。正式模型固定为A4-2 seed42、epoch8、阈值0.40；不得用Test结果重新选epoch或阈值。

## 1. 安装

在`implementation`目录执行：

```bash
python -m pip install -e .
```

推荐复现环境为Python 3.11.4、NumPy 1.24.3、PyTorch 2.0.1。正式exact-int8推理主体在CPU NumPy路径执行，PyTorch用于保留的研究与checkpoint工具。

## 2. 测试

最终部署验收先运行：

```bash
python -m unittest software/tests/test_a4_2_runtime.py -v
```

它严格检查36→64→32→4结构、seed42、epoch8、阈值0.40、int8权重SHA-256、元数据和部署决策的一致性。

如需回归随工程保留的全部历史研究模块，请在上面给出的复现环境中运行：

```bash
python -m unittest discover -s software/tests -v
```

历史全量回归包含非部署路径的因果模型、学生模型和旧发布工具；它不替代最终A4-2专项验收。

## 3. 单序列推理

输入必须是包含`ev`结构化数组的EV-UAV NPZ文件：

```bash
python software/scripts/run_a4_2.py path/to/input.npz --output runs/a4_2_predictions.npz
```

自定义批大小：

```bash
python software/scripts/run_a4_2.py path/to/input.npz --output runs/a4_2_predictions.npz --batch-size 8192
```

输出目录可自定，但程序不会覆盖已有文件。

## 4. 数据集

真实EV-UAV数据集不上传GitHub。如需复现完整评测，请将数据集放置或软链接到：

```text
implementation/data/EV-UAV-dataset/
```

数据文件必须与`data/index.json`中的SHA-256一致。不得修改独立Test后再宣称原协议结果。

## 5. 正式结果复核

核心决策文件：

```text
results/final_A4-2/deployment_decision.json
results/final_A4-2/step7_decision.json
results/final_A4-2/validation/split_summary.json
results/final_A4-2/test/split_summary.json
```

其中`step7_decision.json`保存原门槛未全部通过的事实，`deployment_decision.json`保存基于IoU、Precision和Fa优先级作出的最终工程选择。

## 6. 交付清单

代码、权重、结果和文档变更后，在`implementation`目录重新生成SHA-256清单：

```bash
python software/scripts/build_delivery_manifest.py
```

`DELIVERY_MANIFEST.json`不包含真实数据集、Python缓存或清单自身；上传GitHub前可用其中的大小和SHA-256核对交付完整性。

