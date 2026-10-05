# Step5服务器运行说明（历史复现）

Step5正式回放已完成，七步研究也已结束。以下命令只用于复现冻结过程；最终部署入口为`../../implementation/README.md`，不得重新选择epoch或阈值。

Step5不进行训练，只修正A4-2选型并从选中checkpoint重新回放Train-dev。上传包已经包含A4-2/A4-2-null的Step4摘要、预注册checkpoint和密度参数，不需要重新上传Step4结果。数据集本身不在包内。

## 1. 解压与数据集链接

```bash
unzip H2_step5_server_20261004.zip
```

```bash
cd H2_step5_server_20261004
```

```bash
rm -f data/EV-UAV-dataset/PLACE_DATASET_HERE.txt
```

```bash
rmdir data/EV-UAV-dataset
```

```bash
ln -s /export01/frshu/experiment/SNN-model/EV-UAV-dataset data/EV-UAV-dataset
```

如果服务器上的真实数据集路径不同，只替换最后一条命令中的源路径。

## 2. 校验包和环境

```bash
sha256sum -c ../H2_step5_server_20261004.zip.sha256
```

```bash
python -m unittest discover -s tests -v
```

```bash
python run_step5.py preflight --step4-results input/step4_results --output runs/preflight.json --require-cuda
```

## 3. 可选冒烟

冒烟只验证选型，不导出正式冻结模型。

```bash
python run_step5.py run-all --step4-results input/step4_results --output runs/smoke --device cpu --smoke
```

预期选型为A4-2 epoch8、阈值0.40，但程序必须自行得出该结果。

## 4. 正式回放

Step5只需一张GPU。其他GPU留空不影响结果。

```bash
CUDA_VISIBLE_DEVICES=0 python run_step5.py run-all --step4-results input/step4_results --output runs/step5_formal --device cuda --run
```

执行完成后验证checkpoint、指标与导出一致性：

```bash
CUDA_VISIBLE_DEVICES=0 python run_step5.py verify --results runs/step5_formal --step4-results input/step4_results --output runs/step5_formal/verification/artifact_consistency.json --device cuda
```

## 5. 回传结果

```bash
python collect_step5_results.py runs/step5_formal --output step5_results.zip
```

将`step5_results.zip`下载并放入本地`seven_step_improvement_study/step5/results/`。回传包排除可再生缓存，保留决策、全部指标、冻结checkpoint、int8权重、图表、验证报告和结果清单。

出现中断时不要覆盖已有正式目录；确认中断原因后使用新的输出目录重新执行。Step5没有优化器训练状态，也不存在续训10轮入口。
