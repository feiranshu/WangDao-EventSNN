# S3 服务器运行说明 · GEN2-S3-001

本包只包含 Train-main 79 条与 Train-dev 10 条的冻结 C1 候选缓存、索引和训练代码；原始 EV-UAV 数据集不在包内。S3-B 的研究 ANN 已在本地按统一协议完成 10 epoch，其检查点、历史和来源记录随包提供；服务器只需训练跨窗口保留状态 SNN 的 10 epoch，seed42。S3-C 的窗口重置 SNN 是**条件对照**，须在 S3-B 结果回传并审查后另行运行。Validation、Test、HW-calibration 不在本包中，也不在程序允许的数据角色中。所有命令为单行，可整条复制。

## 1. 解压并链接服务器现有数据集

```bash
unzip GEN2_S3_server_20261012.zip
```

```bash
cd GEN2_S3_server_20261012
```

```bash
rm -f 1_H2_model_improvement/implementation/data/EV-UAV-dataset/PLACE_DATASET_HERE.txt
```

```bash
rmdir 1_H2_model_improvement/implementation/data/EV-UAV-dataset
```

```bash
ln -s /export01/frshu/experiment/SNN-model/EV-UAV-dataset 1_H2_model_improvement/implementation/data/EV-UAV-dataset
```

如果服务器数据集位置不同，只替换最后一条命令中的源路径。程序会逐份核对原始 Train 文件哈希与包内缓存来源；不要把 Validation/Test 目录复制到包内。

## 2. 压缩包、环境与前置检查

压缩包 SHA 在解压目录的上一级校验：

```bash
sha256sum -c ../GEN2_S3_server_20261012.zip.sha256
```

```bash
python -c 'import torch,numpy; print(torch.__version__,numpy.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")'
```

```bash
python -m unittest discover -s 3_ANN_to_SNN/s3 -p 'test_*.py' -v
```

```bash
python 3_ANN_to_SNN/s3/preflight.py --output 3_ANN_to_SNN/s3/runs/preflight.json --require-cuda
```

前置检查必须显示 `status=pass`、79/10 序列、2,219,864/438,966 候选及目标 CUDA 设备。它会核对冻结索引、权重、特征协议和每份缓存，并比较 CPU/CUDA 的预测、状态、损失、梯度和一步更新。若失败，先修环境或实现，不启动正式训练。`requirements.txt` 是版本基线；须使用服务器上带 CUDA 的 PyTorch 环境。

## 3. 可选 CPU 冒烟

冒烟只用 `train_000`/`train_002` 各一条及 1 epoch，输出标记为 `pilot_not_stage_result`：

```bash
python 3_ANN_to_SNN/s3/train.py --model ann --smoke --epochs 1 --device cpu --output 3_ANN_to_SNN/s3/runs/smoke_ann
```

```bash
python 3_ANN_to_SNN/s3/train.py --model snn --smoke --epochs 1 --device cpu --output 3_ANN_to_SNN/s3/runs/smoke_snn
```

## 4. 正式 S3-B 参考训练

包内 `3_ANN_to_SNN/s3/runs/ann/` 是完整本地 ANN 参考，请保留。以下任务使用全量 Train-main/Train-dev；不要把 S2 末轮检查点当成新目标函数的续训起点。每轮输出无权 CE、统一权重和归一化 CE、优化步数、裁剪次数、AP、完整阈值曲线和成本；每轮追加检查点，第 5/10 轮为预定决策节点。当前空间单元并行方法的实际 GPU 加速效果由运行时间判定，不把运算计数当成实测加速。

```bash
CUDA_VISIBLE_DEVICES=0 python 3_ANN_to_SNN/s3/train.py --model snn --mode persistent --epochs 10 --device cuda --output 3_ANN_to_SNN/s3/runs/persistent_snn
```

若进程或机器中断，使用同一命令追加 `--resume`；它从最后一轮完整检查点恢复，核对协议和代码哈希后重跑未完成轮次。不能因结果不理想删除运行目录重新选择种子、轮次或阈值。

## 5. 回传 S3-B 结果

```bash
python 3_ANN_to_SNN/s3/collect_results.py --task b --runs 3_ANN_to_SNN/s3/runs --output GEN2_S3_B_results_20261012.zip
```

请回传 `GEN2_S3_B_results_20261012.zip` 和自动生成的 `.zip.sha256`。回传包包含前置检查、协议锁、10 轮历史、全部检查点和 SHA 清单；排除候选缓存及原始数据。收到 B 的结果后才能决定 C 是否进入正式比较，以及 S3-D/E 的入口。

## 6. 条件 S3-C：仅在 B 审查后运行

以下命令**目前不是 S3-B 必做步骤**。只有参考训练通过数值与训练充分度复核后，才以独立编号运行窗口重置 SNN；其输入、种子、目标和 10 epoch 预算与持久状态 SNN 相同：

```bash
CUDA_VISIBLE_DEVICES=0 python 3_ANN_to_SNN/s3/train.py --model snn --mode window_reset --epochs 10 --device cuda --output 3_ANN_to_SNN/s3/runs/window_reset_snn
```

```bash
python 3_ANN_to_SNN/s3/collect_results.py --task c --runs 3_ANN_to_SNN/s3/runs --output GEN2_S3_C_results_20261012.zip
```

这项对照只移除额外神经元跨窗口状态；36 维输入自身仍包含历史。S3-D 单点动力学修订和 S3-E 机制入口复核必须依据回传结果另作决定，不在服务器上自动连跑。
