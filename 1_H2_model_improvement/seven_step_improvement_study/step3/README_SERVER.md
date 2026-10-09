# H2步骤三服务器运行说明

> 状态说明：Step3正式运行已经完成并回传，且没有模型替代H2。以下命令只用于复现历史实验；七步研究现已完成，最终部署入口为`../../implementation/README.md`。原Step3上传包及其SHA保持不变。

本包执行zero-update、D3-0、A3-1和A3-2。所有命令均为单行，可直接复制。正式运行只允许Train-main和Train-dev；程序没有Validation/Test入口。

## 1. 解压与进入目录

先核对上传文件SHA-256：

```bash
sha256sum H2_step3_server_20261004.zip
```

输出必须与随包`.zip.sha256`文件一致。然后执行：

```bash
unzip H2_step3_server_20261004.zip
```

```bash
cd H2_step3_server_20261004
```

## 2. 数据目录

包内已经包含`data/index.json`，但不包含数据集。将服务器数据集复制或软链接为：

```text
H2_step3_server_20261004/data/EV-UAV-dataset/train/train_000.npz
```

如果数据位于`/export01/frshu/experiment/SNN-model/EV-UAV-dataset`，可执行：

```bash
rm -f data/EV-UAV-dataset/PLACE_DATASET_HERE.txt && rmdir data/EV-UAV-dataset && ln -s /export01/frshu/experiment/SNN-model/EV-UAV-dataset data/EV-UAV-dataset
```

仅在确认上述绝对路径确实是数据集目录后使用该命令。如果目录已经正确放置，不需要执行。

## 3. 环境、权限与单元测试

```bash
conda activate evuav
```

```bash
python run_step3.py preflight --output runs/preflight.json --require-cuda
```

```bash
python -m unittest discover -s tests -v
```

预期所有测试通过。运行时只检查包内`software/wangdao`和本地项目运行时；对无权限目录的探测会安全跳过，不会重现步骤二最初的`PermissionError`。

## 4. 可选smoke

smoke只验证执行链路，会限制Train-main序列、训练批次和轮数，但仍完整评估Train-dev，因此可能需要几分钟。结果不能作为精度结论。

```bash
python run_step3.py run-all --output runs/step3_smoke --device cuda --smoke
```

smoke成功后应显示`status=smoke_complete`；不要把smoke checkpoint用于正式续训。

## 5. 一次性正式运行

```bash
python run_step3.py run-all --output runs/step3_formal_20261004 --device cuda --run
```

该命令依次执行：zero-update → D3-0 → A3-1 seed42 → A3-2 seed42。每轮训练结果立即打印并追加到各模型的`training_progress.csv`，可据此观察是否收敛。程序拒绝覆盖已有输出目录。

如果zero-update失败，正式训练不会开始；请回传`preflight/zero_update/zero_update_control.json`和终端错误，不要修改容差绕过。

## 6. 在同一上传包中继续后续阶段

如果zero-update、D3-0或A3-1已经完成，但总流程在后续阶段中断，继续执行：

```bash
python run_step3.py run-all --output runs/step3_formal_20261004 --device cuda --run --resume
```

它只会复用模式、配置和完成状态均一致的阶段，不会重新训练已完成模型。smoke结果不能被正式运行复用；D3-0缺少完整`summary.json`或状态不正确时会明确停止，避免把残缺诊断当作正式结果。若现存候选目录状态不是`complete`，程序会停止并要求使用显式checkpoint恢复，避免混写结果。

## 7. 模型未收敛时续训10轮

A3-1从第20轮继续10轮：

```bash
python run_step3.py train --model A3-1 --output runs/step3_formal_20261004/continuations/A3-1-seed42-continue10 --cache runs/step3_formal_20261004/work/cache --device cuda --seed 42 --epochs 10 --learning-rate 0.00001 --resume-checkpoint runs/step3_formal_20261004/runs/A3-1-seed42/epoch_020.pt --epoch-offset 20 --run
```

A3-2从第20轮继续10轮：

```bash
python run_step3.py train --model A3-2 --output runs/step3_formal_20261004/continuations/A3-2-seed42-continue10 --cache runs/step3_formal_20261004/work/cache --device cuda --seed 42 --epochs 10 --learning-rate 0.00001 --resume-checkpoint runs/step3_formal_20261004/runs/A3-2-seed42/epoch_020.pt --epoch-offset 20 --run
```

只有第20轮Train-dev仍持续改善且没有过拟合时才续训。训练损失单独下降不是依据。续训目录必须是正式运行目录下的新`continuations`子目录，原20轮结果保持不变，后续结果收集会自动包含续训指标、checkpoint和int8导出。

## 8. 分阶段运行

需要逐项观察时，可按顺序运行：

```bash
python run_step3.py zero-update --output runs/zero_update --cache runs/cache --device cuda
```

```bash
python run_step3.py diagnose --output runs/D3-0 --cache runs/cache --device cuda --probe-epochs 30
```

```bash
python run_step3.py train --model A3-1 --output runs/A3-1-seed42 --cache runs/cache --device cuda --seed 42 --epochs 20 --learning-rate 0.00001 --run
```

```bash
python run_step3.py train --model A3-2 --output runs/A3-2-seed42 --cache runs/cache --device cuda --seed 42 --epochs 20 --learning-rate 0.00001 --run
```

分阶段运行不会自动验证前一步报告；正式研究优先使用`run-all`。

## 9. 收集并回传结果

```bash
python collect_step3_results.py runs/step3_formal_20261004 --output runs/step3_formal_20261004.results.zip
```

请回传`runs/step3_formal_20261004.results.zip`。回传包包含协议、诊断、逐轮历史、指标、选中checkpoint、int8导出以及已经运行的续训结果，不包含候选缓存。

## 10. 正式运行应看到的文件

```text
runs/step3_formal_20261004/
├─ preflight.json
├─ preflight/zero_update/zero_update_control.json
├─ diagnostics/D3-0/
├─ runs/A3-1-seed42/
├─ runs/A3-2-seed42/
├─ continuations/                 # 仅在执行继续10轮后出现
├─ metrics/
└─ step3_decision.json
```

任何结果都必须同时保留`validation_used=false`和`test_used=false`。步骤三完成后也不要运行Validation或Test；它们仍属于步骤七。
