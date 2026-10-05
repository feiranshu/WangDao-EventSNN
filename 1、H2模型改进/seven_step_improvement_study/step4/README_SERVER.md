# H2步骤四服务器运行说明（历史复现）

Step4正式运行已经完成并回传。本包现在只用于复现A4-1至A4-5及其匹配对照；所有命令都是单行，可整条复制，且只读Train-main和Train-dev。七步研究现已完成，最终部署入口为`../../implementation/README.md`；不要用本包覆盖既有Step4结果或替代最终权重。

## 1. 核对、解压和进入目录

```bash
sha256sum H2_step4_server_20261004.zip
```

输出必须与`H2_step4_server_20261004.zip.sha256`一致。

```bash
unzip H2_step4_server_20261004.zip
```

```bash
cd H2_step4_server_20261004
```

## 2. 放置数据

包内已有`data/index.json`，不包含EV-UAV数据集。最终应能看到：

```text
H2_step4_server_20261004/data/EV-UAV-dataset/train/train_000.npz
```

若数据集位于`/export01/frshu/experiment/SNN-model/EV-UAV-dataset`，确认该路径无误后执行：

```bash
rm -f data/EV-UAV-dataset/PLACE_DATASET_HERE.txt && rmdir data/EV-UAV-dataset && ln -s /export01/frshu/experiment/SNN-model/EV-UAV-dataset data/EV-UAV-dataset
```

## 3. 环境、前置检查与单元测试

```bash
conda activate evuav
```

```bash
python run_step4.py preflight --output runs/preflight.json --require-cuda --require-gpus 4
```

```bash
python -m unittest discover -s tests -v
```

前置检查要求4张CUDA卡均可见。单元测试覆盖特征维数、零初始化、冻结参数组、优化器状态延续、int8形状、排序、续训边界以及空指标安全打印。

## 4. 可选smoke

smoke只在CPU串行跑少量Train-main数据和每模型一个batch，但仍评估完整Train-dev，可能需要数分钟。它只验证链路，不作为精度结论。

```bash
python run_step4.py run-all --output runs/step4_smoke --device cpu --smoke
```

smoke checkpoint不得用于正式续训。

## 5. 四卡正式运行

```bash
python run_step4.py run-all --output runs/step4_formal_20261004 --device cuda --gpus 0,1,2,3 --run
```

程序先构建一次共享只读候选缓存和Train-main密度归一化，再完成等价性审计。首批四模型分别占用GPU 0—3。第二批运行A4-3及其对照；A4-4、A4-5只在冻结准入条件满足时自动启动。

每个模型每轮都会在当前终端实时显示结果，同时写入：

```text
runs/step4_formal_20261004/logs/A4-1-seed42.log
runs/step4_formal_20261004/runs/A4-1-seed42/training_progress.csv
```

训练进度包含loss、两阶段学习率、梯度、耗时、新旧权重范数、饱和率、float/int8 AP、总体IoU、高密度Recall和资格等级。程序拒绝覆盖既有正式目录。

## 6. 中断后继续总流程

已完成且模式一致的候选会被复用；缺失候选会继续运行：

```bash
python run_step4.py run-all --output runs/step4_formal_20261004 --device cuda --gpus 0,1,2,3 --run --resume
```

如果某个已存在候选不是`complete`，程序会停止，防止把半成品混入正式结论。保留失败目录用于排错，修复后将失败目录改名归档，再用同一`--resume`命令继续；不要删除已完成模型。

## 7. 满足边界后成对续训10轮

只有对应候选和对照的`summary.json`都显示`continue10.passes=true`时，下面命令才会执行。A4-1示例：

```bash
python run_step4.py continue-pair --run-root runs/step4_formal_20261004 --model A4-1 --gpus 0,1
```

A4-2示例：

```bash
python run_step4.py continue-pair --run-root runs/step4_formal_20261004 --model A4-2 --gpus 0,1
```

A4-3、A4-4或A4-5需要续训时，只替换`--model`值。候选和匹配对照会分别占用两张卡并从各自`epoch_010.pt`继续到第20轮；只能续训一次，输出写入`continuations/`。

## 8. 收集回传结果

```bash
python collect_step4_results.py runs/step4_formal_20261004 --output runs/step4_formal_20261004.results.zip
```

请回传`runs/step4_formal_20261004.results.zip`。它包含决定、协议、逐轮历史、日志、指标、checkpoint、int8导出和已经执行的续训，不包含可再生候选缓存。

## 9. 正式目录结构

```text
runs/step4_formal_20261004/
├─ preflight.json
├─ preflight/step4_equivalence.json
├─ work/density_transform.json
├─ work/cache/                       # 不进入回传包
├─ logs/
├─ runs/
│  ├─ A4-1-seed42/
│  ├─ A4-1-null-seed42/
│  ├─ A4-2-seed42/
│  ├─ A4-2-null-seed42/
│  ├─ A4-3-seed42/
│  ├─ A4-3-control-seed42/
│  └─ 条件候选A4-4/A4-5及其对照
├─ continuations/                    # 仅满足续训边界并执行后出现
├─ metrics/
└─ step4_decision.json
```

所有正式报告必须同时保留`validation_used=false`和`test_used=false`。Step4结束后仍不要运行Validation/Test；它们只在Step7冻结后各使用一次。
