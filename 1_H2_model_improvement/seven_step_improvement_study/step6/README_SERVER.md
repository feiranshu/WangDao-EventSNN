# Step6服务器运行说明（历史复现）

Step6正式运行及其后的Step7独立验收均已完成。以下命令只用于复现多seed与matched-null证据；最终部署入口为`../../implementation/README.md`。

本包只训练A4-2/A4-2-null的seed43和seed44，固定正式评估epoch 8、阈值0.40。所有命令均为单行，可整条复制。程序只接受Train-main和Train-dev，不提供Validation/Test入口。

## 1. 解压与数据链接

```bash
unzip H2_step6_server_20261005.zip
```

```bash
cd H2_step6_server_20261005
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

如果真实数据集路径不同，只替换最后一条命令中的源路径。

## 2. 校验与前置检查

```bash
sha256sum -c ../H2_step6_server_20261005.zip.sha256
```

```bash
python -m unittest discover -s tests -v
```

```bash
python run_step6.py preflight --step5-results input/step5_results --step4-results input/step4_results --output runs/preflight.json --require-cuda --require-gpus 4
```

前置检查必须显示四张可见GPU、79条Train-main、10条Train-dev、Step5冻结checkpoint哈希一致，且状态为`pass`。

## 3. 可选CPU冒烟

冒烟只跑每个任务一个小批次和epoch 1工程回放，不形成正式科学结论：

```bash
python run_step6.py run-all --step5-results input/step5_results --step4-results input/step4_results --output runs/smoke --device cpu --smoke
```

## 4. 四卡正式训练

### 已解压初版包的密度一致性热修复

初版包在正式训练前使用Python浮点数逐位相等比较Step4密度参数，可能因NumPy/libm最后几个浮点位不同而误报`Train-main density transform differs from frozen Step4 transform`。这不是训练失败，报错时四个任务均未启动。先确认当前文件仍是初版：

```bash
sha256sum run_step6.py
```

初版应显示`09b1e3567eaacd522620b2129bd605f92017e67bfb71bc138e58802815e17b19`。随后在当前解压目录执行以下单行热修复；它不会跳过检查，而是改为要求候选数、序列数完全一致，并要求全部Train-main候选的uint8密度编码完全一致：

```bash
python -c 'from pathlib import Path; p=Path("run_step6.py"); s=p.read_text(encoding="utf-8"); old="    if fit_density(train) != density and not args.smoke:\n        raise RuntimeError(\"Train-main density transform differs from frozen Step4 transform\")"; new="    recalculated_density = fit_density(train)\n    density_equivalent = recalculated_density.train_candidates == density.train_candidates and recalculated_density.train_sequences == density.train_sequences and all(np.array_equal(recalculated_density.apply(np.asarray(record[\"window_rate_eps\"], dtype=np.float64)), density.apply(np.asarray(record[\"window_rate_eps\"], dtype=np.float64))) for record in train)\n    if not density_equivalent and not args.smoke:\n        raise RuntimeError(\"Train-main density transform is not uint8-equivalent to frozen Step4 transform\")"; assert old in s, "hotfix target not found or already patched"; p.write_text(s.replace(old, new), encoding="utf-8"); print("Step6 density check hotfix applied")'
```

已经出现过该报错时，`runs/step6_formal_20261005`中只有通过的preflight和候选缓存，保留即可。修复后必须加`--resume`继续，不能重新使用不带`--resume`的原命令：

```bash
python run_step6.py run-all --step5-results input/step5_results --step4-results input/step4_results --output runs/step6_formal_20261005 --device cuda --gpus 0,1,2,3 --run --resume
```

本地修正版完整包会另外保留，初版已经解压并生成缓存时无需为这项修复重新上传完整压缩包。

### 全新目录运行

```bash
python run_step6.py run-all --step5-results input/step5_results --step4-results input/step4_results --output runs/step6_formal_20261005 --device cuda --gpus 0,1,2,3 --run
```

GPU0—3分别运行A4-2 seed43、null seed43、A4-2 seed44、null seed44。终端会带任务前缀实时显示每一轮结果；完整记录同时写入`logs/`和各任务的`training_progress.csv`。

正式训练后程序会自动固定读取epoch 8、阈值0.40，生成三seed决策、10,000次bootstrap、敏感性报告、完整流水线延迟和指标目录。不要用epoch 10或其他阈值手工替换结果。

## 5. 基础设施中断恢复

```bash
python run_step6.py run-all --step5-results input/step5_results --step4-results input/step4_results --output runs/step6_formal_20261005 --device cuda --gpus 0,1,2,3 --run --resume
```

已完成任务会直接复用；未完成任务会先原样迁入`failed_attempts/`，然后从冻结H2重新运行该单项。Step6禁止因为效果不理想追加10轮、增加seed或重新挑选epoch/阈值。

## 6. 收集回传结果

```bash
python collect_step6_results.py runs/step6_formal_20261005 --output step6_results_20261005.zip
```

请回传`step6_results_20261005.zip`。回传包保留协议、逐轮历史、日志、固定checkpoint/int8权重、全部指标、bootstrap摘要、敏感性、延迟、决策和SHA清单；排除Train-main/Train-dev候选缓存与`__pycache__`。
