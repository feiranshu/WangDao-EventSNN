# Step7服务器运行说明（历史复现）

正式Step7已于2026-10-05完成。以下命令保留用于按原冻结配置复现或核验，不代表仍有待运行任务；不得依据复现结果重新选择模型、epoch或阈值。

本包执行A4-2的补充独立验收，不训练模型、不挑选epoch、不搜索阈值。候选永久固定为A4-2 seed42、epoch 8、阈值0.40、exact-int8；对照固定为H2-F阈值0.28与H2-R阈值0.30。Validation和Test都会各评测一次，Test结果只用于最终比较，绝不反馈到模型选择。

所有命令均为单行，可整条复制执行。

## 1. 解压与数据链接

```bash
unzip H2_step7_server_20261005.zip
```

```bash
cd H2_step7_server_20261005
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

## 2. 压缩包、测试与前置检查

```bash
sha256sum -c ../H2_step7_server_20261005.zip.sha256
```

```bash
python -m unittest discover -s tests -v
```

```bash
python run_step7.py preflight --step5-results input/step5_results --step6-results input/step6_results --step4-results input/step4_results --output runs/preflight.json --require-cuda
```

前置检查不会打开Validation或Test文件。它必须确认候选身份和checkpoint哈希、Step6原始失败记录、数据索引中的24条Validation和24条Test以及CUDA环境全部一致。

## 3. 可选无数据冒烟

下列冒烟只加载冻结checkpoint并用7条合成特征检查exact-int8链路，不会打开Validation或Test，因此不会消耗正式独立评测：

```bash
python run_step7.py smoke --step5-results input/step5_results --step6-results input/step6_results --step4-results input/step4_results --output runs/smoke --device cpu
```

## 4. 正式评测

```bash
python run_step7.py run-all --step5-results input/step5_results --step6-results input/step6_results --step4-results input/step4_results --output runs/step7_formal_20261005 --device cuda --batch-size 65536 --bootstrap-iterations 10000 --run
```

Step7没有训练过程，四张GPU不能像Step6那样对应四个独立训练任务。正式脚本使用一张GPU计算float参考路径，exact-int8、官方指标和bootstrap主要由CPU完成；让四个进程分别跑同一份Validation/Test反而会破坏“一次独立验收”的约束。

程序按固定顺序完成Validation和Test，并分别导出：

- H2-F、H2-R、A4-2 exact-int8、A4-2 float的完整指标；
- PR曲线、阈值曲线、固定工作点对比图和表；
- 整体、高密度、普通密度以及逐序列结果；
- AP、IoU、Precision、Recall、Pd、Fa的10,000次配对序列bootstrap；
- 固定安全门槛、研究目标和最终独立验收结论。

## 5. 基础设施中断恢复

只有进程、服务器或存储中断时才使用：

```bash
python run_step7.py run-all --step5-results input/step5_results --step6-results input/step6_results --step4-results input/step4_results --output runs/step7_formal_20261005 --device cuda --batch-size 65536 --bootstrap-iterations 10000 --run --resume
```

已经完整结束的角色会原样复用，不重新读取；中断的临时目录会迁入`failed_attempts/`后重跑。不要因为结果不理想删除正式目录或重新执行评测。

## 6. 完整性复核与结果回传

```bash
python run_step7.py verify-results --results runs/step7_formal_20261005
```

```bash
python collect_step7_results.py runs/step7_formal_20261005 --output step7_results_20261005.zip
```

请回传`step7_results_20261005.zip`及自动生成的`step7_results_20261005.zip.sha256`。结果包不包含候选缓存，但包含两套数据角色的全部指标、图表、逐序列比较、bootstrap、协议锁、决策和SHA清单。

## 7. 结果解释边界

Step6原协议的`step6_passes=false`永久保留。Step7是用户在看到Step6结论后明确授权、并在打开Validation/Test前冻结的新补充验收协议；Step7可以证明A4-2在独立数据上的效果，但不会事后把Step6历史结论改写为通过。

