# H2-S1 七步计划：服务器操作README

本文件只包含需要服务器长时间运行或访问冻结后数据的操作。请按阶段门顺序执行，不要一次性把所有分支排队。默认项目目录为`/export01/frshu/experiment/SNN-model/H2_S1_steps1to7`，可替换为实际路径。

## 0. 上传、环境和数据

上传外层目录中的`H2_S1_steps1to7_20260923.zip`，然后执行：

```bash
PROJECT_DIR=/export01/frshu/experiment/SNN-model/H2_S1_steps1to7
if [ -e "$PROJECT_DIR" ]; then
  echo "目标目录已存在；请保留旧结果，并把 PROJECT_DIR 改成一个新的空目录。"
  exit 1
fi
mkdir -p "$PROJECT_DIR"
unzip H2_S1_steps1to7_20260923.zip -d "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 正式运行只使用服务器已有的 evuav；不要另外创建 .venv
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate evuav
test "$CONDA_DEFAULT_ENV" = "evuav"
python -m pip install -e . --no-deps --no-build-isolation
python -m pip check

mkdir -p data results/server_logs results/runs
ln -s /你的服务器数据路径/EV-UAV-dataset data/EV-UAV-dataset

python - <<'PY'
from pathlib import Path
p = Path('data/EV-UAV-dataset')
assert p.exists(), p
for name in ('train', 'val', 'test'):
    assert (p / name).is_dir(), p / name
print('dataset:', p.resolve())
PY

echo "conda_env=$CONDA_DEFAULT_ENV" > results/server_logs/environment.txt
which python >> results/server_logs/environment.txt
python --version >> results/server_logs/environment.txt 2>&1
python -c "import numpy,torch; print('numpy',numpy.__version__); print('torch',torch.__version__); print('cuda',torch.cuda.is_available()); print('gpu',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')" >> results/server_logs/environment.txt
nvidia-smi >> results/server_logs/environment.txt
python -B -m unittest discover -s software/tests -v 2>&1 | tee results/server_logs/unit_tests.log
```

预期单元测试为82项全部通过。每次重新登录服务器或开启新终端后，都要先执行上述 Conda 初始化与 `conda activate evuav`。`--no-deps --no-build-isolation`只把当前源码注册到已有环境，不更新服务器依赖；每次解压到新目录后都应重新执行一次。服务器不能联网时仍可按此方式安装当前项目，并须运行测试、保留环境记录。

## 1. 步骤1复现：可选但建议

本地已有完整10条Train-dev结果。服务器复跑用于确认数据、权重和运行环境一致：

```bash
python -B software/scripts/diagnose_h2.py \
  --run \
  --save-predictions \
  --output results/runs/server_H2_diagnosis_full.json \
  2>&1 | tee results/server_logs/step1_h2_diagnosis.log
```

核对`test_used=false`、`validation_used=false`、`sequence_count=10`，并检查：

- `TP+FP+FN+TN=events`；
- `proposal_false_negatives+refiner_false_negatives=fn`；
- `proposal_recall×refiner_conditional_recall=最终recall`；
- `ungated_all_event_diagnostic_at_0.28`不是零分占位。

## 2. 步骤2：先跑A1、A2的seed42

```bash
python -B software/scripts/train_h2_refiner.py \
  --variant A1 --run --device cuda --epochs 20 --seed 42 \
  --batch-size 4096 --learning-rate 1e-4 \
  --output-dir results/runs/A1_seed42 \
  2>&1 | tee results/server_logs/A1_seed42.log

python -B software/scripts/train_h2_refiner.py \
  --variant A2 --run --device cuda --epochs 20 --seed 42 \
  --batch-size 4096 --learning-rate 1e-4 \
  --output-dir results/runs/A2_seed42 \
  2>&1 | tee results/server_logs/A2_seed42.log
```

先比较两个`summary.json`：

```text
h2_frozen_reference
initial_float_refiner
h2_fair_working_point
candidate_fair_working_point
selected_iou_delta_vs_h2_r
history[*].positive_focal_mean / negative_focal_mean
history[*].selected_positive / selected_negative
final_evaluation.per_sequence
```

A2必须能证明困难/随机负样本截断实际发生；若选中负样本数始终等于全部候选，A2没有完成预期变量改变。

进入补种子的最低开发门槛：公平约束满足、seed42相对H2-R至少提高0.5个百分点。若A1、A2均未达到，先审查`train_089`、`train_092`及训练曲线；连续两次受控尝试无稳定收益则停止该分支。

假设A1入围，补种子命令为：

```bash
python -B software/scripts/train_h2_refiner.py \
  --variant A1 --run --device cuda --epochs 20 --seed 43 \
  --batch-size 4096 --learning-rate 1e-4 \
  --output-dir results/runs/A1_seed43 \
  2>&1 | tee results/server_logs/A1_seed43.log

python -B software/scripts/train_h2_refiner.py \
  --variant A1 --run --device cuda --epochs 20 --seed 44 \
  --batch-size 4096 --learning-rate 1e-4 \
  --output-dir results/runs/A1_seed44 \
  2>&1 | tee results/server_logs/A1_seed44.log
```

若A2入围，将命令中的A1全部改为A2。不要为已失败分支补种子。

## 3. 步骤3：仅在A1/A2仍留下历史型错误时运行

本地诊断对历史信息只有弱支持。只有A1/A2正式结果仍显示弱目标、窗边界或连续窗错误时，才运行A3和A3-null：

```bash
python -B software/scripts/train_h2_refiner.py \
  --variant A3 --run --device cuda --epochs 20 --seed 42 \
  --batch-size 4096 --learning-rate 1e-4 \
  --output-dir results/runs/A3_seed42 \
  2>&1 | tee results/server_logs/A3_seed42.log

python -B software/scripts/train_h2_refiner.py \
  --variant A3-null --run --device cuda --epochs 20 --seed 42 \
  --batch-size 4096 --learning-rate 1e-4 \
  --output-dir results/runs/A3_null_seed42 \
  2>&1 | tee results/server_logs/A3_null_seed42.log
```

A3必须同时满足：历史列梯度非零、A3优于A3-null、公平约束通过、增益不由单条序列主导。若A3成为A*候选，A3和A3-null都补seed43/44；保持完全相同预算。若A3与A3-null接近，结论是历史信息未获支持，不保留额外4维。

## 4. 确定A*后运行步骤4

先选定一种A*配方，并确保A*已有seed42/43/44。以下示例假设A*为A1，主教师摘要为`results/runs/A1_seed42/summary.json`。

### 4.1 S-R1直接训练与KD主对照

```bash
python -B software/scripts/train_h2_student.py \
  --student S-R1 --mode direct --feature-variant A1 \
  --teacher-summary results/runs/A1_seed42/summary.json \
  --run --device cuda --epochs 20 --seed 42 \
  --batch-size 4096 --learning-rate 1e-3 \
  --output-dir results/runs/SR1_direct_A1_seed42 \
  2>&1 | tee results/server_logs/SR1_direct_A1_seed42.log

python -B software/scripts/train_h2_student.py \
  --student S-R1 --mode kd --feature-variant A1 \
  --teacher-summary results/runs/A1_seed42/summary.json \
  --run --device cuda --epochs 20 --seed 42 \
  --batch-size 4096 --learning-rate 1e-3 --temperature 2 --kd-weight 0.2 \
  --output-dir results/runs/SR1_kd_A1_seed42 \
  2>&1 | tee results/server_logs/SR1_kd_A1_seed42.log
```

若A*是A2/A3/A3-null，同时替换`--feature-variant`、教师摘要和输出目录。脚本会继承A2采样，或使用A3/A3-null对应的36维输入。

比较`selected_iou_delta_vs_h2_r`与`selected_iou_delta_vs_teacher`。S*必须高于H2，并且相对A*损失不超过0.3个百分点。KD未优于直接训练时，保留直接学生。

只给入围的直接或KD配方补seed43/44；教师继续固定为A*的seed42，这样学生种子波动不会混入教师变化。命令只需修改`--seed`和输出目录。

### 4.2 S-R2条件分支

仅当S-R1明显超过H2但相对A*损失超过0.3个百分点时，才把`--student S-R1`改为`--student S-R2`，先跑direct seed42，再决定是否跑KD。S-R2只有16.13%的32维权重降幅，不满足20%目标时要如实报告。

每个入围学生目录会包含：

```text
summary.json
epoch_*.pt
student_float32_v1.npz
dev_predictions/
```

## 5. 步骤5在线分支：当前不要自动运行

只有A*精度结论稳定、剩余预算充足且确有25 ms在线发布需求时，才启动C0基线：

```bash
python -B software/scripts/train_causal.py \
  --variant C0 --run --device cuda --epochs 20 --seed 42 \
  --learning-rate 1e-3 --output-dir results/runs/C0_seed42 \
  2>&1 | tee results/server_logs/C0_seed42.log
```

C1—C4不得因为入口存在就全部运行。C2的严格无历史对照现在保留同一融合层和参数量：

```bash
python -B software/scripts/train_causal.py \
  --variant C2 --no-memory --run --device cuda --epochs 20 --seed 42 \
  --learning-rate 1e-3 --output-dir results/runs/C2_no_memory_seed42
```

需要估计解除门控上限时，使用独立诊断字段，不把它当部署输出：

```bash
python -B software/scripts/evaluate_causal.py \
  --checkpoint /对应checkpoint.pt \
  --baseline /完整C0_Train-dev评测.json \
  --role train_dev --device cuda --ungated-diagnostic \
  --output results/runs/Cx_ungated_diagnostic.json
```

## 6. 步骤7：冻结名单

只有A*三种子全部完成后才能冻结。以下示例仍假设A*=A1：

```bash
python -B software/scripts/release_h2_family.py freeze \
  --accuracy-summary \
    results/runs/A1_seed42/summary.json \
    results/runs/A1_seed43/summary.json \
    results/runs/A1_seed44/summary.json \
  --output results/h2_family_release_v1.json
```

如果S*也有三种子并通过门槛，在同一条命令加入：

```text
--student-summary \
  results/runs/SR1_direct_A1_seed42/summary.json \
  results/runs/SR1_direct_A1_seed43/summary.json \
  results/runs/SR1_direct_A1_seed44/summary.json
```

冻结入口自动检查：完整10条Train-dev、无smoke、数据索引、权重哈希、同配方42/43/44、A*开发门槛、S*精度容差。冻结后不要移动或修改模型文件和摘要。

## 7. 一次Validation，再一次Test

下面两段会读取受保护数据。只在冻结清单人工复核完成后执行。

### 7.1 Validation，仅一次

```bash
python -B software/scripts/release_h2_family.py evaluate \
  --manifest results/h2_family_release_v1.json \
  --role validation --device cuda --threads 1 --batch-size 4096 \
  --output results/h2_family_validation_v1.json \
  2>&1 | tee results/server_logs/h2_family_validation_v1.log
```

程序会先创建`results/h2_family_release_v1_validation_started.json`。即使结果不理想，也不得据此换模型、改阈值或重新训练。

### 7.2 Test，仅在Validation完成后一次

```bash
python -B software/scripts/release_h2_family.py evaluate \
  --manifest results/h2_family_release_v1.json \
  --role test \
  --validation-result results/h2_family_validation_v1.json \
  --device cuda --threads 1 --batch-size 4096 \
  --output results/h2_family_test_v1.json \
  2>&1 | tee results/server_logs/h2_family_test_v1.log
```

程序不会在Validation/Test重新扫描阈值。输出同时包含H2-F、H2-R、A*和可选S*，并以序列为重采样单位给出IoU差的95%区间。旧Test不是新的盲测；独立封存集仍标记为待完成。

## 8. 必须回传的文件

在步骤2/3完成后先回传以下内容，以便决定是否进入步骤4：

```text
results/server_logs/environment.txt
results/server_logs/unit_tests.log
results/server_logs/A*.log
results/runs/A1_seed*/summary.json
results/runs/A2_seed*/summary.json
results/runs/A3_seed*/summary.json             # 若运行
results/runs/A3_null_seed*/summary.json        # 若运行
入围A*的epoch_*.pt与dev_predictions/
```

步骤4后再回传：

```text
results/runs/SR*/summary.json
入围S*的epoch_*.pt
入围S*的student_float32_v1.npz
对应日志和dev_predictions/
```

最终冻结与评测后回传：

```text
results/h2_family_release_v1.json
results/h2_family_release_v1_validation_started.json
results/h2_family_validation_v1.json
results/h2_family_release_v1_test_started.json
results/h2_family_test_v1.json
对应_predictions/和server_logs/
```

候选缓存和原始数据不需回传。任何失败运行都保留原`summary.json`和日志，不删除后换同名目录重跑。
