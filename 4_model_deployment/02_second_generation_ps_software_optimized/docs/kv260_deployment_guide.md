# KV260 部署、排障与验收指南

来源：`# 第一代 SNN 模型 KV260 部署：问题复盘、已验证修复与标准流程`。目录重构时按主题拆分，技术口径未改变。本文沿用第一代已实测 PL 基线；第二代仅包含 PS/软件优化，未重新生成 bitstream。

## 1. 当前已经验证到什么程度

截至本次实验结束，以下链路均已真实在 KV260 上通过，不是仿真结果：

1. 已验证 `wangdao_pl_flat` 可以在 KV260 上正常加载，FPGA manager 状态为 `operating`。
2. 两个固定 smoke patch 全部通过：
   - Patch 0：128 个 `0x00000000`
   - Patch 1：128 个 `0xFFFFFFFF`
   - 两组 16 维输出均与固定 golden 完全一致。
3. 单条真实 HW-calibration 序列 `train_022.npz`：
   - 80 windows
   - 336 patches/window
   - 26880 patches
   - 430080 个 int32 Q12 logits
   - 板端 capture 与 PC S1 golden：
     - `mismatch_words = 0`
     - `max_abs_logit_error = 0`
     - `BIT_EXACT_PASS = True`
4. `train_022` 完整模型：
   - Proposal-SNN @ PL
   - 候选生成 + 32D feature + H2 Refiner @ PS
   - `WANGDAO_FULL_PS_PL_PASS sequences=1`
   - `WANGDAO_FULL_VERIFY_PASS sequences=1`
5. 完整 10 条 Train-dev：
   - 10 条全部完成 PL Proposal-SNN
   - 10 条全部完成 PS H2 Refiner
   - 10 条全部 PL + PS bit-exact
   - 最终得到：
     - `WANGDAO_FULL_PS_PL_PASS sequences=10`
     - `WANGDAO_FULL_VERIFY_PASS sequences=10`
6. 10 条 Train-dev 的板端结果已经从 KV260 拉回 Windows：
   - `captures/*.capture.bin`：10 个
   - `predictions/*.pred.bin`：10 个

因此，当前部署结论是：

> **第一代冻结 SNN 的 KV260 完整部署链路已经跑通。PL Proposal-SNN 和 PS H2 Refiner 均已在完整 Train-dev 上与 PC frozen golden 精确一致。**

目前尚未完成的下一步是：

> 使用已经回传到 PC 的 10 条 `predictions/*.pred.bin`，走项目统一评测口径计算 Train-dev 的 Event IoU / Precision / Recall / F1 等模型指标。

---
## 4. 必须保留的已验证 FPGA 文件

这是本次最关键的部署经验之一。

### 4.1 正确 bitstream

文件：

```text
wangdao_pl_flat.bit.bin
```

Size：

```text
7797692 bytes
```

SHA256：

```text
3bf7473f26f5ddcd1157c2513706d179982eb131e8cb38880601d30e105dee23
```

alias：

```text
wangdao.bit.bin
```

必须与上述 bitstream 内容相同。

---

### 4.2 正确 shell.json

Size：

```text
50 bytes
```

SHA256：

```text
0377daaf2050b88df14de9a79d120144cf49381d00fa552402395a322cb7b1a3
```

内容：

```json
{
  "shell_type": "PL_FLAT",
  "num_slots": "1"
}
```

---

### 4.3 正确 DTBO —— 必须永久替换本地旧版本

#### 错误旧版 DTBO

Size：

```text
2024 bytes
```

SHA256：

```text
4d058b969fd960013b697e90f132b7635bb0af32eae8aef6e0c6ffe83845d32f
```

此 DTBO 是本次最开始 `xmutil loadapp wangdao_pl_flat` 失败的重要原因之一。

典型表现：

```text
Load Error: -1: Unable to load wangdao_pl_flat
```

历史底层 overlay 错误还出现过：

```text
create_overlay: Failed to create overlay (err=-12)
```

**本地部署目录中如果仍然存在这个 hash 的 DTBO，必须替换。**

#### 已验证可运行 DTBO

Size：

```text
1986 bytes
```

SHA256：

```text
0a1c9eb846b6929129b315b5e3ff510a06e332a74744dc048a9bc46c47ca2706
```

这是本次最终成功加载并完成所有 Train-dev 验证的 DTBO。

后续 Agent 应确保：

```text
kv260_deployment/firmware/artifacts/wangdao_pl_flat.dtbo
```

或者项目中最终用于部署的 DTBO，必须是这个版本。

---
## 5. 本次遇到的所有主要问题与解决方案

---

### 问题 1：上电后 `xmutil loadapp wangdao_pl_flat` 失败

表现：

```text
Load Error: -1: Unable to load wangdao_pl_flat
```

最开始发现当前项目里的 DTBO hash 是：

```text
4d058b...
```

而历史成功版本是：

```text
0a1c9e...
```

#### 解决

使用完整 known-good board package：

```text
wangdao_pl_flat.bit.bin
wangdao_pl_flat.dtbo
shell.json
wangdao_axi_smoke.sh
safe_board_run.sh
patch0_words.txt
patch0_expected.txt
patch1_words.txt
patch1_expected.txt
```

并验证 hash 后，由 `safe_board_run.sh` 完整安装、切换、验证和恢复。

结果：

```text
wangdao_pl_flat: Loaded with slot_handle 0
FPGA STATE = operating
WANGDAO_PASS
WANGDAO_PASS
== DEPLOY_OK ==
```

#### 应永久修改

不要让部署脚本继续使用旧 2024-byte DTBO。

---

### 问题 2：手工 `install → unloadapp → loadapp` 偶尔失败，而 safe runner 能成功

本次曾出现：

```text
sudo xmutil unloadapp
sudo xmutil loadapp wangdao_pl_flat

Load Error: -1
```

但同一套文件由 `safe_board_run.sh` 执行时能成功。

期间还出现过终端粘贴被打乱，例如命令文本与 shell prompt 拼接。

#### 结论

不要把一串敏感的 FPGA 切换命令依赖人工粘贴。

#### 解决

将“安装 package → listapps → unload → load → 检查 operating → 退出时恢复默认 PL”固化进 shell script。

推荐保留：

```text
safe_board_run.sh
```

并新增专用于长批量实验的：

```text
batch_hold_run.sh
```

而不是每次用 `sed` 临时修改。

---

### 问题 3：`k26-starter-kits` 提示 `No empty slot`

表现：

```text
Load Error: -3: No empty slot for k26-starter-kits
```

随后：

```bash
sudo xmutil listapps
```

发现：

```text
k26-starter-kits ... 0->0
```

#### 原因

默认应用本来已经处于 active。

#### 处理

这不是故障。

以后先：

```bash
sudo xmutil listapps
```

只有当三个应用都 `-1` 时才需要：

```bash
sudo xmutil loadapp k26-starter-kits
```

---

### 问题 4：执行 DTBO 路径导致 `Permission denied`

曾误输入：

```bash
/path/to/wangdao_pl_flat.dtbo
```

shell 会尝试“执行二进制 DTBO”，因此：

```text
Permission denied
```

#### 处理

无害。

DTBO 应：

```bash
cp ...
sha256sum ...
install ...
```

不能直接执行。

---

### 问题 5：PC 的 `(base)` 环境没有 torch

表现：

```text
ModuleNotFoundError: No module named 'torch'
```

#### 原因

最开始使用了 Anaconda `base`。

#### 解决

切换到已有：

```text
evuav
```

环境后，torch import 正常。

当前实际环境：

```text
Python 3.8.20
```

#### 后续建议

项目最终建议统一到 Python 3.10。

原因：项目脚本已经出现了 Python 3.9+ API：

```python
Path.is_relative_to()
```

当前 3.8 环境需要额外兼容 patch。

---

### 问题 6：数据集软链接在 E 盘，`relative_to(ROOT)` 崩溃

数据目录：

```text
dataset/EV-UAV-dataset
```

在 Windows 项目内显示存在，但实际指向：

```text
E:\experiment\...\EV-UAV-dataset
```

脚本：

```python
"path": path.relative_to(ROOT).as_posix(),
```

报错：

```text
ValueError:
'E:\...\train_022.npz'
does not start with
'D:\WangDao_EventSNN\...\01_first_generation_baseline'
```

#### 本次临时修复

在：

```text
host_tools/deployment_preparation/prepare_kv260_batch.py
```

将：

```python
"path": path.relative_to(ROOT).as_posix(),
```

改成：

```python
"path": path.as_posix(),
```

#### 推荐永久修复

不要假定数据集一定在 repo 内。

建议统一加 helper：

```python
def portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
```

然后所有 manifest path 都用该 helper。

---

### 问题 7：PowerShell 显示数据路径中文乱码

`manifest.json` 中曾显示：

```text
E:/experiment/FPGA涓婄殑SNN瀹炵幇/...
```

#### 原因

Windows PowerShell `Get-Content` 的终端编码显示问题。

#### 结论

JSON 实际可被 Python 正常读取，不影响部署和推理。

不应因为终端显示乱码而重新生成数据。

---

### 问题 8：PowerShell 不允许执行 `run_batch.ps1`

表现：

```text
未对文件 run_batch.ps1 进行数字签名
PSSecurityException
UnauthorizedAccess
```

#### 解决

只对当前 PowerShell 进程临时放开：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
```

确认：

```powershell
Get-ExecutionPolicy -Scope Process
```

输出：

```text
Bypass
```

也可以：

```powershell
Unblock-File .\kv260_deployment\pc_automation\run_batch.ps1
```

#### 应写入 README

这是 Windows 部署前置步骤，不是模型错误。

---

### 问题 9：`run_batch.ps1` 错误解析相对路径

用户当前目录明明是：

```text
D:\WangDao_EventSNN\...\01_first_generation_baseline
```

传入：

```powershell
-Manifest .\experiments\prepared_packages\hw_smoke\manifest.json
```

但脚本却寻找：

```text
C:\Users\sfr08\experiments\prepared_packages\hw_smoke\manifest.json
```

#### 原因

脚本使用：

```powershell
[System.IO.Path]::GetFullPath($Manifest)
```

在 Windows PowerShell 5.1 中，这里可能基于 process working directory，而不是当前 PowerShell location。

#### 本次解决

在调用前生成绝对路径：

```powershell
$manifest = Join-Path $PWD.Path 'experiments\prepared_packages\hw_smoke\manifest.json'
$capture  = Join-Path $PWD.Path 'experiments\prepared_packages\hw_smoke\captures'
```

然后：

```powershell
-Manifest $manifest
-CaptureDirectory $capture
```

#### 推荐永久修复

`run_batch.ps1` 内部应使用 PowerShell-aware 路径解析，例如：

```powershell
$manifestPath = (Resolve-Path -LiteralPath $Manifest).Path
```

对输出目录则基于：

```powershell
(Get-Location).Path
```

构造绝对路径。

---

### 问题 10：`run_batch.ps1` 推理成功后仍报 `Remote command failed`

这是本次最重要的脚本 bug 之一。

板端已经输出：

```text
WANGDAO_BATCH_PASS patches=26880 logits=430080
```

但 PowerShell 随后报错：

```text
Remote command failed (1)
```

错误命令本质是：

```bash
test $(stat -c %s train_022.inputs.bin) -eq 1720320
```

然而：

```text
inputs.bin  = 13762560 bytes
capture.bin = 1720320 bytes
```

#### 根因

`run_batch.ps1` 第 146 行错误检查了 `{3}`（input file），实际上应该检查 `{4}`（capture file）。

错误版本：

```powershell
test `$(stat -c %s {3}) -eq {5}
```

正确版本：

```powershell
test `$(stat -c %s {4}) -eq {5}
```

#### 必须永久修改

文件：

```text
kv260_deployment/pc_automation/run_batch.ps1
```

必须保留 `{4}` 修复。

---

### 问题 11：SSH / sudo 要反复输入密码

批量脚本运行期间可能出现：

```text
amd-edf@192.168.100.2's password:
Password:
Connection ... closed.
amd-edf@192.168.100.2's password:
...
```

#### 原因

脚本内部不是一个长期 SSH session，而是多次分别调用：

```text
ssh
scp
ssh
scp
...
```

而且：

```text
-InteractiveSudo
```

又会触发 sudo 密码。

#### 结论

功能上正常，不代表掉线。

#### 可选优化

后续可考虑：

- SSH public key
- Windows OpenSSH connection reuse
- 板端一次性 runner，减少 ssh/scp 次数

但不是当前正确性阻塞项。

---

### 问题 12：safe runner 会在结束后自动恢复默认 PL，不适合外部批量脚本

`safe_board_run.sh` 设计正确：

```text
LOAD custom PL
→ smoke
→ EXIT trap
→ RESTORE k26-starter-kits
```

但 Windows `run_batch.ps1` 需要在一段时间内保持：

```text
wangdao_pl_flat 0->0
```

#### 本次解决

从 `safe_board_run.sh` 派生：

```text
batch_hold_run.sh
```

在确认：

```text
FPGA STATE = operating
```

后插入：

```sh
echo "WANGDAO_HOLD_READY"
read HOLD_INPUT
```

这样：

```text
第一个 SSH：
batch_hold_run.sh
        ↓
WANGDAO_HOLD_READY
        ↓
保持 SNN PL active

第二个 SSH / Windows：
执行真实 batch

完成后：
回第一个 SSH 按 Enter
        ↓
继续 smoke
        ↓
自动恢复 k26-starter-kits
```

#### 必须永久改进

建议正式加入：

```text
kv260_deployment/board_runtime/batch_hold_run.sh
```

不要每次用 `sed` 现场生成。

---

### 问题 13：Python 3.8 不支持 `Path.is_relative_to()`

当前：

```text
Python 3.8.20
```

而：

```python
Path.is_relative_to()
```

从 Python 3.9 才存在。

受影响脚本至少包括：

```text
host_tools/deployment_preparation/prepare_ps_h2_batch.py
host_tools/result_evaluation/score_kv260_batch.py
```

#### 本次临时修复

因为相关 manifest / H2 weight 本身就在 repo 内，将：

```python
str(path.relative_to(ROOT).as_posix())
if path.is_relative_to(ROOT)
else str(path)
```

改成：

```python
path.relative_to(ROOT).as_posix()
```

#### 推荐永久方案

二选一：

**方案 A（推荐）：**

项目明确要求：

```text
Python >= 3.10
```

并提供独立部署环境。

**方案 B：**

加入 Python 3.8 compatible helper：

```python
def relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
```

不要直接使用：

```python
Path.is_relative_to()
```

---

### 问题 14：KV260 `dmesg` 出现 device-tree overlay memory leak warning

日志出现过：

```text
OF: overlay: WARNING: memory leak will occur if overlay removed
```

以及：

```text
OF: ERROR: memory leak, expected refcount 1 instead of 2
```

涉及：

```text
/axi/proposal_axi_lite_ref@a0000000
/axi/clocking0
/axi/clocking1
```

#### 实际结论

在这些 warning 存在的情况下，本次仍然成功完成：

```text
load wangdao_pl_flat
固定 smoke
单真实序列
10 条 Train-dev
restore starter kit
```

所以目前不能把这些 warning 当成本次计算错误。

#### 但应该注意

不要无必要地反复：

```text
loadapp → unloadapp → loadapp → unloadapp
```

建议每次实验：

```text
1 次加载
→ 完成整批任务
→ 1 次恢复
```

---

### 问题 15：tar 解压提示文件时间在未来

KV260 解压 Windows 打包文件时出现：

```text
time stamp ... is ... s in the future
```

#### 原因

KV260 当前系统时间曾显示为：

```text
Jan 8 2025
```

而 Windows 文件时间为 2026。

这是板上系统时钟未同步，不是数据损坏。

#### 结论

不影响 bitstream、binary、模型结果和 `cmp`。

后续可单独解决 NTP / RTC / 手工校时，但不是部署正确性的阻塞项。

---

### 问题 16：电脑 / 板卡断电重启后担心结果丢失

在 10 条 Train-dev 全部跑完以后，PC 和 KV260 都重启过。

重新登录后检查：

```bash
ls captures/*.capture.bin | wc -l
ls predictions/*.pred.bin | wc -l
```

结果：

```text
10
10
```

说明结果保存在 Linux 文件系统中，不会因为断电自动丢失。

#### 处理方式

不重新跑 268800 patches。

直接：

```bash
tar -czf ~/SNN_1.0/train_dev10_board_results.tar.gz \
    captures \
    predictions \
    ps_manifest.json
```

然后：

```powershell
scp ...
```

回 PC。

---
## 6. 固定 smoke 的 known-good 验收标准

成功必须看到：

```text
wangdao_pl_flat: Loaded with slot_handle 0
```

然后：

```text
FPGA STATE
operating
```

Patch 0：

```text
WANGDAO_PASS inputs=128 logits=16 status=0x00000002
```

Patch 1：

```text
WANGDAO_PASS inputs=128 logits=16 status=0x00000002
```

最终：

```text
== DEPLOY_OK ==
```

然后自动恢复：

```text
k26-starter-kits ... 0->0
```

---
## 7. 单条真实序列已验证结果

序列：

```text
train_022.npz
```

数据：

```text
80 windows
336 patches/window
26880 patches
430080 logits
```

输入：

```text
13762560 bytes
```

golden：

```text
1720320 bytes
```

最终：

```text
actual_words = 430080
golden_words = 430080
mismatch_words = 0
max_abs_logit_error = 0
BIT_EXACT_PASS = True
```

完整 PL+PS：

```text
events=21165
candidates=4175
```

验收：

```text
WANGDAO_FULL_PS_PL_PASS sequences=1
WANGDAO_FULL_VERIFY_PASS sequences=1
```

---
## 8. 完整 Train-dev 10 条序列

Train-dev 全集就是以下 10 条：

```text
train_002.npz
train_020.npz
train_034.npz
train_039.npz
train_041.npz
train_050.npz
train_060.npz
train_066.npz
train_089.npz
train_092.npz
```

准备阶段：

```text
10 sequences
× 80 windows
× 336 patches
= 268800 patches
```

PS 数据总事件：

```text
1237183 events
```

各序列：

| Sequence | Events | Candidates |
|---|---:|---:|
| train_002 | 44571 | 11104 |
| train_020 | 23042 | 4941 |
| train_034 | 20849 | 10815 |
| train_039 | 20340 | 8682 |
| train_041 | 44937 | 10696 |
| train_050 | 18300 | 5191 |
| train_060 | 43606 | 17234 |
| train_066 | 20297 | 7445 |
| train_089 | 405722 | 184993 |
| train_092 | 595519 | 218029 |

总 candidates：

```text
479130
```

注意：

```text
train_089
train_092
```

事件数和候选数远高于其余 8 条，因此 PS H2 阶段明显更重，这是正常的。

最终板端：

```text
WANGDAO_FULL_PS_PL_PASS sequences=10
```

最终验证：

```text
[1] PASS train_002 (PL + PS H2)
[2] PASS train_020 (PL + PS H2)
[3] PASS train_034 (PL + PS H2)
[4] PASS train_039 (PL + PS H2)
[5] PASS train_041 (PL + PS H2)
[6] PASS train_050 (PL + PS H2)
[7] PASS train_060 (PL + PS H2)
[8] PASS train_066 (PL + PS H2)
[9] PASS train_089 (PL + PS H2)
[10] PASS train_092 (PL + PS H2)

WANGDAO_FULL_VERIFY_PASS sequences=10
```

---
## 9. 推荐固化后的标准部署流程

---

### Stage A：开机后确认默认 PL

KV260：

```bash
sudo xmutil listapps
```

推荐初始状态：

```text
wangdao_pl_flat     -1
k26-starter-kits    0->0
k24-starter-kits    -1
```

---

### Stage B：先做 known-good smoke

进入 known-good package：

```bash
cd ~/SNN_1.0/known_good_smoke
sudo -v
./safe_board_run.sh
```

验收：

```text
WANGDAO_PASS
WANGDAO_PASS
== DEPLOY_OK ==
```

如果 smoke 不通过：

> 不要继续跑真实数据。

---

### Stage C：PC 生成 Train-dev PL batch

Windows：

```powershell
python .\host_tools\deployment_preparation\prepare_kv260_batch.py `
  --role train_dev `
  --output-dir $BatchDev10
```

必须：

```text
sequences = 10
patches = 268800
```

---

### Stage D：PC 生成 PS H2 batch

```powershell
python .\host_tools\deployment_preparation\prepare_ps_h2_batch.py `
  --manifest "$BatchDev10\manifest.json" `
  --output-dir $FullDev10
```

必须：

```text
sequences = 10
```

然后确保：

```text
sequences/*.inputs.bin  = 10
sequences/*.golden.bin  = 10
events/*.events.bin     = 10
h2_golden/*.pred.bin    = 10
```

---

### Stage E：上传完整包

Windows：

```powershell
tar.exe -czf $ArchiveDev10 -C $FullDev10 .
scp $ArchiveDev10 amd-edf@192.168.100.2:~/SNN_1.0/train_dev10_full.tar.gz
```

KV260：

```bash
rm -rf ~/SNN_1.0/train_dev10_full
mkdir -p ~/SNN_1.0/train_dev10_full

tar -xzf ~/SNN_1.0/train_dev10_full.tar.gz \
  -C ~/SNN_1.0/train_dev10_full
```

---

### Stage F：加载 PL 并 HOLD

第一个 SSH：

```bash
cd ~/SNN_1.0/known_good_smoke
sudo -v
./batch_hold_run.sh
```

停在：

```text
WANGDAO_HOLD_READY
```

不要按 Enter。

---

### Stage G：第二个 SSH 跑完整模型

```bash
cd ~/SNN_1.0/train_dev10_full
sudo xmutil listapps
```

必须：

```text
wangdao_pl_flat ... 0->0
```

然后：

```bash
sh board_full_run.sh 0xA0000000 wangdao_pl_flat
```

最终必须：

```text
WANGDAO_FULL_PS_PL_PASS sequences=10
```

---

### Stage H：板端 bit-exact 验证

```bash
sh verify_full.sh
```

最终：

```text
WANGDAO_FULL_VERIFY_PASS sequences=10
```

---

### Stage I：恢复默认 PL

回第一个 HOLD SSH：

```text
Enter
```

让 runner 自动结束并恢复：

```text
k26-starter-kits ... 0->0
```

---

### Stage J：保存板端结果

结果目录：

```text
captures/
predictions/
```

打包：

```bash
tar -czf ~/SNN_1.0/train_dev10_board_results.tar.gz \
  captures \
  predictions \
  ps_manifest.json
```

---

### Stage K：回传 PC

Windows：

```powershell
scp amd-edf@192.168.100.2:~/SNN_1.0/train_dev10_board_results.tar.gz `
  "$BoardResult\train_dev10_board_results.tar.gz"

tar.exe -xzf "$BoardResult\train_dev10_board_results.tar.gz" `
  -C $BoardResult
```

确认：

```text
captures    = 10
predictions = 10
```

---
## 14. 推荐加入“一键 Train-dev”入口

最终建议 Agent 增加一个 Windows runner，例如：

```text
run_train_dev10.ps1
```

流程自动化：

```text
artifact preflight
→ prepare PL batch
→ prepare H2 batch
→ package
→ upload
→ 提示用户启动 HOLD
→ board full run
→ verify
→ download result
→ host evaluation
→ 保存 metrics JSON
```

最终用户只需要看到：

```text
WANGDAO_TRAIN_DEV10_DEPLOY_PASS
WANGDAO_TRAIN_DEV10_BIT_EXACT_PASS
WANGDAO_TRAIN_DEV10_EVAL_PASS
```

---
## 15. 不要再重复做的事情

当前已经证明不需要：

- 重新刷 QSPI
- 更换 Linux image
- 重做网络配置
- 随机安装另一套 XRT
- 重新排查 ZOCL
- 重新制作另一套 bitstream
- 再用旧 2024-byte DTBO
- 每次实验重复很多轮固定 patch
- 因 overlay warning 就立即重做整个系统
- PC/板卡重启后重新跑已经保存在文件系统中的 10 条结果

---
## 17. 最终验收定义

本项目后续应该把以下三个层级分别定义清楚。

### Level 1：PL smoke

```text
WANGDAO_PASS
DEPLOY_OK
```

### Level 2：完整部署 bit-exact

```text
WANGDAO_FULL_PS_PL_PASS
WANGDAO_FULL_VERIFY_PASS
```

### Level 3：模型效果

```text
Train-dev metrics
Event IoU / Precision / Recall / F1 / ...
```

只有 Level 2 通过，才能说明：

> 硬件部署没有引入数值实现误差。

只有 Level 3 完成，才能说明：

> 当前第一代 SNN 实际模型性能是多少。

目前：

```text
Level 1 ✅
Level 2 ✅（完整 Train-dev 10/10）
Level 3 ⏳ 下一步计算
```

---

# 给后续 Agent 的最终任务

请基于本文直接更新第一代模型部署文件夹，目标不是只修改 README，而是把本次所有临时修复正式工程化。

优先级：

1. **替换错误 DTBO**
2. **修 `run_batch.ps1` capture size bug**
3. **修 Windows 相对路径 bug**
4. **修 dataset 跨盘 `relative_to()` bug**
5. **统一 Python 版本或修 `is_relative_to()` 兼容**
6. **新增正式 `batch_hold_run.sh`**
7. **加入 artifact hash preflight**
8. **加入 Train-dev10 自动准备、上传、运行、验证、结果回传入口**
9. **加入统一 host evaluation**
10. **保证已有已验证行为不被破坏**

更新完成后，必须至少能够重新得到：

```text
WANGDAO_PASS
WANGDAO_FULL_PS_PL_PASS sequences=10
WANGDAO_FULL_VERIFY_PASS sequences=10
```

并在 PC 上保留：

```text
captures = 10
predictions = 10
```

作为回归测试标准。
