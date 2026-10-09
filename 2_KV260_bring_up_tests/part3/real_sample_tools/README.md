# KV260 冻结 SNN：真实样本位精确对拍工程

## 1. 目标与当前状态

本目录只完成一道明确的正确性闸门：把 EV-UAV 数据集中的一个真实
Proposal patch 编码为 128 个 32 位输入 word，在 PC 的冻结整数模型和
KV260 RTL 上分别推理，并要求 16 个 Q12 输出逐位一致。

最终交付文件为：

- `sample_000_input_words.txt`：128 个 32 位输入 word；
- `sample_000_expected_logits.txt`：16 个位精确 golden 输出；
- `sample_000_meta.json`：样本来源、模型哈希、编码、量化及字节映射。

已完成：

- 冻结模型可逐位复现已经上板通过的 `patch0` 和 `patch1`；
- 已从 `val_000.npz` 导出 window 0、patch 195 的真实正样本；
- 已生成 16 个 golden logits、元数据和 24 个临时输入候选；
- 已生成两组联合通道探针，组合指纹可唯一识别全部 24 种排列；
- 板端脚本已通过 POSIX shell 语法检查，并始终在退出时恢复默认 PL。

尚需一次板端测量：确认软件时间步 `t0..t3` 与 32 位 word 的四个物理
字节通道之间的映射。零输入和全 `0xFF` 输入无法分辨该映射，因此不能
凭经验直接把某个候选改名成最终文件。

## 2. 目录

```text
real_sample_tools/
├── board/                       # 安全加载、捕获和验收脚本
├── generated/lane_probe/        # 2 组输入、48 个 golden、manifest
├── common.py
├── generate_protocol_probes.py
├── compare_lane_capture.py
├── export_real_patch.py
└── upload_bundle/               # PC 端组装后整目录上传

../real_samples/sample_000/
├── input_candidates/            # 判序前的 24 个候选，不能直接作为最终交付
├── sample_000_expected_logits.txt
├── sample_000_meta.json         # 此时 board_ready=false
├── sample_000_temporal_uint8.npy
└── sample_000_target_cells.npy
```

## 3. PC 端重新生成（通常不必重复）

在 Windows PowerShell 中：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force

$python = 'D:\software\anaconda\envs\snn_env\python.exe'
$tools = 'D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\real_sample_tools'

& $python "$tools\generate_protocol_probes.py"
& $python "$tools\export_real_patch.py"
& "$tools\prepare_upload_bundle.ps1"
```

成功时分别出现：

```text
joint unique fingerprints: 24
BOARD_READY: no (run lane probe and rerun with --time-to-lane)
```

这里的 `BOARD_READY: no` 是有意的：golden 已就绪，但输入 word 还不能在
没有硬件证据时选定字节排列。

## 4. 一次性上传

`upload_bundle` 已包含经过验证的 bitstream/DTBO、原 smoke test、新脚本和
联合探针。加载脚本会先运行 `artifact_preflight.sh`，对 bitstream、DTBO 和
`shell.json` 的字节数及 SHA256 做强制校验；不匹配时不会安装或切换 PL。
不要用 archive 中的旧 DTBO 覆盖它。

在 Windows PowerShell 中执行一次：

```powershell
scp -r "D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\real_sample_tools\upload_bundle" `
  amd-edf@192.168.100.2:~/wangdao_real_sample
```

这次只输入一次 SSH 密码。

## 5. 板端只运行两条命令

先刷新一次 `sudo` 凭据，再连续完成状态清零验证和字节通道判定；正常情况下
整段只输入一次板端 `sudo` 密码：

```sh
cd ~/wangdao_real_sample
sudo -v && \
  sh safe_reset_probe.sh && \
  sh safe_lane_probe.sh
```

末尾必须同时看到：

```text
WANGDAO_ARTIFACT_PREFLIGHT_PASS
RESET_PROBE_PASS sequence=patch1,patch0,patch1
FPGA_STATE=operating
```

通道判定的成功标志只有一个：

```text
LANE_PROBE_PASS time_to_lane=XXXX probes=2
```

其中 `XXXX` 必须是 `0,1,2,3` 的一个排列。脚本结束后应恢复
`k26-starter-kits`；若异常退出，`trap` 也会尝试恢复。

## 6. 生成最终三个文件

把上一节的 `XXXX` 代入，例如探测结果为 `0123`。推荐使用带完整性检查的
收尾脚本：

```powershell
$tools = 'D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\real_sample_tools'
& "$tools\finalize_sample.ps1" -TimeToLane 0123
```

成功标志：

```text
FINAL_SAMPLE_READY time_to_lane=0123 inputs=128 logits=16
```

此时 `..\real_samples\sample_000\` 中的以下三个文件才是最终版：

```text
sample_000_input_words.txt       128 行
sample_000_expected_logits.txt    16 行
sample_000_meta.json              board_ready=true
```

## 7. 最终真实样本上板验收

将最终三个文件一次上传：

```powershell
scp "D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\real_samples\sample_000\sample_000_input_words.txt" `
    "D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\real_samples\sample_000\sample_000_expected_logits.txt" `
    "D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\real_samples\sample_000\sample_000_meta.json" `
    amd-edf@192.168.100.2:~/wangdao_real_sample/
```

板端执行：

```sh
cd ~/wangdao_real_sample
sh safe_one_sample.sh \
  sample_000_input_words.txt \
  sample_000_expected_logits.txt
```

最终必须看到：

```text
WANGDAO_PASS inputs=128 logits=16 ...
REAL_SAMPLE_PASS ...
FPGA_STATE=operating
```

`WANGDAO_PASS` 表示 16 个 word 全部逐位相同；任何一个 word 不同都会打印
索引、实际值与期望值，并以非零状态退出。

## 8. 这一步不代表什么

本实验通过证明的是一个真实 patch 的端到端位精确一致性，不等于完整数据集
检测准确率。完整指标需遍历所有窗口和 336 个 patch，执行与 PC 相同的阈值、
拼框/NMS 和标注匹配，再统计 Precision、Recall、mAP。速度需批量重复并分别
测量寄存器传输和 PL 推理时间；功耗需读取板载电源遥测或外接功率计。它们是
位精确闸门通过后的下一阶段，不能从这 16 个 logits 自动推导。
