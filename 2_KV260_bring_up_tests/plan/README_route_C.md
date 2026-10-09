# KV260 路线 C：第三部分优先——冻结版 SNN 真实样本推理

> 更新时间：2026-09-19  
> 文档定位：独立于主 `README.md` 的第三部分执行文档，不覆盖原路线 C 总计划。  
> 当前决策：暂时跳过第二部分 AXI DMA/AXI-Stream，不使用第二部分尚未建立的 DMA 框架。  
> 当前目标：在已经通过双 patch 的冻结版 SNN 上，打通“真实事件样本 → 128 个输入 word → 16 个 Proposal logits → 软件 golden 对照”。

---

## 0. 下一位智能体先读这里

不要重新刷 SD/QSPI，不要重新安装 XRT、ZOCL、编译器、内核开发包，也不要重新制作另一套 DTBO。

当前已经完成：

```text
KV260 / EDF 2025.1 启动                         PASS
永久静态地址 192.168.100.2/24                  PASS
SSH 重启后可用                                  PASS
修正版 runtime DTBO                            PASS
wangdao_pl_flat 加载                            PASS
AXI-Lite 128-word 写入                          PASS
start / status=0x00000002                       PASS
16 个输出读取                                   PASS
patch0 / patch1 golden                          PASS
== DEPLOY_OK ==                                 PASS
测试后恢复 k26-starter-kits                     PASS
```

现在的直接任务不是 DMA，而是：

```text
1. 先把板端已经验证过的“工作版文件”备份回 PC
2. 确认 4 个时间步在每个 32-bit word 中的字节排列
3. 确认一次 start 是否会清空上一样本的 SNN 状态
4. 用现有软件工程导出 1 个真实 patch 的 [4,128] uint8 特征
5. 生成 128 个 packed words 和 16 个 raw golden logits
6. 只传该样本的派生文件及审核后的通用 runner 到板卡
7. 短时运行，逐 word 对照，并恢复 k26-starter-kits
```

第二部分只是暂停，不是删除。当前冻结 bitstream 没有 DMA/AXI-Stream 接口，真实样本阶段继续使用 AXI-Lite，性能结果只能作为功能验证。

---

## 1. 当前实际硬件通路

```text
板端文本文件中的 128 个 32-bit word
            ↓ devmem / AXI-Lite
0xA0000100 ～ 0xA00002FC
            ↓
冻结版 Proposal SNN
            ↓
0xA0000900 ～ 0xA000093C
            ↓ devmem / AXI-Lite
16 个 32-bit raw logits
```

寄存器：

| 地址 | 作用 |
|---|---|
| `0xA0000000` | control，写 bit0=1 启动 |
| `0xA0000004` | status，bit0=busy、bit1=done、bit2=error |
| `0xA0000008` | config，只读检查 |
| `0xA0000100`～`0xA00002FC` | 128×32-bit 输入 |
| `0xA0000900`～`0xA000093C` | 16×32-bit 输出 |

正常完成状态已经观察到：

```text
status = 0x00000002
busy   = 0
done   = 1
error  = 0
```

### 1.1 当前冻结 SNN 的验证结论

截至 2026-09-19，固定向量功能冒烟已经判定成功，依据不是“程序没有报错”，而是以下闸门全部通过：

```text
[PASS] 修正版 DTBO 能被当前 EDF runtime overlay 正常加载
[PASS] xmutil 显示 wangdao_pl_flat 占用 slot 0
[PASS] /sys/class/fpga_manager/fpga0/state = operating
[PASS] patch0 写入128个 0x00000000
[PASS] patch0 读回16个 0x00000000
[PASS] patch1 写入128个 0xFFFFFFFF
[PASS] patch1 读回的16个 word 与 expected 逐 word 完全一致
[PASS] 两次最终 status 都是 0x00000002
[PASS] 两次均打印 WANGDAO_PASS inputs=128 logits=16
[PASS] 总流程打印 == DEPLOY_OK ==
[PASS] 结束后 wangdao_pl_flat 已卸载
[PASS] 结束后 k26-starter-kits 已恢复到 slot 0
[PASS] 恢复后 FPGA manager 仍为 operating
```

最终成功日志：

```text
~/wangdao_route_c_snn_smoke_fixed_wrapper.log
```

必须能够从日志中找到：

```text
WANGDAO_PASS inputs=128 logits=16 status=0x00000002 polls=0
WANGDAO_PASS inputs=128 logits=16 status=0x00000002 polls=0
== DEPLOY_OK ==
```

`polls=0` 只表示第一次读取 status 时硬件已经完成，不是失败。真正的成功条件是 `busy=0、done=1、error=0`，并且16个结果全部匹配。

因此当前可以准确表述为：

> 冻结版 Proposal SNN 已在 KV260 PL 上真实运行；两个已知输入向量的128×32-bit AXI-Lite 写入、启动、完成状态、16×32-bit 输出以及软件期望值均完全匹配，测试后默认 PL 已安全恢复。

当前结果**不能**表述为：

```text
真实样本已经通过
完整数据集已经通过
检测准确率已经验证
吞吐率已经验证
DMA 或 AXI-Stream 已经通过
```

下一阶段“真实样本成功”的判定更严格：真实 patch 必须来自已确认的数据预处理，128个 word 的时间步打包顺序必须已经证实，FPGA 的16个 raw Q12 logits 必须与位精确软件模型逐 word 一致，并且测试后恢复 `k26-starter-kits`。详细闸门见第10节。

---

## 2. 已确认的软件 golden 模型

本地存在位精确模型：

```text
D:\WangDao_EventSNN\0_mlp_original_project\software\weights\S1P_INT8_ACC24_MEM24_Q12.npz
SHA256 F05F40359E70FACE33A0D513ABCFACC5A1519BA2D99DF73076B41193ABE4EF69
```

其来源浮点权重：

```text
D:\WangDao_EventSNN\0_mlp_original_project\software\weights\S0P_T4_POS15_KD50_S42.npz
SHA256 FB5FD635EE8F4B9660F98D6C71299DE9E40B9D14FA83E51B2306853B2DB86952
```

模型配置：

```text
结构                    128 → 48 → 24 → 16
time_steps              4
输入                    uint8 / Q0.8
累加器                  signed 24-bit
膜电位                  signed 24-bit / Q12
beta                    7/8
threshold               0.25
reset                   soft_reset
readout                 mean_membrane
输出 raw logits_q       signed integer / Q12
```

已在本机 `snn_env` 中运行该模型进行对照：

```text
4×128 全零输入
→ 16 个 0x00000000
→ 与板端 patch0 完全一致

4×128 全 0xFF 输入
→ FFFFB0D8 FFFFCC02 FFFFECC2 FFFFF973
  FFFF984B FFFF8E5D FFFFBA6C FFFFE996
  FFFFAC04 FFFF9330 FFFF9D64 FFFFCBEF
  FFFFD86C FFFFAE80 FFFF9BAC FFFFBA5C
→ 与板端 patch1 的16个 word 完全一致
```

因此 `S1P_INT8_ACC24_MEM24_Q12.npz` 是当前冻结 bitstream 的**强匹配软件 golden 基线**。虽然没有完整 Vivado 源工程可证明 bitstream 的生成链，但两个极端向量的16路 raw 输出逐字一致，后续真实样本应首先使用该位精确模型，而不是浮点模型直接比较。

本机验证环境：

```text
D:\software\anaconda\envs\snn_env\python.exe
```

---

## 3. 16 个输出不是“16个类别”

现有软件工程给出的空间定义：

```text
传感器                  346 × 260
时间窗                  100 ms
时间步                  4 × 25 ms
空间 patch              32 × 32
patch stride            16
每个窗口 patch 网格     16 × 21 = 336 个 patch
输入 cell               4 × 4 像素
输入维度                2 个极性 × 8 × 8 = 128
输出 cell               8 × 8 像素
单 patch 输出           4 × 4 = 16
```

因此冻结硬件的一次调用处理的是**一个空间 patch**：

```text
一个 patch 的 4 个时间步 × 128 个 uint8 特征
                     ↓
          16 个 4×4 Proposal heatmap logits
```

这16个输出不是16个语义类别。它们对应 32×32 patch 内的 4×4 粗网格位置。后续不能直接对16个值做“16分类准确率”。

候选预处理代码：

```text
D:\WangDao_EventSNN\0_mlp_original_project\software\wangdao\h2\representation.py
D:\WangDao_EventSNN\0_mlp_original_project\software\wangdao\s0p.py
D:\WangDao_EventSNN\0_mlp_original_project\software\wangdao\fixed.py
```

关键函数：

```text
build_temporal_patch_features(...)
  → [window, time=4, patch=336, feature=128] uint8

build_temporal_patch_targets(...)
  → [window, time=4, patch=336, output=16]

FixedProposalSNN.forward(...)
  → 输入 [time=4, batch, 128]
  → 输出 logits_q [batch, 16]
```

128维特征的排列由软件定义为：

```text
polarity-major
→ 8×8 个 4-pixel feature cells
→ 展平为 2 × 8 × 8 = 128
```

计数压缩使用软件工程中的 `compress_counts()`，不能用未经核对的线性归一化代替。

---

## 4. 目前仍需确认的两个硬件协议

### 4.1 四个时间步的 word 打包顺序

软件输入是：

```text
temporal[time=4, feature=128]，每项 uint8
```

硬件寄存器是：

```text
input_word[feature=128]，每项 uint32
```

最可能的结构是每个 feature 的4个时间步打包成一个 word：

```text
word[i] = byte_lane_0 | byte_lane_1<<8 | byte_lane_2<<16 | byte_lane_3<<24
```

但当前全零和全 `0xFF` patch 无法区分时间步字节顺序。以下内容仍然未知：

```text
bits[7:0]   对应 t0、t1、t2 还是 t3
bits[15:8]  对应哪个时间步
bits[23:16] 对应哪个时间步
bits[31:24] 对应哪个时间步
```

不得直接假设“小端主机”就等于硬件时间步顺序。下一步应生成非对称的4时间步 probe，由软件模型计算所有候选顺序的16路输出，再与一次板端结果比较。

建议 probe 使用彼此不同且不会全部落在零响应区的时间步，例如从一个真实 patch 取 `[4,128]` 特征，而不是继续用全零/全 `0xFF`。

### 4.2 每次 start 是否清空神经元状态

软件 `FixedProposalSNN.forward()` 每个样本从零膜电位开始。当前执行顺序是 patch0 后 patch1：patch0 全零不会充分证明下一次 start 会清空状态。

在批量真实样本前应做一次短测试：

```text
patch1 → patch0 → patch1
```

验收：

```text
中间 patch0 仍为全零
两次 patch1 的16个输出完全相同
```

若不满足，说明内部状态可能跨 start 保留。在找到明确 reset 控制前，不得连续批量推理；第一阶段只能重新加载 PL 后运行单一样本。又因为 overlay 历史上出现过 refcount/memory-leak 警告，也不能靠数千次 load/unload 代替正确 reset。

---

## 5. 已验证板端工作包（已完成备份）

板端成功版本已经备份到：

```text
D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\board_working_2026-09-19\wangdao_board_package
```

真实样本工具使用的自包含副本位于：

```text
D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\real_sample_tools\upload_bundle
```

两个活动包都必须通过 `artifact_preflight.sh`，至少确认：

```text
bit.bin = 3BF7473F26F5DDCD1157C2513706D179982EB131E8CB38880601D30E105DEE23
dtbo    = 0A1C9EB846B6929129B315B5E3FF510A06E332A74744DC048A9BC46C47CA2706
```

修正前的 2024-byte DTBO 已隔离到：

```text
D:\WangDao_EventSNN\2_KV260_bring_up_tests\archive\legacy_snn_package_before_cleanup\wangdao_pl_flat.dtbo
SHA256 4D058B969FD960013B697E90F132B7635BB0AF32EAE8AEF6E0C6FFE83845D32F
```

该文件只用于历史追溯，禁止复制回活动包或板卡。上传前必须先看到：

```text
WANGDAO_ARTIFACT_PREFLIGHT_PASS
```

---

## 6. 真实样本在 PC 上应生成什么

原始数据集、Python 环境和模型权重留在 PC 上。对每一个准备上板的 patch，只导出三个小文件：

```text
sample_000_input_words.txt       128 行，每行一个 0xXXXXXXXX
sample_000_expected_logits.txt    16 行，每行一个 0xXXXXXXXX
sample_000_meta.json              样本来源与编码元数据
```

`sample_000_meta.json` 至少记录：

```json
{
  "source_sequence": "val_xxx.npz",
  "window_index": 0,
  "patch_index": 0,
  "time_steps": 4,
  "feature_shape": [4, 128],
  "word_count": 128,
  "word_byte_order": "TO_BE_CONFIRMED",
  "golden_model": "S1P_INT8_ACC24_MEM24_Q12",
  "golden_model_sha256": "F05F40359E70FACE33A0D513ABCFACC5A1519BA2D99DF73076B41193ABE4EF69",
  "logit_format": "signed Q12 raw int32",
  "expected_count": 16
}
```

第一版 PC 导出工具建议放在新目录：

```text
D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\real_sample_tools\
```

建议文件：

```text
export_real_patch.py       从事件文件选 window/patch，生成 [4,128]
pack_input_words.py        按已确认 byte order 生成128个 word
golden_logits.py           调用 FixedProposalSNN 生成16个 logits_q
verify_export.py           检查数量、范围、哈希和可重复性
```

在时间步字节顺序尚未确认前，`pack_input_words.py` 只能生成 probe，不能把假定顺序标记为正式格式。

---

## 7. 到底需要传什么到板卡

### 7.1 当前板卡未重刷、工作目录仍在时

现有硬件包和修复脚本已经在板端，**不需要重新传 bitstream、DTBO、shell.json、devmem wrapper 或两个固定 patch**。

确认：

```bash
cd ~/wangdao_board_package
sha256sum wangdao_pl_flat.bit.bin wangdao_pl_flat.dtbo
ls -l shell.json wangdao_axi_smoke.sh safe_board_run.sh
```

接下来真正需要传到板卡的只有：

```text
必须：
  sample_000_input_words.txt
  sample_000_expected_logits.txt
  sample_000_meta.json

等创建并审核后再传：
  wangdao_infer.sh 或 wangdao_infer（二选一）

为确认协议而临时传：
  时间步 byte-lane probe 输入/expected
  patch1→patch0→patch1 reset probe 文件或 runner
```

### 7.2 不需要传到板卡

```text
整个原始事件数据集
Anaconda / snn_env
PyTorch
原始工程全部 Python 源码
S1P_INT8_ACC24_MEM24_Q12.npz
Vivado/Vitis 工程
AXI DMA 文件
D 盘旧版 wangdao_pl_flat.dtbo
D 盘旧版 wangdao_axi_smoke.sh
```

这些内容在 PC 上负责预处理和产生 golden。板卡只接收最终的128个输入 word、可选16个 expected、元数据和小型 runner。

### 7.3 如果板卡被重刷或 `~/wangdao_board_package` 丢失

从第 5 节备份的 `board_working_2026-09-19` 传输：

```text
wangdao_pl_flat.bit.bin
wangdao_pl_flat.dtbo          必须是 SHA256 0A1C9E...2706
shell.json
wangdao_axi_smoke.sh          使用板端已修正版
safe_board_run.sh
patch0_words.txt
patch0_expected.txt
patch1_words.txt
patch1_expected.txt
```

不要从原始目录取旧 DTBO。`wangdao.bit.bin` 不必传，可由安装脚本从 `wangdao_pl_flat.bit.bin` 复制别名。

---

## 8. 最小传输命令

假设 PC 已生成：

```text
D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\real_samples\sample_000\
```

Windows PowerShell：

```powershell
$src = 'D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\real_samples\sample_000'

ssh amd-edf@192.168.100.2 'mkdir -p ~/wangdao_real_samples/sample_000'

scp "$src\sample_000_input_words.txt" `
    "$src\sample_000_expected_logits.txt" `
    "$src\sample_000_meta.json" `
    amd-edf@192.168.100.2:~/wangdao_real_samples/sample_000/
```

如果已经创建并审核通用 runner：

```powershell
scp 'D:\WangDao_EventSNN\2_KV260_bring_up_tests\part3\real_sample_tools\wangdao_infer.sh' `
    amd-edf@192.168.100.2:~/wangdao_board_package/
```

板端核对：

```bash
cd ~/wangdao_real_samples/sample_000

grep -Ev '^[[:space:]]*(#|$)' sample_000_input_words.txt | wc -l
grep -Ev '^[[:space:]]*(#|$)' sample_000_expected_logits.txt | wc -l
sha256sum sample_000_input_words.txt sample_000_expected_logits.txt sample_000_meta.json
```

必须分别得到128行和16行。

---

## 9. 第一个真实样本的安全运行原则

在 byte order 和 reset 语义确认前，不要直接做批量循环。第一个样本应复用已经修好的 `wangdao_axi_smoke.sh`，因为它能够严格比较128个输入和16个 expected。

逻辑命令为：

```bash
sudo sh ~/wangdao_board_package/wangdao_axi_smoke.sh \
  0xA0000000 \
  ~/wangdao_real_samples/sample_000/sample_000_input_words.txt \
  ~/wangdao_real_samples/sample_000/sample_000_expected_logits.txt
```

但只能在以下安全外壳中执行：

```text
确认 k26-starter-kits 在 slot 0
→ 设置 EXIT/signal trap
→ 加载 wangdao_pl_flat
→ 确认 fpga_manager=operating
→ 运行一个样本
→ 保存输出/dmesg
→ 无论成功失败都恢复 k26-starter-kits
```

优先复用板端已验证的 `safe_board_run.sh` 的加载、trap 和恢复逻辑，不要重新写一个没有清理路径的简化脚本。修改为通用 `wangdao_infer.sh` 前，必须先展示和审核完整脚本。

当前温度读取路径仍未闭环，因此：

```text
允许：单个 probe、单个真实 patch、少量短测试
暂缓：数百/数千 patch、完整序列、长时间 stress、性能结论
```

overlay 有历史 refcount/memory-leak 警告，不要无意义地反复 load/unload；出现 overlay 异常优先重启，重启后静态网络会自动恢复。

---

## 10. 第一个真实样本的验收标准

```text
[ ] 使用的 bitstream SHA256 为 3BF747...DEE23
[ ] 使用的 DTBO SHA256 为 0A1C9E...2706
[ ] 软件 golden 模型 SHA256 为 F05F40...EF69
[ ] 输入确实来自 build_temporal_patch_features
[ ] 输入原始形状为 [4,128] uint8
[ ] 32-bit word 的4个 byte lane 顺序已经用非对称 probe 证实
[ ] start/reset 行为已经用 patch1→patch0→patch1 证实
[ ] 输入文件严格128个 word
[ ] expected 文件严格16个 word
[ ] status 最终为 0x00000002
[ ] FPGA 16个 raw word 与 logits_q 逐 word 完全一致
[ ] 保存 sample meta、输入、expected、实际输出和完整日志
[ ] 测试后 k26-starter-kits 回到 slot 0
[ ] 没有把16个输出误写成16分类结果
```

只有这些通过，才能把结果写成：

> 冻结版 Proposal SNN 的一个真实 patch 已与位精确软件模型逐 word 对齐。

不能写成：

```text
完整数据集已跑通
检测准确率已验证
吞吐率已验证
DMA 已完成
事件相机已完成
```

---

## 11. 一个样本通过后的顺序

```text
1 个真实 patch 精确匹配
        ↓
选取同一 window 的 10 个不同 patch
        ↓
确认 reset 和重复性
        ↓
短时运行一个 window 的少量 patch
        ↓
补齐温度读取和风扇行为验证
        ↓
再决定是否用 AXI-Lite 做小规模数据集功能验证
```

一个100 ms window 含336个 patch。仅靠 `devmem` 逐寄存器访问可以验证功能，但速度慢、系统调用多，不应将其结果视为最终吞吐性能。

如果后续目标变成完整数据集、高吞吐 TCP 或事件相机实时输入，仍必须回到第二部分或取得源工程后重构 AXI-Stream/DMA。当前 `.bit.bin` 不能靠软件转换成 DMA 版本。

---

## 12. 给下一位智能体的明确任务单

按顺序执行，不要跳步：

```text
[ ] A. 从板端备份修正版 DTBO、smoke、safe runner、wrapper 和成功日志
[ ] B. 核对工作版 DTBO 哈希 0A1C9E...2706
[ ] C. 固定软件 golden 为 S1P_INT8_ACC24_MEM24_Q12.npz
[ ] D. 编写 PC 侧真实 patch 导出工具
[ ] E. 用非对称 probe 确认4个时间步的 byte-lane 顺序
[ ] F. 用 patch1→patch0→patch1 确认 start/reset 语义
[ ] G. 导出 sample_000 的128 words、16 expected 和 meta
[ ] H. 只上传这三个派生文件及审核后的通用 runner
[ ] I. 在安全 trap/恢复流程中运行一个真实 patch
[ ] J. 逐 word 比较并归档日志
[ ] K. 恢复 k26-starter-kits
```

当前最关键的技术问题已经从“硬件能否运行”收敛为两个小协议问题：

```text
4 个时间步如何打包进每个 32-bit word
每次 start 是否重置 SNN 内部状态
```

在这两个问题确认后，真实样本上板不需要 DMA，也不需要修改 bitstream。
