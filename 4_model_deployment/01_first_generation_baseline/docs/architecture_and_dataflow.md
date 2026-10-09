# 架构与数据通路

来源：`# 第一代 SNN 模型 KV260 部署：问题复盘、已验证修复与标准流程`。目录重构时按主题拆分，技术口径未改变。本文记录第一代已实测成功链路。

## 2. 当前硬件与软件架构

### 2.1 板卡与系统

- Board：AMD/Xilinx Kria KV260 / K26
- Board ID 历史确认：`SCK-KV-G`
- Linux：AMD Embedded Development Framework Linux
- Kernel：
  ```text
  6.12.10-xilinx-g0a0f70e531c7
  ```
- FPGA application manager：
  ```bash
  xmutil
  ```
- 默认应用：
  ```text
  k26-starter-kits
  ```
- 第一代 SNN PL 应用：
  ```text
  wangdao_pl_flat
  ```

### 2.2 PC 与板卡网络

KV260：

```text
192.168.100.2/24
```

Windows PC 直连网卡：

```text
192.168.100.1/24
```

SSH：

```powershell
ssh amd-edf@192.168.100.2
```

### 2.3 第一代模型实际部署结构

完整第一代模型不是“全部放进 FPGA”。

真实链路：

```text
EV-UAV events
    ↓
100 ms windows
    ↓
336 patches/window
    ↓
Proposal-SNN @ PL
128 → 48 → 24 → 16
T = 4
    ↓
16-D Proposal logits
    ↓
候选生成 @ PS
    ↓
32-D feature @ PS
    ↓
H2 Refiner @ PS
32 → 64 → 32 → 4
    ↓
event predictions
```

也就是说：

```text
Proposal-SNN        → PL
候选 / feature / H2 → PS ARM
```

---
## 3. PL 硬件接口

当前 bitstream 使用 AXI-Lite，不使用 DMA。

Base：

```text
0xA0000000
```

寄存器：

```text
control       +0x000
status        +0x004
config        +0x008

input regs    +0x100 ~ +0x2FC
              128 × 32-bit

output regs   +0x900 ~ +0x93C
              16 × 32-bit
```

状态寄存器：

```text
bit0 = busy
bit1 = done
bit2 = error
```

正常完成时已多次实测：

```text
status = 0x00000002
```

---

## 迁移后端到端数据周转

1. PC 读取 `dataset/index.json`，经 `host_tools/deployment_preparation/` 将事件按 100 ms 窗口和每窗 336 个 patch 转换为板端二进制包，同时生成 golden、PS/H2 输入与 manifest。
2. PC 自动化脚本将显式组装的部署包上传到 KV260 的 SD 卡文件系统；包不依赖仓库内部相对层级。
3. PS 上的 C 驱动从 SD 卡文件读取每个 512-byte patch（128×32-bit），通过 `/dev/mem` 映射的 AXI-Lite 寄存器窗口写入 PL；当前 bitstream 没有 DMA。
4. PS 写 control 寄存器启动一次 Proposal-SNN，轮询 status 的 busy/done/error；PL 完成 `128→48→24→16, T=4` 并给出 16 维 Q12 logits。
5. PS 从 16 个输出寄存器读回 logits，在 ARM 上执行候选生成、32 维特征构建与 H2 Refiner（`32→64→32→4`），形成事件预测。
6. PS 将 prediction、capture、runtime、功耗样本和日志写回 SD 卡；PC 下载后由 `host_tools/result_evaluation/` 评分，正式运行落在 `experiments/board_runs/<run_name>/`。

职责边界：PC 只做离线准备、上传下载和评测；PS 负责文件 I/O、AXI-Lite 驱动、候选/特征/H2 与结果落盘；PL 只运行 Proposal-SNN。第二代保持同一 PL 固件，仅优化 PS/软件侧路径。
