# KV260 路线 A：Linux Vitis/XRT + PS DDR ↔ PL 实验

> 更新日期：2026-09-18  
> 状态：路线 A 的完整执行方案；当前因开发机磁盘/内存条件暂缓，待 Linux 构建环境就绪后执行。  
> 主线目标：先用 AMD 官方 `vadd` 打通 2025.1 版本闭环，再完成 SD 卡与 TCP 两组数据搬运实验，最后迁移 SNN 加速器。

---

## 总原则

路线 A 使用 Vitis/XRT 的标准加速流程：

```text
主机程序（ARM64，运行在 KV260 PS）
        ↓ XRT/OpenCL 或 XRT Native API
PS DDR 中的 BO 缓冲区
        ↓ AXI Master / NoC / HP-HPC 端口
PL 中的计算内核（xclbin）
        ↓
结果写回 PS DDR
```

必须保持严格的 2025.1 版本闭环：

```yaml
Vivado: 2025.1
Vitis: 2025.1
Platform: xilinx_kv260_base_202510_1
Common Image: xilinx-zynqmp-common-v2025.1
Sysroot: 2025.1 ARM64 sysroot
首个验证应用: AMD 官方 Simple Vector Addition（vadd）
```

首次验证阶段不要混入 2024.x、2025.2 或其他版本的 platform、sysroot、XRT、设备树、内核模块和 xclbin。当前 Generic EDF 仅用于确认板卡可启动、网络和 SSH 可用，不是最终 XRT 运行环境。

本文分为两部分：

1. 前期准备：记录已完成工作，并建立可复现的 Linux/Vitis/XRT 构建环境。
2. 正式实验：官方 vadd、SD 卡数据集、TCP 数据流和后续 SNN 迁移。

---

# 第一部分：前期准备

## 1. 已完成工作

### 1.1 板卡与连接

- 板卡：AMD/Xilinx Kria KV260 Vision AI Starter Kit，系统识别为 `ZynqMP KV260 revB`。
- 串口连接已完成，可进入板端终端。
- 网线直连 PC 已完成。
- KV260 临时静态地址已配置为 `192.168.100.2/24`。
- Windows 端地址使用 `192.168.100.1/24`。
- PC 可以 ping 通板卡，并已成功通过 SSH 登录：

```powershell
ping 192.168.100.2
ssh ubuntu@192.168.100.2
```

### 1.2 固件、系统备份与 Generic EDF

- Boot Firmware 已升级到 `1.05`。
- 原系统/原 SD 卡内容已经备份。
- Generic EDF 2025.1 已能够启动。
- 已确认 FPGA Manager 状态为 `operating`。
- 已确认当前 Generic EDF 上：
  - 用户态 XRT 命令存在；
  - `xrt-smi examine` 显示 `0 devices found`；
  - 没有可用的 ZOCL 计算设备节点；
  - `/dev/dri/card0` 是显示控制器，不是 XRT 计算设备。
- 因此已停止在 Generic EDF 上拼装 XRT、ZOCL、DKMS 或混装其他版本软件包。

这一判断是预期结论，不代表 KV260 硬件损坏。最终系统将由路线 A 生成的 `sd_card.img` 覆盖。

### 1.3 当前 Windows 开发机已有内容

在执行本路线前，应保留并核对现有安装：

- Windows 版 Vivado（可用于纯 RTL/IP Integrator 设计、综合、实现和 bitstream 生成）。
- 现有 SNN RTL/HLS/测试数据与第一次上板资料。
- 串口工具、SSH、SCP。
- 独立保存的板卡备份与恢复资料。

Windows Vivado 能生成 FPGA 侧 bitstream，但不能完整替代本路线所需的 Linux Vitis Embedded 加速流程。路线 A 还需要在受支持的 Linux 环境中完成 ARM64 host 交叉编译、`v++` kernel compile/link、platform/sysroot 绑定以及 `v++ --package`。

### 1.4 已知开发机资源限制

此前检查到的资源大致为：

```text
物理内存：约 31.7 GB
C 盘可用：约 128.9 GB
D 盘总容量/可用：约 244.1 GB / 93.5 GB
E 盘可用：约 77.5 GB
当前没有完整 Ubuntu 24.04.x + Linux Vitis 2025.1 环境
```

这套资源不适合在同一块较小磁盘上同时堆叠 Windows 工具、虚拟机、Vitis/Vivado、Common Image、sysroot 和大型 hardware build 临时文件。

建议：

- Linux 构建环境至少预留约 200 GB，建议 300 GB 或更多。
- 推荐 32 GB 以上可用内存。
- 若宿主机只有 32 GB，不要把全部内存分配给虚拟机；建议 VM 分配 20–24 GB，并配置 48–64 GB swap。
- `v++` hardware link 若出现 `Killed`、OOM 或无明显报错退出，优先检查内存、swap 和磁盘空间。
- 对后续较大的 SNN 内核，原生 Ubuntu 或远程 Linux 构建机通常比本机 VM 更稳。

---

## 2. 路线 A 的可选 Linux 构建环境

三种方案按稳定性排序：

### 方案 A1：独立 SSD 安装原生 Ubuntu（推荐）

准备一块 500 GB 或 1 TB SSD，安装 AMD 2025.1 工具支持的 Ubuntu 版本。若严格按照当前项目约定，则使用 Ubuntu 24.04.1，并在安装前再次核对 Vitis 2025.1 官方支持矩阵。

优点：

- 内存和磁盘开销最小；
- `v++` link 更稳定；
- USB、网络和文件路径问题更少；
- 最适合后续 SNN 大型构建。

### 方案 A2：虚拟机

仅建议在新增大容量磁盘后使用：

```text
虚拟磁盘：300 GB 或更大，动态扩展
内存：20–24 GB
CPU：按宿主机实际核心数合理分配
Swap：48–64 GB
网络：NAT 可用于下载；需要直连板卡时增加桥接或第二网卡
```

不要把工程放在共享文件夹中直接构建；将工程复制到虚拟机的 ext4 文件系统中，避免大小写、符号链接和 I/O 性能问题。

### 方案 A3：远程 Linux 工作站/服务器

服务器完成 Vivado/Vitis 构建，Windows PC 只负责编辑、传输镜像和板卡联调。需要确保服务器满足磁盘、内存、许可证和 Vitis 2025.1 支持系统要求。

---

## 3. 开发环境安装前检查

在 Linux 中执行：

```bash
lsb_release -a
uname -a
free -h
df -h
swapon --show
```

验收条件：

- 系统版本符合 Vitis/Vivado 2025.1 支持要求；
- 安装盘与工程盘空间充足；
- 内存与 swap 满足构建要求；
- 能正常访问 AMD 下载站和 Ubuntu 软件源；
- 系统时间正确。

建议目录：

```bash
mkdir -p ~/xilinx/2025.1
mkdir -p ~/xilinx/2025.1/common
mkdir -p ~/work/kv260
```

---

## 4. 安装 Vivado/Vitis 2025.1

从 AMD 官方站点获取 2025.1 安装器。安装时至少包含：

- Vivado 2025.1；
- Vitis 2025.1；
- Zynq UltraScale+ MPSoC 器件支持；
- Embedded Development 所需组件。

安装目录示例：

```text
/tools/Xilinx/2025.1/
```

每次打开新终端后加载环境：

```bash
source /tools/Xilinx/Vitis/2025.1/settings64.sh
```

若实际安装路径不同，以真实路径为准。检查：

```bash
which vivado
which vitis
which v++
vivado -version
v++ --version
```

三者必须指向 2025.1，不能依赖 Windows 版 Vivado 的安装目录，也不要同时加载旧版本环境脚本。

---

## 5. 优先检查 Vitis 自带 KV260 Platform

安装 Vitis 2025.1 后，先检查：

```bash
find "$XILINX_VITIS/base_platforms" -maxdepth 2 \
  -type f -name '*.xpfm' 2>/dev/null | sort

find "$XILINX_VITIS/base_platforms" -maxdepth 2 \
  -iname '*kv260*' 2>/dev/null | sort
```

目标 platform：

```text
xilinx_kv260_base_202510_1
```

如果已经存在，首次官方 vadd 验证直接使用，不重新下载，也不从源码重建 platform。这样变量最少。

设定路径示例：

```bash
export PLATFORM="$XILINX_VITIS/base_platforms/xilinx_kv260_base_202510_1/xilinx_kv260_base_202510_1.xpfm"
test -f "$PLATFORM" && echo "platform found: $PLATFORM"
```

如果目录结构不同，用 `find` 得到真实 `.xpfm` 路径。只有确认安装中确实没有该 platform 时，才从 AMD 官方渠道单独获取 `xilinx_kv260_base_202510_1`，解压后通过 `PLATFORM_REPO_PATHS` 注册：

```bash
export PLATFORM_REPO_PATHS=/absolute/path/to/platforms
```

不要为了第一个 vadd 自建 platform。

---

## 6. 下载 Common Image 并安装 2025.1 sysroot

获取：

```text
xilinx-zynqmp-common-v2025.1.tar.gz
```

解压：

```bash
mkdir -p ~/xilinx/2025.1/common
tar -xf xilinx-zynqmp-common-v2025.1.tar.gz \
  -C ~/xilinx/2025.1/common

find ~/xilinx/2025.1/common -maxdepth 3 \
  \( -name sdk.sh -o -name Image -o -name rootfs.ext4 \) -print
```

通常解压后还会有一级目录，因此不要把 `sdk.sh` 写死在上一级。示例：

```bash
cd ~/xilinx/2025.1/common/xilinx-zynqmp-common-v2025.1
ls
chmod +x sdk.sh
./sdk.sh -y -d ~/xilinx/2025.1/sysroots
```

加载 sysroot 环境脚本。实际文件名以安装输出为准：

```bash
find ~/xilinx/2025.1/sysroots -name environment-setup-* -type f
source ~/xilinx/2025.1/sysroots/environment-setup-cortexa72-cortexa53-xilinx-linux
```

检查交叉编译器：

```bash
echo "$CC"
echo "$CXX"
${CXX%% *} --version
```

### 6.1 必须理解的镜像组成

`v++ --package` 不会凭空构造一个带 XRT/ZOCL 的 rootfs。第一轮必须使用同一套 2025.1 资产：

```text
2025.1 Common Image
  ├─ Image
  ├─ rootfs.ext4
  └─ ARM64 sysroot
           +
KV260 2025.1 Platform
  └─ 硬件平台及 Linux/XRT 平台描述
           +
ARM64 vadd host + vadd.xclbin
           ↓
      v++ --package
           ↓
       sd_card.img
```

不要使用当前 Generic EDF 的 rootfs 参与首次 package。

---

## 7. 获取并固定官方 vadd 示例

优先使用 Vitis 2025.1 安装中附带或 AMD 2025.1 对应仓库中的 `Simple Vector Addition` 示例。复制到独立工作目录：

```bash
mkdir -p ~/work/kv260/vadd_official
```

记录来源和版本：

```bash
git rev-parse HEAD 2>/dev/null || true
v++ --version
```

禁止把其他发布版示例的 Makefile、host 源码或 kernel 配置与 2025.1 platform 混用。

---

## 8. 第一部分验收清单

开始正式实验前必须全部满足：

- [x] KV260 Boot FW 1.05。
- [x] 原系统已备份。
- [x] Generic EDF 2025.1 可启动。
- [x] 串口、网络、SSH 已打通。
- [x] 已确认 Generic EDF 不是最终 XRT 环境。
- [ ] 已准备足够容量的 Linux 构建环境。
- [ ] Vivado 2025.1、Vitis 2025.1、`v++` 均可用。
- [ ] 找到 `xilinx_kv260_base_202510_1.xpfm`。
- [ ] 获取 `xilinx-zynqmp-common-v2025.1`。
- [ ] 安装并加载 2025.1 ARM64 sysroot。
- [ ] 官方 vadd 源码与版本已经固定。
- [ ] 至少保留一张可重新刷写的 SD 卡。

---

# 第二部分：正式实验

## 9. 实验总体目标

第二部分分为三个闸门：

1. 官方 vadd：证明 platform、XRT、ZOCL、host、xclbin 和系统镜像闭环正确。
2. 自建数据搬运实验：分别使用 SD 卡文件和 TCP 输入，验证 PS DDR ↔ PL 的高吞吐通路。
3. SNN 迁移：在相同的 XRT 数据通路上替换计算内核，并为事件相机直连 PL 预留混合架构。

在官方 vadd 通过之前，不迁移 SNN，不修改 platform，不更换 rootfs，也不引入自定义设备树。

---

## 10. 闸门一：官方 Simple Vector Addition

### 10.1 编译 kernel

加载工具环境：

```bash
source /tools/Xilinx/Vitis/2025.1/settings64.sh
export PLATFORM="$XILINX_VITIS/base_platforms/xilinx_kv260_base_202510_1/xilinx_kv260_base_202510_1.xpfm"
```

如果官方示例自带 Makefile，优先使用其 2025.1 原始构建命令。等价流程通常包括：

```bash
v++ -c -t hw \
  --platform "$PLATFORM" \
  -k vadd \
  -o build/vadd.xo \
  src/vadd.cpp

v++ -l -t hw \
  --platform "$PLATFORM" \
  -o build/vadd.xclbin \
  build/vadd.xo
```

参数、kernel 名和源文件路径必须以所采用的官方示例为准。先用 `-t sw_emu` 或 `-t hw_emu` 可缩短早期调试，但最终必须完成 `-t hw`。

### 10.2 交叉编译 ARM64 host

```bash
source ~/xilinx/2025.1/sysroots/environment-setup-cortexa72-cortexa53-xilinx-linux
```

优先执行官方示例 Makefile。若手工编译，必须从 sysroot 获取 XRT 头文件和库，不要链接开发机本地 x86_64 库。示意：

```bash
$CXX -O2 -std=c++17 src/host.cpp \
  -o build/vadd \
  --sysroot="$SDKTARGETSYSROOT" \
  -I"$SDKTARGETSYSROOT/usr/include/xrt" \
  -L"$SDKTARGETSYSROOT/usr/lib" \
  -lxrt_coreutil -pthread

file build/vadd
```

`file` 输出必须是 AArch64/ARM64 可执行文件。

### 10.3 打包最终 SD 卡镜像

先定位同一 Common Image 下的文件：

```bash
export COMMON=~/xilinx/2025.1/common/xilinx-zynqmp-common-v2025.1
test -f "$COMMON/Image"
test -f "$COMMON/rootfs.ext4"
```

优先使用官方 vadd 工程的 package 目标。典型命令示意：

```bash
v++ --package -t hw \
  --platform "$PLATFORM" \
  build/vadd.xclbin \
  --package.out_dir package \
  --package.rootfs "$COMMON/rootfs.ext4" \
  --package.kernel_image "$COMMON/Image" \
  --package.sd_file build/vadd
```

具体选项以 Vitis 2025.1 生成的命令和官方示例为准。目标产物：

```text
package/sd_card.img
```

保存完整构建日志：

```bash
mkdir -p logs
# 在执行官方 make/package 时使用 tee 保存输出
```

### 10.4 刷卡与首次启动

1. 再次确认目标 SD 卡设备，避免覆盖开发机磁盘。
2. 使用受信任的镜像写入工具将 `sd_card.img` 写入 SD 卡。
3. 安全弹出 SD 卡并插入 KV260。
4. 串口连接保持打开，记录完整启动日志。
5. 若首次启动需要扩容、设置账户或重启，按镜像提示完成。
6. 恢复临时网络地址与 SSH。

不要在未确认设备名时直接复制带有 `/dev/sdX` 的写盘命令。

### 10.5 板端检查

```bash
uname -a
cat /etc/os-release
sudo xmutil boardid
sudo xmutil bootfw_status
command -v xrt-smi
modinfo zocl 2>&1 || true
lsmod | grep zocl || true
ls -la /dev/dri
sudo xrt-smi examine
sudo dmesg -T | grep -Ei 'zocl|xrt|fpga|dfx|drm' | tail -n 200
```

不要把 `modinfo zocl` 单独作为生死判断；驱动可能内建、模块名可能变化，或设备节点在加载 xclbin 后才出现。最终闸门是：

```text
系统正常启动
+ XRT 能识别计算设备
+ 内核日志无致命错误
+ 官方 vadd 正常运行
+ vadd 结果校验通过
```

### 10.6 运行官方 vadd

按 package 生成的目录结构进入应用目录，示意：

```bash
chmod +x ./vadd
./vadd ./vadd.xclbin
```

实际参数以官方示例 README 为准。保留：

- 串口启动日志；
- `xrt-smi examine` 输出；
- `dmesg` 相关输出；
- vadd 完整运行结果；
- 生成镜像、platform、Common Image 和工具版本记录。

只有结果校验通过，才进入下一实验。

---

## 11. 闸门二：PS DDR ↔ PL 数据搬运基线

### 11.1 工程来源

可使用已有示例目录作为起点：

```text
D:\WangDao_EventSNN\2_KV260_bring_up_tests\part2\kv260_vadd_demo
```

在 Linux 中复制到 ext4 工作目录，不要直接在 Windows 共享目录中进行 hardware link：

```bash
cp -a /path/to/kv260_vadd_demo ~/work/kv260/
cd ~/work/kv260/kv260_vadd_demo
```

### 11.2 自定义 kernel 接口

第一个自定义 kernel 保持简单，例如：

```cpp
extern "C" void stream_copy(
    const int *in,
    int *out,
    int count)
{
#pragma HLS INTERFACE m_axi port=in  offset=slave bundle=gmem0
#pragma HLS INTERFACE m_axi port=out offset=slave bundle=gmem1
#pragma HLS INTERFACE s_axilite port=in    bundle=control
#pragma HLS INTERFACE s_axilite port=out   bundle=control
#pragma HLS INTERFACE s_axilite port=count bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int i = 0; i < count; ++i) {
#pragma HLS PIPELINE II=1
        out[i] = in[i];
    }
}
```

`m_axi` 端口由 Vitis platform 接入 PS DDR；控制寄存器通过 AXI-Lite/XRT 启动。它与路线 C 中手写 AXI DMA 的目标相似，但这里的缓冲区分配、地址映射、cache 一致性和 kernel 调度由 XRT/平台协同管理。

### 11.3 基线验收

- 固定输入长度和数据类型；
- host 分配 input/output BO；
- 将 input 同步到设备；
- 启动 kernel；
- 将 output 同步回 host；
- 逐元素校验；
- 记录 H2D、kernel、D2H 和总耗时。

---

## 12. 实验一：SD 卡数据集 → PS DDR → PL

### 12.1 数据通路

```text
PC 上的数据集
      ↓ 写入 SD 卡或 SCP
KV260 Linux 文件系统
      ↓ read()/mmap()
PS 用户态缓冲区
      ↓ XRT BO 写入/同步
PS DDR
      ↓ PL m_axi burst
SNN/测试 kernel
      ↓
PS DDR 输出 BO
      ↓ XRT 同步
Host 校验并保存结果
```

### 12.2 实验步骤

1. 在 PC 端生成确定性测试数据与 golden 输出。
2. 将文件放入 SD 卡数据分区，或通过 SCP 传到板端。
3. host 读取文件头并检查尺寸、数据类型、端序和样本数。
4. 申请 XRT BO，并把输入复制到映射缓冲区。
5. 同步输入 BO，启动 kernel，等待完成。
6. 同步输出 BO，逐项与 golden 比较。
7. 对多组数据循环测试，记录吞吐率和延迟。

### 12.3 建议记录

```text
数据集版本
样本数量
单样本字节数
输入/输出 BO 大小
PL 时钟
kernel 执行时间
总端到端时间
吞吐率
错误样本数与最大误差
```

### 12.4 通过标准

- 连续运行至少 100 轮无错误；
- 输出与 golden 完全一致，或满足预先定义的定点误差阈值；
- 无 XRT、IOMMU、FPGA manager 或 kernel fatal error；
- 数据量增加时，吞吐趋势合理且没有随机 hang。

---

## 13. 实验二：TCP → PS DDR → PL

### 13.1 数据通路

```text
PC 数据发送程序
      ↓ Ethernet / TCP
KV260 socket 接收线程
      ↓ 帧解析与缓冲
PS DDR / XRT BO
      ↓ PL m_axi burst
SNN/测试 kernel
      ↓
PS DDR 输出 BO
      ↓
结果回传 PC 或保存到 SD 卡
```

### 13.2 协议建议

使用明确的二进制帧，不依赖一次 `recv()` 就得到一整帧：

```text
magic       4 bytes
version     2 bytes
flags       2 bytes
sequence    4 bytes
payload_len 4 bytes
payload     N bytes
crc32       4 bytes（可选）
```

host 必须处理粘包、拆包、超时、断线重连和非法长度。

### 13.3 双缓冲

建议使用 ping-pong BO：

```text
时刻 T：网络线程填充 BO-A，PL 处理 BO-B
时刻 T+1：网络线程填充 BO-B，PL 处理 BO-A
```

第一版先实现单缓冲正确性，再加入双缓冲和流水并行。

### 13.4 实验步骤

1. PC 端读取与 SD 实验相同的数据集。
2. 按固定协议发送到 KV260。
3. KV260 接收完整帧后写入/映射 XRT BO。
4. 同步 BO 并启动 kernel。
5. 获取结果并回传摘要或完整数据。
6. PC 端与 golden 比较。
7. 记录网络吞吐、排队时间、kernel 时间和端到端延迟。

### 13.5 通过标准

- 数据内容与 SD 实验一致时，结果一致；
- 连续长时间传输无丢帧、乱序或缓冲区越界；
- sequence 连续，非法帧能够被拒绝；
- 网络中断后能安全退出或按设计重连；
- 双缓冲版本的吞吐不低于单缓冲基线。

---

## 14. 闸门三：迁移 SNN 硬件加速器

### 14.1 迁移顺序

不要一次性把完整 SNN、TCP、多线程和自定义镜像同时加入。建议：

1. `vadd` 官方示例通过。
2. 自定义 `stream_copy` 通过。
3. 替换为单层/单时间步 SNN kernel。
4. 验证权重、膜电位、脉冲输出和定点格式。
5. 增加多层、多时间步。
6. 最后加入 TCP 双缓冲和性能优化。

### 14.2 SNN kernel 建议接口

第一版可将下列对象放入 PS DDR：

- 输入事件/特征；
- 权重；
- 神经元状态；
- 输出脉冲；
- 配置参数。

PL kernel 通过多个 `m_axi` bundle 并行访问，控制面使用 AXI-Lite。设计时明确：

- 数据位宽与定点格式；
- tensor/事件布局；
- burst 长度与对齐；
- batch、时间步和层边界；
- 状态是否跨调用保留；
- golden model 的舍入、饱和规则。

### 14.3 与事件相机直连的关系

将来事件相机从 KV260 扩展口/针脚直接进入 PL 后，实时前端不一定需要 XRT 搬运每个事件。推荐混合架构：

```text
事件相机 → PL 输入接口 → 事件预处理/SNN 数据通路
                              ↓
                        PL 内部 FIFO/BRAM

PS + XRT → 配置、权重加载、状态读取、离线回放、结果管理
```

也就是说，XRT 仍可服务于控制和 PS DDR 大块数据，而实时传感器路径可以完全留在 PL 内部。路线 A 并不强迫事件相机数据先绕到 PS。

---

## 15. 结果归档与可复现性

每次成功构建至少归档：

```text
tools.txt                 # Vivado/Vitis/v++ 版本
platform.txt              # xpfm 路径、名称、校验信息
common-image.txt          # Common Image 文件名与哈希
sysroot.txt               # environment-setup 路径
git-revisions.txt         # 示例和自有工程提交
build-command.txt         # 完整构建命令
build.log
package.log
sd_card.img.sha256
boot.log
xrt-smi.txt
dmesg-xrt.txt
test-results.csv
```

镜像与大型中间产物不要直接复制到源码目录中混放。为每个可启动版本建立唯一版本号和日期。

---

## 16. 常见问题与处理原则

### `xrt-smi examine` 显示 0 devices

先确认启动的是路线 A 生成的最终镜像，而不是 Generic EDF；随后检查 xclbin/应用加载流程、设备树、ZOCL 日志和 platform 版本。不要在 Generic EDF 上随机安装 DKMS 包。

### `modinfo zocl` 报 module not found

不能单独据此判定失败。联合检查 `lsmod`、内核配置、`dmesg`、设备节点、`xrt-smi` 和 vadd 实际执行结果。

### ARM host 无法运行

用 `file` 检查是否误编译成 x86_64；确认使用 2025.1 sysroot 的 ARM64 交叉编译器，并确认动态库来自目标 rootfs。

### `v++ -l` 被杀死或卡住

检查：

```bash
free -h
swapon --show
df -h
dmesg | tail
```

优先扩充内存/swap/磁盘，不要先修改 kernel 逻辑。

### Package 后无法启动

核对 platform、Image、rootfs.ext4、sysroot 是否全部来自 2025.1；恢复串口完整日志；必要时刷回已备份的可启动卡确认硬件和启动介质。

---

## 17. 路线 A 恢复执行顺序

当硬件资源允许后，严格按以下顺序恢复：

```text
准备独立 SSD / 原生 Ubuntu 或足够大的 VM
        ↓
安装 Vivado 2025.1 + Vitis 2025.1
        ↓
检查 Vitis 自带 xilinx_kv260_base_202510_1
        ↓
获取 xilinx-zynqmp-common-v2025.1
        ↓
安装并加载 2025.1 ARM64 sysroot
        ↓
构建 AMD 官方 vadd（hw）
        ↓
v++ --package 生成 sd_card.img
        ↓
刷卡、启动、运行 vadd 并通过结果校验
        ↓
自定义 PS DDR ↔ PL stream_copy
        ↓
SD 卡数据集实验
        ↓
TCP 数据流实验
        ↓
迁移 SNN kernel
        ↓
事件相机 PL 直连 + XRT 控制/离线混合架构
```

当前在 Linux 构建环境就绪前，不需要继续修改 KV260 上的 Generic EDF。板端只需保留串口、临时网络和 SSH 能力，等待最终 `sd_card.img`。

---

## 18. 官方资料入口

- [Vitis 2025.1 Installation Requirements](https://docs.amd.com/r/2025.1-English/ug1742-vitis-release-notes/Installation-Requirements)
- [Vitis Embedded：Pre-built Base Platforms](https://docs.amd.com/r/2025.1-English/ug1701-vitis-accelerated-embedded/Pre-built-Base-Platforms?contentId=_xeC~aHj3jOBnstAStlzHA)
- [Prepare the Common Image](https://docs.amd.com/r/2025.1-English/Vitis-Tutorials-Vitis-Platform-Creation/Prepare-the-Common-Image?contentId=Gsc3ttCeyW8DMZxWo2kZqQ)

首次执行时仍应以 2025.1 官方文档和示例自带 README/Makefile 为准；本文用于固定本项目的版本边界、执行顺序和验收条件。
