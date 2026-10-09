# KV260 官方资料索引（适用于 SK-KV260-G）

整理日期：2026-09-10  
目标硬件：**Kria KV260 Vision AI Starter Kit，SKU：SK-KV260-G**（红色旧版外观、加密开启）。

本目录只保存 AMD/Xilinx 官方公开资料和源码；没有修改 `原始工程`、`H2-SNN-KV260` 或任何现有网络代码。

## 推荐阅读与动手顺序

1. 先阅读 `03_official_reference_implementation/01_KV260_Official_Workshop/Part 1 - Setup Board.md`：接线、microSD 镜像、串口登录与首次启动。
2. 在 Windows 主机侧配合阅读同目录的 `Linux set-up.md`、`ffplay_windows.md`、`scp.md`，完成串口、SSH/SCP 与网络联通。
3. 阅读官方在线手册 **UG1089** 的“Initial Setup”“Powering”“Boot Devices”“Vivado Board Flow”“Board Reset, Firmware Update, and Recovery”章节，再给板子上电。
4. 先运行 HLS 的矩阵乘法例程：`03_official_reference_implementation/04_HLS_Design_Flow_KV260/source/matmult`。它与本项目的 MLP 全连接层最接近，适合后续替换为定点 MAC/SNN 神经元阵列。
5. 再阅读 `02_Vitis_Embedded_Platform_Source_2025.1/Xilinx_Official_Platforms/xilinx_kv260_base`，建立 2025.1 的 KV260 基础平台；需要自行构建 Linux/设备树时再使用 `03_meta_kria_2025.1`。

## 已离线保存的官方内容

| 路径 | 内容 | 适合何时使用 |
| --- | --- | --- |
| `03_official_reference_implementation/01_KV260_Official_Workshop` | Xilinx 官方 KV260 Workshop 的全部教程正文和 39 张原始图示 | 首次上电、写卡、串口、网络、运行 Accelerated Applications |
| `03_official_reference_implementation/02_Vitis_Embedded_Platform_Source_2025.1` | AMD/Xilinx 官方 Vitis Embedded Platform Source 的 **2025.1** 分支 | 建立/改造 KV260 Vitis 基础平台 |
| `03_official_reference_implementation/03_meta_kria_2025.1` | AMD/Xilinx 官方 meta-kria 的 **rel-v2025.1** 分支 | Yocto/PetaLinux、设备树、板级 BSP |
| `03_official_reference_implementation/04_HLS_Design_Flow_KV260` | AMD University Program 官方 HLS 教程与完整源码 | FIR、卷积、矩阵乘法、Sobel；理解 HLS→Vivado→KV260 流程 |

`01_KV260_Official_Workshop` 上游仓库少数 Markdown 文件名带 `:`，不能在 Windows 文件系统中直接检出。为保证内容可用，已将它们保存为 `Part 1 - Setup Board.md` 至 `Part 4 - Optional Demo.md`；正文和原始图片均来自同一官方提交，未改写。

## 官方在线技术手册（AMD 已迁移至文档门户）

AMD 当前将这些手册从固定 PDF 链接迁移为在线文档；本机网络访问旧 PDF 链接会跳转到门户页面，因此这里保留**官方在线原文链接**而不保留伪 PDF 文件。

- [UG1089 — Kria KV260 Vision AI Starter Kit User Guide](https://docs.amd.com/r/en-US/ug1089-kv260-starter-kit/Software-Getting-Started)：最重要的板卡手册。覆盖接口、12 V 供电、启动介质、microSD、`xmutil`、加速应用、Vivado 流程、恢复与固件更新。
- [DS986 — Kria KV260 Vision AI Starter Kit Data Sheet](https://docs.amd.com/r/en-US/ds986-kv260-starter-kit/Product-Details)：板卡规格、资源、电气/机械与订货信息。该文档明确 `SK-KV260-G` 为加密开启版本。
- [DS987 — Kria K26 SOM Data Sheet](https://docs.amd.com/v/u/en-US/ds987-k26-som)：K26 SOM 资源、存储器和 PS/PL 接口规格。
- [UG1085 — Zynq UltraScale+ MPSoC Technical Reference Manual](https://docs.amd.com/v/u/en-US/ug1085-zynq-ultrascale-trm)：PS、AXI、DDR、时钟、中断、启动等底层机制；实现 PL 加速器时查阅。
- [Kria Apps 文档](https://xilinx.github.io/kria-apps-docs/)：加速应用、平台架构与构建说明。
- [2025.1 meta-kria BSP 说明](https://github.com/Xilinx/meta-kria/blob/rel-v2025.1/README.kria.bsp.md)：与本地 `03_meta_kria_2025.1` 对应的上游说明。

## 与当前 H2-SNN-KV260 项目的对应关系

- 第一阶段不需要 Vitis AI/DPU：先把定点 MLP/SNN 计算核做成 HLS 或 Verilog IP，通过 AXI-Lite 配置、AXI DMA/PS DDR 搬运数据。
- 最优先阅读本地 `source/matmult`：矩阵乘法的数据通路、片上缓冲、流水线与接口方式可以直接迁移到两级 MLP 的全连接层。
- `source/fir` 和 `source/conv_filter` 适合学习流式数据、行缓冲与 HLS pipeline；`source/sobel` 是完整的 HLS/overlay/PYNQ 演示，但不应作为 SNN 算法基线。
- 官方 Workshop 的系统镜像和 Accelerated Application 示例较早，适合学习首次启动，不要将其旧版软件包与 Vivado/Vitis 2025.1 工程混用。新工程应以本地的 2025.1 Vitis 平台和 meta-kria 分支为准。

## 首次通电前的最低清单

- 12 V、至少 3 A、5.5 mm 外径 / 2.5 mm 内径、中心正极电源；
- 16 GB 或更大的 microSD 卡与读卡器；
- USB-A 转 Micro-B **数据线**（用于 J4 的 UART/JTAG）；
- 网线；建议另备 HDMI 显示器与键盘（不是最小必需条件）。

首次启动建议只进行官方镜像写卡、串口观察和网络确认；暂不要更新 QSPI 固件或改动启动配置。确认能稳定进入 Linux 后，再创建本项目专用的 FPGA 工程目录。
