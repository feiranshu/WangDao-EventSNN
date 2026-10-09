# KV260 到手验收、启动与开发指南（中文实操版）

适用对象：**Kria KV260 Vision AI Starter Kit，SK-KV260-G**（旧版红色板亦适用）。  
整理日期：2026-09-14。  
依据：AMD/Xilinx 官方 KV260 Linux Boot、UG1089、DS986 与 BIST 文档。

> 本文是面向本项目的中文操作归纳，不是官方教程的逐句译文。所有可能随发行版变化的镜像、固件与命令，以文末官方链接为准。

## 0. 先选对路线

扫码链接是 **2022.1** 年代的文档入口，并不意味着必须安装 2022.1 系统。

| 目的 | 推荐系统/路线 | 不推荐的做法 |
| --- | --- | --- |
| 到手验板、检查基本硬件 | 官方 Ubuntu Server 24.04 + BIST 1.2（可选） | 一上来就改写 QSPI 或烧录自己的 BOOT.BIN |
| 学习旧版官方视觉演示 | Ubuntu Desktop 22.04 + 相应旧版应用 | 为运行旧演示而把后续 FPGA 工程长期锁死在 2022.1 |
| H2-SNN-KV260 的最终自定义部署 | Vivado/Vitis 2025.1 + 本地 `Vitis_Embedded_Platform_Source_2025.1` / `meta_kria_2025.1` | 把 Ubuntu/BIST 镜像当作最终产品镜像 |

本项目建议：先用一张独立 microSD 卡进行 Ubuntu 24.04 验板；验收通过后，再单独制作 2025.1 的开发/部署卡。这样恢复与排障最简单。

## 1. 收货后、上电前的静态检查

1. 核对板卡标签为 `SK-KV260-G`；红色 PCB 的旧版属于正常外观，不代表是其他型号。
2. 检查 K26 SOM、风扇/散热器、两条排线/连接器附近没有松动、弯针、进液或明显返修痕迹。
3. 检查风扇已经插在 **J13**；KV260 正常配置下风扇是板载主动散热的一部分。
4. 核对电源铭牌：**12 V DC、至少 3 A、5.5 mm 外径 / 2.5 mm 内径、中心正极**。只允许从 **J12** 输入主电源。
5. 准备一张空的 16 GB 或更大 microSD 卡（建议 32 GB、Class 10/A1）和读卡器。首次验收卡与未来项目卡分开。
6. 准备 **USB-A 转 Micro-B 数据线**。仅充电线无法提供 UART/JTAG 通讯。
7. 建议准备网线；HDMI/DisplayPort 显示器、键盘鼠标不是最小必需项，但会让首次排障容易很多。

**不要做：**不要在未能稳定启动 Linux 前更新 QSPI 固件；不要把不明 `BOOT.BIN` 写入 QSPI；不要以 USB 口给板子供电。

## 2. 最小连接与首次上电

KV260 没有单独的电源开关；电源适配器接入市电后即开始上电启动。

按下列次序操作：

1. 将已写好系统镜像的 microSD 插入 **J11**。
2. 将 Micro-B 端插到 **J4**；先不要把 USB-A 端接到电脑。
3. 接好网线到 **J10**。可选：接 HDMI（J5）或 DisplayPort（J6），再接 USB 键盘/鼠标。
4. 将电源圆头插入 **J12**，但暂时不插市电。
5. 将 USB-A 端接到电脑；在 Windows 设备管理器观察新增的 FTDI 串口。
6. 最后把电源接入市电。风扇和电源指示灯应工作；约 10–15 秒应有 UART 启动输出，完整 Linux 启动通常需约 1 分钟。

若有心跳灯但显示器无输出，先确认显示器已开机并选对 HDMI/DP 输入源；不要先判断板子损坏。

## 3. 写入“验板 microSD 卡”

### 3.1 为什么先用 Ubuntu 24.04

目前官方面向 KV260 的新用户建议 Ubuntu 24.04；BIST 1.2 的预构建测试环境目标也是 Ubuntu 24.04。它适合验证板卡本身，而不是作为最终 SNN 系统。

### 3.2 写卡步骤

1. 在 [Ubuntu for AMD](https://ubuntu.com/download/amd) 页面选择 **Kria K26 SOMs → Ubuntu Server 24.04 LTS** 对应 KV260 的镜像。
2. 下载镜像及其官方校验信息（若页面提供）。写卡前校验下载文件的哈希。
3. 安装 Raspberry Pi Imager，选择“使用自定义镜像”，选中下载的镜像，再仔细选择目标 microSD 盘符。
4. 开始写入，完成后让工具执行验证；安全弹出 microSD。
5. 把卡插进 J11，进入第 2 节的接线/上电流程。

写卡会**清空所选 microSD**。确认盘符正确，绝不能选择电脑硬盘或装有项目数据的移动盘。

## 4. Windows 串口登录与首轮健康检查

### 4.1 串口设置

KV260 的 J4 使用 FTDI USB 转串口/JTAG。Windows 通常会枚举四个 COM 口，其中**编号第二个**为 UART；若未出现，安装 [FTDI VCP 驱动](https://ftdichip.com/drivers/vcp-drivers/)。

用 PuTTY 或 Tera Term 打开该 COM 口：

```text
波特率：115200
数据位：8
停止位：1
校验：None
流控：None
```

Ubuntu 官方镜像的首次默认账户通常为 `ubuntu` / `ubuntu`，首次登录后按系统提示修改密码。密码由你自己设置，勿写入项目文档或提交到仓库。

### 4.2 基础健康检查（只读、建议必做）

登录后依次执行。以下命令不会刷写 QSPI，也不会编程 PL：

```bash
uname -a
cat /etc/os-release
lsblk
ip -br link
ip -br addr
lsusb
dmesg -T | tail -n 80
sudo xmutil bootfw_status
sudo xmutil listapps
```

记录终端输出到验收记录中。最低通过标准：

- 系统完整启动、能够登录；
- 能看到 microSD 根文件系统；
- J10 网口获取地址或链路状态为 UP；
- `xmutil bootfw_status` 能返回启动固件状态（镜像带有该工具时）；
- 无持续出现的供电、I2C、MMC 或风扇错误。

有显示器时，还应确认：桌面/控制台图像稳定；插拔 USB 键盘或 U 盘能被 `lsusb` 识别；风扇持续转动。

### 4.3 关机规则

每次断电前先执行：

```bash
sudo shutdown -h now
```

待串口停止输出、文件系统卸载后，再拔市电。直接拔电可能损坏 microSD 中的系统文件。

## 5. 如何判定“板子功能齐全”

不要将“所有接口都测试一遍”与“基础验收”混为一谈。摄像头、四个 U 盘、PMOD 测试转接板、显示器都不是板卡本体自带物；没有这些附件时，对应接口不能判为故障，只能标记为“未测”。

### A. 基础验收（建议到手当天完成）

| 项目 | 方法 | 通过表现 |
| --- | --- | --- |
| 供电与风扇 | 上电后观察 | 风扇转动，无异味/异常发热 |
| QSPI 启动链 | UART 启动日志 + `xmutil bootfw_status` | 能进入 Linux，无反复重启 |
| microSD | `lsblk`、实际从卡启动 | 系统盘可读写，启动稳定 |
| USB-UART/JTAG | Windows 出现 FTDI 端口，UART 可登录 | 115200 串口稳定无乱码 |
| 网口 J10 | 接路由器或电脑 | 获得 IP 或可 ping 网关 |
| 显示输出（有屏时） | HDMI/DP 接显示器 | 有启动/桌面输出 |
| USB Host | 插入一个普通 U 盘 | `lsusb`、`lsblk` 能看到设备 |

### B. 官方 BIST 深度验收（可选）

官方 BIST 1.2 可覆盖 EEPROM、I2C、INA260 电流监控、风扇 PWM、TPM、以太网、USB、显示、microSD、摄像头等；完整测试需要 **4 个 U 盘、PMOD TPH2 与跳线、显示器、AR1335 摄像头、Raspberry Pi Camera v2、主机 iperf3**。接口与外设对应表见 [官方 KV260 BIST Board Setup](https://xilinx.github.io/kria-apps-docs/kv260/2022.1/bist/1_2/build/html/docs/setup_kv260.html)。

BIST 1.2 目标为 Ubuntu 24.04，官方测试环境标注为 AMD 2024.1。它用于硬件诊断，**不要把它的 PL 固件当成你的 SNN 部署固件**。

建议按风险由低到高运行：

```bash
# 安装并加载官方 BIST 固件
sudo apt install xlnx-firmware-kv260-bist
sudo xmutil listapps
sudo xmutil unloadapp
sudo xmutil loadapp kv260-bist

# 仅列出测试，不执行
cd /opt/xilinx/kria-bist/tests
pytest-3 --collect-only --board kv260
```

完整 BIST 会包含 QSPI 读写、TPM PCR 操作、外设写入等测试。首次验收时，除非你已阅读测试清单并接受其副作用，**不要直接执行** `pytest-3 --board kv260`。优先选择只读或明确知道影响范围的单项测试；外设测试失败时，先检查测试所需附件是否真的已经接入。

## 6. 验证 PL（FPGA）确实可用

Linux 能启动只证明 PS、存储、启动链等基本正常，不能完全证明你的 PL 开发流程。完成基础验收后，再做这个**非永久性**测试：

1. 电脑安装 Vivado 2025.1，并用 J4 数据线连接 KV260。
2. 打开 **Hardware Manager → Open Target → Auto Connect**。
3. 确认能发现 K26 对应的 JTAG 设备（器件名应与 `xck26` 对应）。
4. 用一个最简单的、与你的 Vivado 2025.1 工程匹配的临时 bitstream 执行 **Program Device**。
5. 验证设计预期的可观察结果，例如 LED 翻转、UART 打印或 AXI-Lite 寄存器可读写。
6. 断电再上电；临时下载的 PL 配置应消失，这是正常且安全的。**不要选择 Program Flash / Add Configuration Memory Device。**

对本项目，推荐从已保存的官方 HLS 矩阵乘法例程开始：

`../03_official_reference_implementation/04_HLS_Design_Flow_KV260/source/matmult`

它比图像 Sobel 更接近两级 MLP 的全连接 MAC 阵列。先完成“PS 通过 AXI 配置 IP、DMA 送入向量、读回结果”的闭环，再替换为定点 MLP 或 SNN 神经元内核。

## 7. 从验板系统走向 H2-SNN-KV260 部署

建议分成四张清晰的里程碑表，而不是一次性将完整网络搬上板：

1. **板级通过**：第 5A 节全部通过；记录串口日志、固件状态和系统版本。
2. **PL 通路通过**：临时 bitstream 可通过 JTAG 加载；PS↔PL AXI-Lite 与 DMA 往返正确。
3. **算子通过**：在 PL 运行 int8/定点矩阵乘法，逐向量与 Python 固定点参考结果比对。
4. **网络通过**：逐层导入 H2/S1 权重；使用已建立的统一 validation 评测脚本，核对 IoU、Pd、Fa、延迟、资源和功耗。

不要在第 1 阶段引入事件相机、SNN 时序或整网权重；一旦失败，会无法判断是硬件、接口、量化还是算法问题。

## 8. 故障快速分流

| 现象 | 优先排查 |
| --- | --- |
| 上电无风扇、无灯 | 电源是否 12 V/3 A、J12 是否插到底、极性和插头尺寸是否正确 |
| 有风扇但 UART 无输出 | Micro-B 是否为数据线、FTDI 驱动、第二个 COM 口、115200-8N1、microSD 镜像 |
| UART 有输出但无法进入系统 | microSD 重新写卡并验证、检查串口末尾的 MMC/文件系统报错 |
| 能启动但无网络 | J10、网线、路由器 DHCP；执行 `ip -br link` 和 `ip -br addr` |
| 无显示但 UART 正常 | 显示器输入源、HDMI/DP 线；不要据此判断板子损坏 |
| Vivado 找不到 JTAG | 更换 J4 数据线/USB 口、确认 FTDI 驱动、关闭占用串口的终端后重试 |
| 自定义设计不工作 | 先回退到临时 PL bitstream；用 BIST/官方系统判断是硬件还是自定义设计问题 |

## 9. 官方原文入口

- [扫码页：KV260 Linux Boot 入口](https://xilinx.github.io/kria-apps-docs/kv260/2022.1/build/html/docs/linux_boot.html)
- [Ubuntu 24.04 写卡/启动路径](https://xilinx.github.io/kria-apps-docs/kv260/2022.1/linux_boot/ubuntu_24_04/build/html/docs/intro.html)
- [Ubuntu 22.04 写卡与连接教程](https://xilinx.github.io/kria-apps-docs/kv260/2022.1/linux_boot/ubuntu_22_04/build/html/docs/intro.html)
- [Windows UART 参数与首次登录](https://xilinx.github.io/kria-apps-docs/kv260/2022.1/linux_boot/ubuntu_22_04/build/html/docs/uart.html)
- [BIST 1.2 安装和运行](https://xilinx.github.io/kria-apps-docs/kv260/2022.1/bist/1_2/build/html/docs/run.html)
- [UG1089：KV260 用户手册](https://docs.amd.com/r/en-US/ug1089-kv260-starter-kit)
- [DS986：KV260 电源与电气要求](https://docs.amd.com/r/en-US/ds986-kv260-starter-kit/Power-and-Electrical)
- [Kria Boot Firmware Update / Recovery](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/3020685316/Kria%2BSOM%2BBoot%2BFirmware%2BUpdate)

## 10. 验收记录模板

```text
板卡型号/序列号：
板卡载板版本（Rev 1.0 / Rev 2.0）：
电源型号与铭牌：
microSD 镜像及下载日期：
Ubuntu/Kernel 版本：
bootfw_status 输出：
UART 是否通过：
J10 网络是否通过：
HDMI/DP 是否通过：
USB 是否通过：
Vivado JTAG 是否发现 xck26：
临时 PL 下载是否通过：
未测试接口及原因：
异常日志文件位置：
```
