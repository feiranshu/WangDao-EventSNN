# Part 1：板卡与运行环境基线

本目录对应路线 C 第一部分“前期准备”，用于确认 KV260 硬件身份、Boot Firmware、Linux/内核、网络、XRT 状态及外部内核模块构建环境。

安装镜像、Boot Firmware 二进制和参考资料已移到根目录的“references_and_installation”；这里只保存状态证据和构建验证产物。

## 目录说明

- board_state_backup/kv260_check/：原系统/板卡常规检查快照。
- board_state_backup/kv260_preflash_2025_1/：升级或刷写 2025.1 前的状态快照。
- board_state_backup/verified_board_state_2026-09-19/：固定 IP 前网络、CMA 记录和 hello 模块构建结果。

这些文本多为历史快照，不应直接当作当前板卡状态；操作前仍需重新执行只读检查。

## 逐文件索引

| 文件 | 大小 | 作用 |
|---|---:|---|
| `board_state_backup\kv260_check\boardid.txt` | 889 B | 板卡 EEPROM/板卡标识记录，用于确认硬件身份。 |
| `board_state_backup\kv260_check\bootfw_status.txt` | 196 B | 当时 Boot Firmware 状态与版本记录。 |
| `board_state_backup\kv260_check\dri.txt` | 110 B | 当时 DRI/图形相关检查输出。 |
| `board_state_backup\kv260_check\listapps-before.txt` | 366 B | 变更前 xmutil 应用与槽位状态。 |
| `board_state_backup\kv260_check\os-release.txt` | 400 B | 当时 Linux 发行版信息。 |
| `board_state_backup\kv260_check\uname.txt` | 123 B | 当时内核版本与系统架构信息。 |
| `board_state_backup\kv260_check\xbutil-path.txt` | 16 B | xbutil 可执行文件探测结果。 |
| `board_state_backup\kv260_check\xrt-modules.txt` | 0 B | XRT/ZOCL 内核模块检查结果。 |
| `board_state_backup\kv260_check\xrt-packages.txt` | 2.0 KiB | 已安装的 XRT 相关软件包记录。 |
| `board_state_backup\kv260_check\xrt-smi-path.txt` | 17 B | xrt-smi 可执行文件探测结果。 |
| `board_state_backup\kv260_preflash_2025_1\boardid-ignore-errors.txt` | 1.9 KiB | 忽略部分读取错误时取得的板卡标识，作为刷写前补充证据。 |
| `board_state_backup\kv260_preflash_2025_1\boardid.txt` | 889 B | 板卡 EEPROM/板卡标识记录，用于确认硬件身份。 |
| `board_state_backup\kv260_preflash_2025_1\bootfw_status.txt` | 196 B | 当时 Boot Firmware 状态与版本记录。 |
| `board_state_backup\kv260_preflash_2025_1\ip-address.txt` | 193 B | 刷写前网卡地址配置。 |
| `board_state_backup\kv260_preflash_2025_1\ip-route.txt` | 80 B | 刷写前路由表。 |
| `board_state_backup\kv260_preflash_2025_1\listapps.txt` | 366 B | 刷写前 xmutil 应用与槽位状态。 |
| `board_state_backup\kv260_preflash_2025_1\nmcli-connections.txt` | 186 B | 刷写前 NetworkManager 连接配置。 |
| `board_state_backup\kv260_preflash_2025_1\os-release.txt` | 400 B | 当时 Linux 发行版信息。 |
| `board_state_backup\kv260_preflash_2025_1\uname.txt` | 123 B | 当时内核版本与系统架构信息。 |
| `board_state_backup\kv260_preflash_2025_1\xrt-packages.txt` | 836 B | 已安装的 XRT 相关软件包记录。 |
| `board_state_backup\verified_board_state_2026-09-19\eth0_before_static.txt` | 262 B | 设置固定 IP 前的 eth0 状态。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\..module-common.o.cmd` | 41.2 KiB | 内核构建系统生成的命令与依赖跟踪文件，用于复现和增量构建诊断。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\.hello.ko.cmd` | 319 B | 内核构建系统生成的命令与依赖跟踪文件，用于复现和增量构建诊断。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\.hello.mod.cmd` | 174 B | 内核构建系统生成的命令与依赖跟踪文件，用于复现和增量构建诊断。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\.hello.mod.o.cmd` | 40.9 KiB | 内核构建系统生成的命令与依赖跟踪文件，用于复现和增量构建诊断。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\.hello.o.cmd` | 40.8 KiB | 内核构建系统生成的命令与依赖跟踪文件，用于复现和增量构建诊断。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\.module-common.o` | 1.5 KiB | 内核模块编译产生的目标文件，保留作构建证据。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\.Module.symvers.cmd` | 198 B | 内核构建系统生成的命令与依赖跟踪文件，用于复现和增量构建诊断。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\.modules.order.cmd` | 146 B | 内核构建系统生成的命令与依赖跟踪文件，用于复现和增量构建诊断。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\hello.c` | 404 B | 最小 Linux 内核模块源码，用于验证外部模块编译、加载和卸载链路。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\hello.ko` | 5.4 KiB | 已编译的 hello 测试内核模块，是构建链路通过的实物证据。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\hello.mod` | 35 B | 内核模块构建生成的模块目标清单。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\hello.mod.c` | 373 B | 内核构建系统生成的模块元数据 C 文件。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\hello.mod.o` | 2.7 KiB | 内核模块编译产生的目标文件，保留作构建证据。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\hello.o` | 2.9 KiB | 内核模块编译产生的目标文件，保留作构建证据。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\Makefile` | 152 B | 构建 hello.ko 的内核模块 Makefile。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\Module.symvers` | 0 B | 内核模块构建生成的符号版本表。 |
| `board_state_backup\verified_board_state_2026-09-19\hello_module\modules.order` | 35 B | 本次构建生成的模块顺序记录。 |
| `board_state_backup\verified_board_state_2026-09-19\route_before_static.txt` | 0 B | 设置固定 IP 前的路由快照；空文件表示当时未捕获到路由条目。 |
| `board_state_backup\verified_board_state_2026-09-19\route_c_cma_boot.log` | 363 B | EDF 启动时 CMA 连续内存预留日志，是 DMA 缓冲区判断依据。 |

