# KV260 单 USB 口连接方案

## 1. 适用条件

- PC 只有一个 USB-A 3.0 接口。
- KV260 已有独立的 12V/3A 电源适配器。
- 现有 Micro-USB 数据线已经可以通过 J4 识别 COM6，因此不需要重新购买串口线或 USB 转 TTL 模块。

## 2. 唯一推荐采购清单

1. **ORICO H4928-U3-V1 四口有源 USB 3.0 HUB**，购买时确认包装内含原装 12V/2A 电源适配器和 USB 3.0 A-to-B 上行数据线。
2. **TP-Link UE300 V3 USB 3.0 千兆以太网适配器**，USB-A 接口，RJ45 为 10/100/1000 Mbps。
3. **UGREEN 绿联 NW102 CAT6 六类成品网线，2 米**。

## 3. 连接拓扑

```text
                             PC
                       唯一USB-A 3.0口
                              │
                    HUB原装USB 3.0上行线
                              │
                              ▼
               ORICO H4928-U3-V1 有源HUB
                  │                       │
                  │                       │
        现有Micro-USB数据线          TP-Link UE300 V3
                  │                       │
                  ▼                       │
          KV260 J4（UART/JTAG）           │
                                          │
                                  绿联NW102六类网线
                                          │
                                          ▼
                                  KV260 J10（千兆网口）

        ORICO HUB ← HUB原装12V/2A适配器
        KV260 J12 ← KV260专用12V/3A适配器
        KV260 J11 ← 已烧录Ubuntu的microSD卡
```

## 4. 接线和上电顺序

1. KV260 暂时不要接电。
2. 将 microSD 卡插入 KV260 J11。
3. 给 ORICO HUB 接上它自己的原装 12V/2A 电源。
4. 用 HUB 原装上行数据线连接 HUB 和 PC 的唯一 USB-A 3.0 接口。
5. 将现有 Micro-USB 数据线连接 HUB 与 KV260 J4。
6. 将 TP-Link UE300 V3 插入 HUB 的另一个 USB 3.0 端口。
7. 用绿联 NW102 网线连接 UE300 的 RJ45 与 KV260 J10。
8. 最后将 KV260 专用 12V/3A 电源接入 J12，启动开发板。

> HUB 的 12V/2A 电源和 KV260 的 12V/3A 电源必须贴标签、分别使用，不能互换。不要通过 J4 或 HUB 给 KV260 整板供电。

## 5. Windows 与 KV260 直连地址

Windows 中将 TP-Link UE300 对应网卡的 IPv4 地址设为：

```text
IP：192.168.100.1
子网掩码：255.255.255.0
默认网关：留空
DNS：留空
```

KV260 串口终端执行：

```bash
sudo ip link set eth0 up
sudo ip addr replace 192.168.100.2/24 dev eth0
ping -c 4 192.168.100.1
```

Windows 执行：

```powershell
ping 192.168.100.2
```

随后使用 SSH：

```bash
ssh ubuntu@192.168.100.2
```

## 6. 验收标准

- Windows 设备管理器中 TP-Link UE300 无黄色感叹号，链路速度显示 1.0 Gbps。
- J4 对应的 FTDI 串口仍然出现，MobaXterm 可以继续打开 COM6。
- PC 与 `192.168.100.2` 能双向 ping 通。
- SSH 可以登录 KV260。
- 同时保持串口和 SSH 连接 30 分钟，不发生 USB 设备反复断开、串口跳号或网卡掉线。

## 7. 使用说明

- COM6 用于查看开机日志、修改网络和故障救援。
- J10 千兆网络用于 SSH、SCP、PC 实时传输测试数据和接收 PL 计算结果。
- PC 与 KV260 直连本身不提供互联网；需要联网下载软件时，再启用 Windows 网络共享或临时把 J10 接入路由器。
- 后续做精确吞吐测试时，不要在测试过程中同时通过 JTAG 下载设计。

