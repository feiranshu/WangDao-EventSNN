# Part 2：AXI DMA 连接 PS DDR 与 PL 的实施手册

> 当前状态：第一部分的平台准备和第三部分的冻结 SNN AXI-Lite 验证已完成；第二部分尚未实现。  
> 基线：Windows Vivado/Vitis HLS 2025.1，KV260 Generic EDF 2025.1，Linux 6.12.10-xilinx-g0a0f70e531c7。

## 1. 目标和关键结论

第二部分要建立的新数据通路是：

~~~text
PC TCP / microSD
→ PS/Linux
→ DMA 安全缓冲区（PS DDR）
→ AXI DMA MM2S
→ AXI-Stream vadd / SNN
→ AXI DMA S2MM
→ PS DDR
→ PS/Linux 校验或回传
~~~

AXI DMA 是搬运控制器；DDR 访问的连接和仲裁由 PS AXI 端口、SmartConnect 和内存控制器完成。

第三部分使用 AXI-Lite：CPU 逐个写 128 个 32-bit 输入寄存器、启动、轮询，再逐个读 16 个结果。它包含大量离散访问和系统调用。第二部分一次搬运整块连续数据，并让 PL 连续处理流，因此批量吞吐预期高得多。

注意：

- 单个几百字节的小样本可能被 DMA 配置和中断的固定开销抵消，单次延迟必须实测。
- DMA 的主要优势是批处理吞吐、低 CPU 占用和可扩展性。
- 第三部分的冻结 bitstream 只有 AXI-Lite，不能靠软件变成 DMA 版本；迁移 SNN 时必须重新综合带 AXI-Stream 接口的硬件。

## 2. 当前已有和缺失内容

已完成的前置条件：

- xmutil、fpga_manager、PL_FLAT 可用；
- CONFIG_XILINX_DMA=y；
- CMA 约 800 MiB；
- 当前精确内核的外部模块构建链路已验证；
- Vivado/Vitis HLS 2025.1 已安装；
- PC↔KV260 网络与 SSH 可用。

本目录可直接复用：

| 文件 | 用途 |
|---|---|
| kv260_vadd_demo/pc/generate_sd_data.py | 生成 input_a.bin、input_b.bin、golden.bin |
| kv260_vadd_demo/pc/send_tcp.py | TCP 协议和结果校验 |
| kv260_vadd_demo/host/kv260_vadd.cpp | 复用文件/TCP/校验逻辑 |
| kv260_vadd_demo/kernel/krnl_vadd.cpp | 仅参考加法语义 |

不能直接复用为路线 C 构建：

- 旧 host 依赖 XRT；
- 旧 kernel 是 m_axi kernel；
- 旧 Makefile/link.cfg 生成 xclbin；
- 路线 C 需要 Vivado PL_FLAT + AXI DMA + Linux DMAEngine client。

仍需新建：

- Vivado AXI DMA loopback；
- KV260 风扇 EMIO 路径；
- AXI-Stream vadd；
- 匹配的 bit.bin、DTBO、shell.json；
- Linux DMAEngine client；
- 无 XRT 的 kv260_dma_vadd host；
- loopback、SD、TCP 和性能日志。

## 3. 固定第一版参数

| 项目 | 配置 |
|---|---|
| DMA 模式 | Direct Register / Simple Mode |
| Scatter Gather | OFF |
| Micro DMA | OFF |
| MM2S / S2MM | 均启用 |
| Buffer Length Register Width | 26 bit |
| PL 时钟 | 100 MHz |
| 中断 | 两个通道都接 PS GIC |
| DRE | OFF |
| MM2S stream width | 64 bit |
| S2MM stream width | 32 bit（最终 vadd） |
| 输入/输出对齐 | 8 bytes / 4 bytes |
| 初始 packet | 128 KiB，稳定后 1 MiB |
| 缓冲区 | Linux DMA API 管理 |

26-bit 长度上限是 2^26-1，即 67,108,863 bytes。它只是寄存器上限，不代表 CMA 能稳定分配同样大的连续缓冲区；正式 host 必须支持分块。

vadd 数据格式：

~~~text
输入 TDATA[31:0]  = A[i]
输入 TDATA[63:32] = B[i]
最后一个输入 beat：TLAST=1

输出 TDATA[31:0]  = A[i] + B[i]
最后一个输出 beat：TLAST=1

N 个元素：
MM2S bytes = N×8
S2MM bytes = N×4
~~~

## 4. 建议的新工程结构

~~~text
part2/route_c_axi_dma/
├── hw/
│   ├── bd/
│   ├── constraints/
│   ├── scripts/
│   └── reports/
├── hls/axis_vadd/
├── device_tree/
├── driver/
├── host/
├── package/wangdao_dma_vadd/
├── test/
│   ├── loopback/
│   ├── sd/
│   └── tcp/
└── logs/
~~~

保留 BD Tcl、HLS/RTL、XDC、DTS、driver、host、构建脚本、时序报告和 SHA-256。只保存 bitstream 不算完成。

## 5. 阶段 A：板端基线复核

板端执行：

~~~bash
uname -a
cat /etc/os-release
cat /sys/class/fpga_manager/fpga0/state
sudo xmutil listapps

grep -E '^CONFIG_XILINX_DMA=' /boot/config-$(uname -r) 2>/dev/null || true
zcat /proc/config.gz 2>/dev/null | grep -E '^CONFIG_XILINX_DMA=' || true
find /sys/bus/platform/drivers -maxdepth 1 -iname '*dma*' -print

rpm -q kernel-dev kernel-devsrc
cat /usr/src/kernel/include/config/kernel.release
make -s -C /usr/src/kernel kernelrelease

grep -Ei 'CmaTotal|CmaFree' /proc/meminfo
sudo dmesg | grep -i cma | head -n 50

systemctl status fancontrol --no-pager 2>&1 || true
find /sys/class/thermal -maxdepth 2 -type f -name temp -print -exec cat {} \;
~~~

进入 Vivado 前必须确认：

- 内核、kernel-dev、kernel-devsrc 和 kernel.release 匹配；
- CONFIG_XILINX_DMA 可用；
- CMA、xmutil、fpga_manager 正常；
- 已开始保存串口/SSH 日志。

不要安装随机 XRT/ZOCL，也不要无约束升级内核包。

## 6. 阶段 B：先做 DMA loopback

先验证纯数据通路，不接 vadd：

~~~text
DDR
→ MM2S 64-bit
→ AXI4-Stream Data Width Converter 64→32
→ S2MM 32-bit
→ DDR
~~~

### 6.1 Vivado Block Design

加入：

- Zynq UltraScale+ MPSoC；
- AXI DMA；
- SmartConnect；
- Processor System Reset；
- AXI4-Stream Data Width Converter；
- interrupt concat（按 PS IRQ 接口需要）；
- TTC0-Clk2→EMIO→HDA20/A12 风扇路径。

推荐连接：

~~~text
PS M_AXI_HPM*_FPD
→ SmartConnect
→ DMA S_AXI_LITE

DMA M_AXI_MM2S ─┐
                ├→ SmartConnect → PS S_AXI_HP*_FPD → DDR
DMA M_AXI_S2MM ─┘

DMA M_AXIS_MM2S
→ Width Converter 64→32
→ DMA S_AXIS_S2MM

mm2s_introut + s2mm_introut
→ PS pl_ps_irq
~~~

第一版统一 100 MHz，减少跨时钟问题。

AXI DMA 设置：

- Simple Mode；
- SG OFF；
- Micro DMA OFF；
- MM2S/S2MM ON；
- length width=26；
- DRE OFF；
- MM2S=64 bit；
- S2MM=32 bit。

loopback 的 Data Width Converter 只改变 beat 宽度，不改变总字节数：

~~~text
MM2S length = packet_bytes
S2MM capacity >= packet_bytes
~~~

必须先准备 S2MM，再启动 MM2S。

### 6.2 风扇硬闸门

自定义 PL_FLAT 会替换 k26-starter-kits。默认设计保留 Linux 主动风扇控制所需的 TTC0-Clk2→EMIO→HDA20 映射。新设计必须保留该路径或提供已验证的替代方案。

风扇路径未确认前，仅允许短时加载和单次 loopback；禁止连续 100 次、长时压力和大规模性能测试。

### 6.3 构建验收

依次执行：

1. Validate Design；
2. 保存 BD Tcl；
3. 生成 Wrapper；
4. Synthesis；
5. Implementation；
6. 检查 DRC 和时序；
7. 确认 WNS/TNS 无违反；
8. 生成 bitstream；
9. 保存地址、IRQ、clock/reset 记录；
10. 保存 utilization 和 timing_summary。

## 7. 阶段 C：DTBO 和 PL_FLAT 包

DTBO 必须从本次硬件地址和中断生成，不得复制旧 SNN DTBO。

至少描述：

- DMA reg 地址和长度；
- MM2S/S2MM channel；
- 两个中断与 interrupt-parent；
- clock/reset；
- #dma-cells；
- client 的 dmas/dma-names；
- firmware-name。

地址和 IRQ 只能取自 Vivado Address Editor、XSA/硬件导出和当前 PS 映射，不要照抄示例值。

包结构：

~~~text
/lib/firmware/xilinx/wangdao_dma_vadd/
├── wangdao_dma_vadd.bit.bin
├── wangdao_dma_vadd.dtbo
└── shell.json
~~~

bit.bin、DTBO、shell.json 必须是同一构建版本，并记录 SHA-256。

加载：

~~~bash
sudo xmutil unloadapp || true
sudo xmutil loadapp wangdao_dma_vadd
sudo xmutil listapps
cat /sys/class/fpga_manager/fpga0/state

sudo dmesg -T | grep -Ei   'fpga|overlay|xilinx.*dma|dmaengine|irq|iommu|axi' | tail -n 200
~~~

必须满足：

- overlay apply 成功；
- fpga_manager=operating；
- xilinx_dma probe 成功；
- 无 IRQ、clock、reset、IOMMU 或 AXI decode 错误。

## 8. 阶段 D：Linux DMAEngine client

正式方案：

~~~text
kv260_dma_vadd
→ mmap/ioctl
→ kv260_dma_client
→ Linux DMAEngine API
→ xilinx_dma
→ AXI DMA IP
~~~

禁止：

- 把 malloc 指针当 DMA 地址；
- 使用任意 /dev/mem 物理地址作为正式缓冲区；
- 让用户态直接维护 cache 一致性；
- 直接操作 DMA 寄存器作为最终方案。

client 驱动负责：

- 从 DT 获取 MM2S/S2MM channel；
- 分配或映射 DMA 安全 tx/rx buffer；
- 校验长度和对齐；
- 先提交 RX，再提交 TX；
- dma_async_issue_pending；
- 用 completion 等待中断；
- 超时后 terminate；
- 返回实际 rx_bytes 和错误码；
- 在 remove 时释放 channel、buffer 和设备节点。

推荐字符设备接口：

~~~text
/dev/kv260_dma_vadd

mmap tx
mmap rx
ioctl GET_CAPS
ioctl RUN
ioctl GET_STATUS
~~~

GET_CAPS 返回容量、对齐、最大 packet 和版本。RUN 接收 tx_bytes、rx_capacity、timeout_ms 和 request_id。

编译：

~~~bash
make -C /usr/src/kernel M=$PWD modules
modinfo ./kv260_dma_client.ko
sudo insmod kv260_dma_client.ko
dmesg | tail -n 100
ls -l /dev/kv260_dma_vadd
~~~

vermagic 必须对应 uname -r。

卸载顺序：

~~~text
停止 host/TCP server
→ 确认无进行中 DMA
→ 关闭 fd 和 mmap
→ rmmod/unbind client
→ 确认 channel 释放
→ xmutil unloadapp
~~~

## 9. 阶段 E：loopback 验收

依次测试：

- 64 bytes 递增值；
- 4 KiB；
- 128 KiB；
- 1 MiB；
- 全 0；
- 全 FF；
- 固定伪随机数据。

每次：

~~~text
填充 tx
→ 清空 rx
→ 提交 S2MM
→ 提交 MM2S
→ 等待两个 completion
→ 比较实际长度
→ 逐字节比较
→ 保存状态、耗时和 dmesg
~~~

通过条件：

- 全部逐字节一致；
- TLAST 正确结束 packet；
- 两个通道均由中断完成；
- 无 timeout；
- 无 internal/slave/decode error；
- 重载 overlay 后可重复；
- 最后恢复 k26-starter-kits。

loopback 未通过前不要接 vadd。

## 10. 阶段 F：AXI-Stream vadd

HLS 接口骨架：

~~~cpp
#include <ap_axi_sdata.h>
#include <ap_int.h>
#include <hls_stream.h>

using axis_in_t  = ap_axiu<64, 0, 0, 0>;
using axis_out_t = ap_axiu<32, 0, 0, 0>;

extern "C" void axis_vadd(
    hls::stream<axis_in_t>& input,
    hls::stream<axis_out_t>& output) {
#pragma HLS INTERFACE axis port=input
#pragma HLS INTERFACE axis port=output
#pragma HLS INTERFACE ap_ctrl_none port=return

    bool last = false;
    while (!last) {
#pragma HLS PIPELINE II=1
        axis_in_t in = input.read();
        ap_int<32> a = in.data.range(31, 0);
        ap_int<32> b = in.data.range(63, 32);

        axis_out_t out;
        out.data = ap_int<32>(a + b);
        out.keep = -1;
        out.strb = -1;
        out.last = in.last;
        output.write(out);
        last = in.last;
    }
}
~~~

必须验证：

- TVALID/TREADY 背压；
- TKEEP/TSTRB；
- TLAST 只在最后一项；
- II=1；
- 正/负/零/边界值；
- int32 溢出语义；
- C simulation；
- C/RTL co-simulation。

替换 loopback：

~~~text
DMA M_AXIS_MM2S 64-bit
→ axis_vadd
→ DMA S_AXIS_S2MM 32-bit
~~~

此时：

~~~text
MM2S length = N×8
S2MM capacity >= N×4
~~~

重新完成综合、实现、时序、bitstream、DTBO 和打包。不要混用 loopback DTBO 与 vadd bitstream。

## 11. 阶段 G：无 XRT host

新程序建议名为 kv260_dma_vadd。

从旧 host 复用：

- 文件读取/写入；
- TCP 请求与响应；
- verify；
- request_id/status；
- 日志和参数解析。

删除/替换：

- xrt::device；
- xrt::bo；
- xrt::kernel；
- xclbin；
- BO sync。

新流程：

~~~text
读 A/B 或接收 TCP
→ 检查 count
→ 按 64-bit [B|A] 打包 tx
→ ioctl RUN
→ 取得 N×4 bytes 输出
→ 校验/保存/回传
~~~

打包：

~~~cpp
std::uint64_t packed =
    static_cast<std::uint64_t>(static_cast<std::uint32_t>(a[i])) |
    (static_cast<std::uint64_t>(static_cast<std::uint32_t>(b[i])) << 32);
~~~

模式：

~~~bash
sudo ./kv260_dma_vadd loopback --bytes 131072
sudo ./kv260_dma_vadd sd input_a.bin input_b.bin output.bin
sudo ./kv260_dma_vadd server 5000
~~~

分别记录 File read、Pack/Copy、DMA+PL、Result copy、File write、TCP end-to-end。不能把 TCP 或文件总耗时称为 PL 时间。

## 12. 阶段 H：SD 实验

Windows：

~~~powershell
cd "D:\WangDao_EventSNN\2_KV260_bring_up_tests\part2\kv260_vadd_demo"
python .\pc\generate_sd_data.py --count 4096 --out .\data

ssh amd-edf@192.168.100.2 "mkdir -p ~/dma_vadd"
scp .\data\input_a.bin amd-edf@192.168.100.2:~/dma_vadd/
scp .\data\input_b.bin amd-edf@192.168.100.2:~/dma_vadd/
scp .\data\golden.bin amd-edf@192.168.100.2:~/dma_vadd/
~~~

板端：

~~~bash
cd ~/dma_vadd
sudo ./kv260_dma_vadd sd input_a.bin input_b.bin output.bin
cmp output.bin golden.bin
echo $?
~~~

要求错误元素为 0、TEST PASSED、cmp=0，并保存版本、哈希、温度、状态和日志。

## 13. 阶段 I：TCP 实验

板端：

~~~bash
sudo ./kv260_dma_vadd server 5000
~~~

Windows：

~~~powershell
cd "D:\WangDao_EventSNN\2_KV260_bring_up_tests\part2\kv260_vadd_demo"
python .\pc\send_tcp.py --host 192.168.100.2 --port 5000 --count 1024
python .\pc\send_tcp.py --host 192.168.100.2 --port 5000 --count 16384
python .\pc\send_tcp.py --host 192.168.100.2 --port 5000 --count 262144
python .\pc\send_tcp.py --host 192.168.100.2 --port 5000 --count 1048576
~~~

超过已验证 packet 时，host 分块；每个 packet 独立 TLAST，结果按顺序拼接。

风扇和温度验证后再连续运行 100 次。任何一次失败立即停止并保存 dmesg，不做“忽略错误继续”。

## 14. 如何证明 DMA 比第三部分快

第三部分每个 patch 的有效载荷是：

~~~text
输入 128×4 = 512 bytes
输出 16×4 = 64 bytes
合计 576 bytes
~~~

但它通过大量离散 AXI-Lite 操作完成。DMA 则使用 burst、连续流和一次中断，并可以批量多个 patch。

必须实测：

| 指标 | 第三部分 | 第二部分 |
|---|---|---|
| 单 patch 延迟 | AXI-Lite | DMA |
| batch=10/100/336 | 总耗时与平均耗时 | 总耗时与平均耗时 |
| CPU 占用 | 记录 | 记录 |
| 有效带宽 | 记录 | MM2S/S2MM |
| PL 周期 | 可选 | 建议硬件计数器 |
| TCP 端到端 | 不适合高吞吐 | 记录 |

正确结论应是：

- 单个小 patch：DMA 未必占优；
- 多 patch/batch：DMA 吞吐应明显更高；
- 完整数据集/实时流：应使用 DMA/AXI-Stream。

不能在没有数据前写“已经加速”。

## 15. 迁移到实际 SNN

vadd 全部通过后：

1. 取得可重建 SNN 源工程；
2. 固定模型哈希、量化和软件 golden；
3. 增加 AXI-Stream 输入/输出；
4. 定义 packet 与 patch 边界；
5. 定义 TLAST、TKEEP、时间步顺序和 reset；
6. 复用 DMA client/host；
7. 先测 patch0/patch1；
8. 再测 sample_000；
9. 再测 10 个 patch；
10. 再测一个 336-patch window；
11. 最后做数据集和性能。

TLAST 不自动等于 SNN 状态 reset，必须在协议或控制寄存器中明确。

## 16. 常见问题

| 现象 | 检查 |
|---|---|
| loadapp 失败 | bit.bin/DTBO/shell.json 版本、firmware-name、overlay 日志 |
| DMA 不 probe | compatible、reg、clock、reset、IRQ、channel 子节点 |
| client 无 channel | dmas、dma-names、#dma-cells、probe 顺序 |
| MM2S timeout | tx 地址/对齐/长度、HP 连接、clock/reset |
| S2MM timeout | 是否先启动 RX、TLAST、rx capacity |
| 数据错位 | 64-bit [B|A]、端序、对齐 |
| 尾部缺失 | TLAST、TKEEP、S2MM length |
| 小包成功大包失败 | CMA、packet、分块、cache 同步 |
| 第二次失败 | channel/completion/状态未清理 |
| overlay 卸载卡住 | fd、mmap、IRQ、channel 仍占用 |
| 风扇异常 | TTC0-Clk2→EMIO→HDA20/A12、XDC、fancontrol |

## 17. 验收清单

### 环境

- [ ] 内核与开发包匹配
- [ ] CONFIG_XILINX_DMA 可用
- [ ] CMA、xmutil、fpga_manager 正常
- [ ] 保存串口/SSH 日志

### 硬件

- [ ] BD Tcl、XDC、HLS/RTL 可重建
- [ ] Simple Mode，SG/Micro OFF，length=26
- [ ] 中断接入
- [ ] 风扇路径验证
- [ ] 实现和时序通过
- [ ] bit.bin/DTBO/shell.json 同版本

### loopback

- [ ] 64 B、4 KiB、128 KiB、1 MiB 全一致
- [ ] 中断完成，无 timeout/error
- [ ] 重载可重复
- [ ] 安全恢复默认 PL

### vadd

- [ ] HLS C sim 和 co-sim
- [ ] TLAST/TKEEP/TREADY 正确
- [ ] 单次结果全对
- [ ] SD cmp=0
- [ ] TCP 四种规模全对
- [ ] 温控通过后连续 100 次全对

## 18. 最短执行顺序

~~~text
建立 route_c_axi_dma 目录
→ 复核板端基线
→ Vivado DMA loopback
→ 匹配 DTBO/PL_FLAT
→ DMAEngine client
→ 64 B loopback
→ 扩到 1 MiB
→ HLS AXI-Stream vadd
→ 重新综合/实现/打包
→ 无 XRT host
→ SD
→ TCP
→ 温控后的 100 次稳定性
→ AXI-Lite vs DMA 实测
→ 再迁移 SNN
~~~

不要跳过 loopback直接调 vadd；不要跳过 vadd直接调 SNN。

## 19. 官方依据

- AMD AXI DMA PG021，Simple DMA、长度、对齐和中断：  
  https://docs.amd.com/r/en-US/pg021_axi_dma/Direct-Register-Mode-Simple-DMA
- AMD AXI DMA PG021，26-bit length：  
  https://docs.amd.com/r/en-US/pg021_axi_dma/Field-Descriptions
- AMD Vitis HLS UG1399，AXI4-Stream/TLAST/TKEEP：  
  https://docs.amd.com/r/en-US/ug1399-vitis-hls/How-AXI4-Stream-Works
- AMD Kria Wiki，PL 管理和 TTC0-Clk2→EMIO→HDA20 风扇路径：  
  https://xilinx-wiki.atlassian.net/wiki/plugins/viewsource/viewpagesrc.action?pageId=2471788698

## 20. 禁止事项

- 不把 malloc 指针直接当 DMA 地址；
- 不把 /dev/mem 任意物理地址当正式方案；
- 不混用 bit.bin、DTBO、shell.json；
- 不复制其他内核版本的 ko；
- 不在 channel/IRQ/mmap 被占用时卸载 overlay；
- 不在风扇和温度未确认时压力测试；
- 不在 loopback 未通过时调 vadd；
- 不在 vadd 未通过时迁移 SNN；
- 不把 TCP 总耗时写成 PL 计算时间；
- 不把预期加速写成已测性能。

