# 事件数据驱动 SNN 加速器

本仓库把《个人创新设计报告》中的方案落实为可测试的软硬件协同工程。首版重点是冻结数据、量化与 PS–PL 接口，并提供与 RTL 共用的位精确黄金模型。

## 当前已实现

- EV-UAV 风格事件表读取、固定时间窗和有界恒事件数切片；
- 与标签无关的切片边界、按目标 ID 生成边界框；
- 64 位 AER 包编解码；
- signed power-of-two 权重量化、精确/近似饱和加法和 LIF 黄金模型；
- 精确/近似自动切换的滞回控制器；
- 模型 manifest、权重编码和黄金测试向量导出；
- 双模式加法器、LIF、AER FIFO/解析器、状态 cache 和 3×3 预取地址生成 RTL；
- PS 侧寄存器及 AER 包 C 接口；
- 软件单元测试和 RTL/Vivado 入口脚本。

完整 EMS-YOLO 训练、EV-UAV 数据、Vivado 实现结果和板级功耗数据依赖外部资源，不伪造在仓库中。它们按 [docs/upstream.md](docs/upstream.md) 接入。

## 快速验证

```powershell
$python = "C:\Users\sfr08\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
& $python -m unittest discover -s software/tests -v
& $python scripts/export_example.py --output results/generated/example
```

在接入训练集后，先从固定窗口统计动态切片候选值：

```powershell
& $python scripts/calibrate_slicing.py data/EV-UAV/train --timestamp-unit us
```

如已安装 Icarus Verilog：

```powershell
./scripts/run_rtl_tests.ps1
```

## 目录

- `software/snn_accel`：数据、AER、量化、位精确模型及导出协议；
- `hardware/rtl`：可综合 SystemVerilog 基础模块；
- `hardware/tb`：RTL 自检 testbench；
- `firmware`：PS 驱动公共接口；
- `configs`：冻结的软硬件配置和实验矩阵；
- `docs`：架构、实验协议、上游接入和路线图；
- `results`：只保存小型汇总与说明，原始大文件默认忽略。
