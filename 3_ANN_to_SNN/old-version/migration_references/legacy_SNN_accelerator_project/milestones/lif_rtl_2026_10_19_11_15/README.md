# FP32 / FP16 / INT8 精确与近似 LIF RTL 对比工程

本目录独立对应项目时间线中的 **2026-10-19—2026-11-15** 阶段，目标是完成可复现的 LIF 算术单元、阈值与复位逻辑，并为用户在 Vivado 2025.1 中自行综合提供六个公平对照顶层。

目前结论是：原工程已经有准确的定点 LIF 和 LOA 近似定点 LIF；本次又补齐了 FP32、FP16、INT8 三种格式的精确/近似对照。Python 功能测试和不依赖浮点 IP 的 XSIM 测试已经通过，**本 README 不包含虚构的综合数据，LUT、FF、DSP、Fmax 和功耗必须由实际 Vivado 报告填写。**

## 1. 六个综合顶层

| 数值格式 | 精确顶层 | 近似顶层 | 近似位置 |
|---|---|---|---|
| FP32 | `fp32_lif_exact_top` | `fp32_lif_approx_top` | 用 Mitchell 对数域近似乘法替换精确尾数乘法；加法和阈值比较保持精确 |
| FP16 | `fp16_lif_exact_top` | `fp16_lif_approx_top` | 与 FP32 相同，格式改为 IEEE-754 binary16 |
| INT8 | `int8_lif_exact_top` | `int8_lif_approx_top` | 更新加法的低 2 位采用 LOA，高位、阈值和饱和判断保持精确 |

浮点顶层使用以下离散 LIF 方程，并固定 `E_rest=0`、发放后硬复位到 `+0.0`：

```text
candidate = round(round(alpha × membrane) + current)
spike     = candidate >= threshold
membrane  = spike ? +0.0 : candidate
```

浮点精确模式使用 Xilinx Floating-Point Operator v7.1 的乘法、加法和 `>=` 比较 IP。浮点近似模式只把乘法替换为 `fp_mitchell_multiplier.sv`，因此精确/近似对照共享相同的加法、比较、控制和复位逻辑。

INT8 使用：

```text
leak      = membrane >>> 2
decayed   = membrane - leak
candidate = sat8(decayed + current)
```

近似 INT8 将 `candidate` 的加法替换为 `K=2` 的 LOA：低位按位 OR，低位不向高位产生进位。饱和前误差绝对值最多为 `2^K-1=3 LSB`，且本定义的近似结果不会高于精确结果。

## 2. 近似乘法的定义

对正规浮点数：

```text
A = (-1)^sA × 2^eA × (1 + fA)
B = (-1)^sB × 2^eB × (1 + fB)
log2(1 + f) ≈ f
```

RTL 只累加指数和尾数小数部分，不例化尾数乘法器。对两个有限正规数，Mitchell 乘法的理论最大单次相对误差为 `1/9≈11.11%`，在 `1.5×1.5` 处表现为精确值 `2.25`、近似值 `2.0`。2 的整数次幂相乘保持精确。

特殊值策略：

- FP16/FP32 的 subnormal 输入和结果均 FTZ（flush-to-zero）；
- `0×Inf` 输出规范 quiet NaN；
- 有限非零数乘 Inf 输出带正确符号的 Inf；
- 指数上溢输出 Inf，下溢输出带符号零。

## 3. 已完成的功能对比

下表来自固定随机种子 `24307140011` 的位级模型，不是综合结果。单步统计使用 50,000 个样本；长序列统计使用 256 条、每条 200 步的递归轨迹。完整原始数据位于 `reports/precision_compare/functional_metrics.json`。

| 格式 | 单步 MAE | 归一化 MAE | 平均相对误差 | P95 绝对误差 | 单步脉冲不一致 | 长序列脉冲不一致 | 长序列精确/近似脉冲数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP32 | 0.01449 | 1.812% | 3.085% | 0.04606 | 1.866% | 2.375% | 1073 / 167 |
| FP16 | 0.01455 | 1.819% | 3.090% | 0.04639 | 1.802% | 2.275% | 1052 / 137 |
| INT8 | 0.74454 LSB | 0.586% | 3.841% | 3 LSB | 0.394% | 10.063% | 8602 / 8188 |

这些结果应这样解释：

- FP32 与 FP16 的近似误差非常接近，说明当前误差主要由 Mitchell 乘法而不是浮点尾数位宽决定；FP16 的主要价值预计是硬件资源和带宽，而不是降低近似误差。
- INT8 单步误差有严格的 3 LSB 上界，单步脉冲判决最稳定；但递归膜电位会把微小误差累积为脉冲相位偏移，所以长序列不一致率不能用单步结果替代。
- 两个浮点近似模型的长序列脉冲数显著偏少，说明原始 Mitchell 乘法存在系统性低估。它适合用作 QAT/KD 的硬件误差模型，**不适合不经重训练直接替换教师模型。**
- 最终整网应重新搜索 `alpha`、阈值和每层近似开关，并以检测 mAP 而不是单神经元 MAE 作最终判断。

## 4. 硬件性能表：等待用户综合

以下六行必须由相同器件、相同时钟约束和相同 OOC 流程生成。当前状态故意保留为“待综合”。

| 顶层 | LUT | FF | DSP48 | BRAM | WNS | 估算 Fmax | 状态 |
|---|---:|---:|---:|---:|---:|---:|---|
| `fp32_lif_exact_top` | — | — | — | — | — | — | 待用户综合 |
| `fp32_lif_approx_top` | — | — | — | — | — | — | 待用户综合 |
| `fp16_lif_exact_top` | — | — | — | — | — | — | 待用户综合 |
| `fp16_lif_approx_top` | — | — | — | — | — | — | 待用户综合 |
| `int8_lif_exact_top` | — | — | — | — | — | — | 待用户综合 |
| `int8_lif_approx_top` | — | — | — | — | — | — | 待用户综合 |

综合前的合理预期仅用于检查结果是否异常，不能当作结论：

- FP32 精确乘法通常最昂贵，FP16 次之；
- Mitchell 模块移除了尾数乘法，但浮点精确加法和比较 IP 仍可能使用 DSP，因此不能在综合前宣称整个近似神经元为 0 DSP；
- INT8 两个顶层只有神经元更新，不包括 `int8_lif_baseline_top` 中的 `INT8×INT8 MAC`；
- LOA 可能缩短低位进位链，但节省比例和 Fmax 变化必须以 post-route 报告为准。

## 5. 用户综合步骤

请在 PowerShell 中执行：

```powershell
cd D:\WangDao_EventSNN\milestones\lif_rtl_2026_10_19_11_15

# 1. 生成功能完全相同的 FP32/FP16 精确加法、乘法和比较 IP
.\scripts\create_fp_ip.ps1

# 2. 分别综合并实现六个顶层
.\scripts\run_precision_synthesis.ps1
```

默认器件为 `xczu3eg-sfvc784-1-i`，时钟约束为 150 MHz。脚本会把工程暂存到纯英文临时路径，避免 Vivado 对中文路径的兼容问题。

完成后重点检查：

- `reports/precision_compare/<run>/metrics.json`
- `reports/precision_compare/<run>/utilization_route.rpt`
- `reports/precision_compare/<run>/timing_route.rpt`
- `reports/precision_compare/<run>/power_route.rpt`
- `reports/precision_compare/hardware_metrics.csv`
- `reports/precision_compare/hardware_comparison.json`

浮点控制器为了让六个顶层易于复现，目前限制为“同一神经元只允许一个事务在途”。这适合单元资源/关键路径比较，但吞吐率不代表未来 PE 阵列；阵列版本需要把 IP 改为全流水并允许每周期接收一次更新。

## 6. 验证状态

- Python：20 项测试全部通过；
- XSIM：6 个 testbench 全部输出 `PASS`；
- FP16/FP32 Mitchell RTL 已与 Python 模型逐项比对，各通过 2,052 组位模式黄金向量（合计 4,104 组），覆盖正规数、2 的幂、符号、NaN、Inf、溢出和 FTZ；
- FP 精确/近似完整顶层依赖用户生成的 Xilinx IP，最终 IP 时序仿真和综合由用户执行；
- 本轮没有运行任何综合，也没有操控 Vivado GUI。

## 7. 旧进度汇报 PPT 中可直接复用的内容

根目录 `SNN_progress_report.pptx` 中以下内容适合保留并更新：

- 第 2 页的总体“事件相机—SNN—软硬件协同”路线可作为项目总览，但需替换为目前的 PS/PL 分工和三精度对照；
- 第 8 页的 `mAP@0.5=0.446` 曲线可保留为历史基线，标题必须注明“10 Epoch/旧配置”，不能与后续稳定教师模型混用；
- 第 9 页的近似 LIF 公式和硬件结构与本次实现高度一致，建议改成“精确乘法 IP vs Mitchell 乘法器”的 FP32/FP16/INT8 对比页；
- 第 3 页红字提出的量化、速度、功耗以及 FP32/INT8 差异问题，现在正好可以由本工程的六组综合报告回答；
- 第 10—13 页的多级 LIF 和 Spiking Transformer 暂不属于当前主论文关键路径，除非后续消融证明明显收益，否则建议移到备选方案或附录。

## 8. 目录结构

- `rtl/`：定点 LIF、Mitchell 乘法器、浮点控制器和六个对照顶层；
- `model/`：位级模型、单元测试和三精度误差统计；
- `tb/`：XSIM 自检 testbench；
- `vivado/`：浮点 IP 生成与六顶层 OOC 实现 Tcl；
- `scripts/`：功能验证、IP 生成、综合和报告汇总入口；
- `ip/`：用户运行脚本后生成的 Xilinx Floating-Point Operator IP；
- `reports/`：功能统计、XSIM 日志和用户综合报告；
- `docs/`：冻结的算术、接口与复现约束。
