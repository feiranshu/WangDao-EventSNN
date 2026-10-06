# LIF 算术与接口冻结规范 v2.0

## 定点路径

- 激活和普通权重：有符号 INT8、二进制补码；
- 原有 MAC/PoT PE 的膜电位和突触和：默认有符号 INT20；
- 三精度对照中的 INT8 神经元：膜电位、输入电流和阈值均为有符号 INT8；
- 所有定点溢出均饱和，不允许回绕。

INT8 对照神经元参数固定为 `LEAK_SHIFT=2`、`APPROX_LSB_BITS=2`、硬复位到 0。精确更新为：

```text
leak      = membrane >>> 2
decayed   = membrane - leak
candidate = saturate_int8(decayed + current)
spike     = candidate >= threshold
membrane  = spike ? 0 : candidate
```

LOA 近似加法为：

```text
low  = a[K-1:0] OR b[K-1:0]
high = (a >>> K) + (b >>> K)
raw  = (high <<< K) OR low
sum  = saturate(raw)
```

低位不向高位产生进位。饱和前有 `exact-approx = low_a AND low_b`，所以 `K=2` 时误差绝对值不超过 3 LSB。

## 浮点路径

支持：

- FP32：IEEE-754 binary32，1/8/23；
- FP16：IEEE-754 binary16，1/5/10。

离散方程固定为 `E_rest=0`：

```text
product   = round(alpha * membrane)
candidate = round(product + current)
spike     = candidate >= threshold
membrane  = spike ? +0.0 : candidate
```

精确模式中的乘法、加法和比较均由 Xilinx Floating-Point Operator v7.1 完成。近似模式仅用 Mitchell 乘法器替换乘法，精确加法和阈值比较保持不变。

Mitchell 正规数乘法：

```text
fraction_sum = fraction_a + fraction_b
exp_out      = exp_a + exp_b - bias + carry(fraction_sum)
frac_out     = low_fraction_bits(fraction_sum)
```

它等价于 `log2(1+f)≈f`。理论最大正规数乘法相对误差为 11.11%。subnormal 输入/输出执行 FTZ；NaN、Inf、零、上溢和下溢按 `fp_mitchell_multiplier.sv` 中的显式分支处理。

## 浮点流接口

- `input_valid_i && input_ready_o` 接收一次更新；
- `output_valid_o && output_ready_i` 完成一次更新；
- 当前实现仅允许一个事务在途，`busy_o=1` 时不接受新输入；
- `soft_reset_i` 取消在途事务、清零膜电位和脉冲；
- `membrane_o` 与 `spike_o` 在 `output_valid_o=1` 时有效，并保持到输出握手；
- 输入和输出浮点数据均直接传输 IEEE 位模式，不做整数转换。

## 综合对照约束

六个顶层必须使用同一器件 `xczu3eg-sfvc784-1-i`、同一 150 MHz 时钟约束、同一 Vivado 2025.1 版本和同一 OOC 实现流程。FP 精确/近似对照必须共享相同配置的加法和比较 IP，不能只在其中一侧启用额外流水级。

综合报告至少包含 LUT、FF、DSP48、BRAM、WNS、估算 Fmax 和功耗。所有结果由实际报告生成，不允许用器件手册或经验值代替。
