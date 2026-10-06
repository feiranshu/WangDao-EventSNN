# Vivado 浮点 IP 输出目录

运行 `scripts/create_fp_ip.ps1` 后，这里应出现六个 Xilinx Floating-Point
Operator v7.1 IP：FP32/FP16 的乘法、加法和大于等于比较器。

生成文件体积较大，默认不纳入源码审查。综合前至少确认以下文件存在：

- `fp32_mul_exact/fp32_mul_exact.xci`
- `fp32_add_exact/fp32_add_exact.xci`
- `fp32_cmp_ge_exact/fp32_cmp_ge_exact.xci`
- `fp16_mul_exact/fp16_mul_exact.xci`
- `fp16_add_exact/fp16_add_exact.xci`
- `fp16_cmp_ge_exact/fp16_cmp_ge_exact.xci`

近似 FP 顶层仍需要精确加法和比较 IP；它只移除了精确尾数乘法器。
