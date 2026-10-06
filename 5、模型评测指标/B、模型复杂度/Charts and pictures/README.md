# B 类正式展示

- `H2_vs_H2-plus_模型复杂度对比.xlsx`：总览汇总 A01—A06，逐序列工作表可核对 96 行明细（两模型 × Validation/Test 各 24 条）。
- `EV-UAV_vs_H2_vs_H2-plus_Validation_模型复杂度.png` 与 `EV-UAV_vs_H2_vs_H2-plus_Test_模型复杂度.png`：延续原 2×2 柱状图，依次显示 A01 参数量、A02 静态载荷、A03 核心张量预算上限和 A04 平均每序列稠密 MAC。EV-SpSegNet 只有 A01 使用论文报告的 4.0M；A02—A04 原论文未报告，因此图中保留缺失标记。

注意：A01 采用对数坐标，使 4.0M 与约 12K 参数能同时看清。A03 是核心张量体积预算，不是同时存活峰值或上板 SRAM。完整指标口径见上一级 `README.md`。
