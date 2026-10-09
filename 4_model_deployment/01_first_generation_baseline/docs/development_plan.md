# 第一代 SNN 模型 KV260 部署计划

## 1. 冻结输入

部署对象由第 3 部分提供：

- S1 exact Proposal-SNN；
- S3 LOA approximate Proposal-SNN；
- T=4、int8 权重、24 位累加器、24 位 Q12 膜电位；
- 冻结 Proposal/事件阈值、A1-H2 PS Refiner 和数据索引。

板级调试不得反向改变模型、阈值、位宽、复位或近似规则。需要模型变化时回到第 3 部分建立新版本和新黄金向量。

## 2. 已完成

- 24 位 exact/LOA 加法器、LIF、三层 Proposal 引擎及 AXI-Lite 包装。
- 第一层跳零、背压、复位、连续多窗口和 Python 主机联调。
- Vivado 2025.2.1 下单元、三层 Proposal 与 AXI-Lite 行为回归。
- exact 固定调度版本综合、实现、时序收敛和 bitstream/XSA 导出。
- PL_FLAT 应用包、设备树覆盖、单 patch 及板端批量驱动。
- PS 固定点 H2 Refiner、自包含完整运行脚本和逐字验证脚本。
- KV260 上两个冻结 patch 的 WANGDAO_PASS。

正式证据位于`verification_evidence/`；拆分前的完整技术记录保存在根级本地`../../../archive/4_model_deployment/01_first_generation_baseline/legacy_docs/pre_split_complete_project_docs/`。

## 3. 待完成顺序

1. 在真实 KV260 上运行全部 HW-calibration 序列并保存 capture、日志和逐字差异。
2. 核对 Python、RTL 和 FPGA 输出；任何差异先修接口或实现，不能修改模型掩盖。
3. 完成冻结 exact 流程的 Validation 板测。
4. 在相同工程、接口、时钟、约束和并行度下构建 S3 approximate 版本。
5. 对 exact/approx 分别记录 LUT、FF、BRAM、DSP、WNS/Fmax、延迟、吞吐和功耗。
6. 全部配置冻结后再生成 Test 包并进行最终一次验收。

## 4. 验收边界

- 软件黄金模型、RTL 仿真、Vivado 综合估算和 KV260 实测分别报告。
- 单 patch WANGDAO_PASS 不能写成完整数据集通过。
- Vivado 功耗估算不能写成板级功耗。
- 软件操作计数不能换算为能耗。
- Test 不用于调参，首次结果必须保留。
