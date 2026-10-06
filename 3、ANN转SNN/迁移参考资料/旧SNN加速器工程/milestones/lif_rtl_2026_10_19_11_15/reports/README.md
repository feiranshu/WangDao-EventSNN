# Reports

脚本会在以下目录生成结果：

- `reports/python/`：位精确模型测试日志；
- `reports/xsim/`：XSIM 编译和仿真日志；
- `reports/exact_int8/`：精确 INT8 基线综合、时序、利用率和功耗报告；
- `reports/pot_approx/`：PoT/近似方案对应报告；
- `reports/precision_compare/functional_metrics.json`：FP32、FP16、INT8 功能误差与脉冲差异；
- `reports/precision_compare/<run>/`：用户运行六组对照综合后生成的 OOC 报告；
- `reports/precision_compare/hardware_metrics.csv`：六组硬件指标汇总；
- `reports/summary.json`：自动提取的关键指标。

生成文件应与 Vivado 版本、Git commit 和 Tcl 脚本一起归档。没有生成的报告不得以预测值代替。
