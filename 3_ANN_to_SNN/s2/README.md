# S2 统一参考骨架

本目录独立保存第二代 S2 的协议、实现、测试、训练结果与决策卡。入口以 [`PROTOCOL.md`](PROTOCOL.md) 为准。冻结 H2-pro Proposal 及 C1 候选缓存来自上游工程，本目录只读取并核对来源，不复制或修改冻结权重。

**S2 健康检查已完成**：结果与局限见 [`REPORT.md`](REPORT.md)，阶段决定见 [`DECISION.md`](DECISION.md)。正式完整复训保存在 [`runs/GEN2-S2-001-revision1/`](runs/GEN2-S2-001-revision1/)；初版停止运行单独保留。研究 ANN 在预设 Train-dev 网格选择阈值 0.55、IoU 0.7029；研究 SNN 选择上界 0.60、IoU 0.6339。0.65–0.95 的扩展阈值仅为事后诊断。

- [`core.py`](core.py)：36→48→32→4 研究 ANN、同宽固定位置 LIF SNN、真实时间衰减、四原型轨迹/残余膜电位读出及 Refiner 成本账目。
- [`train.py`](train.py)：Train-main 全序列训练和 Train-dev 同闭合前缀评价；默认 seed42、5 epoch、完整 79/10 序列。
- [`stream.py`](stream.py)：原始事件接口 `reset / push / flush_completed_windows / finalize`，复用冻结 Proposal，完成窗口闭合后才回放研究 Refiner。
- [`COST_PROTOCOL.md`](COST_PROTOCOL.md)：规定 Refiner/Proposal、MAC/AC、状态字节及运行时间的统计边界。
- [`diagnose.py`](diagnose.py)：训练后对冻结 checkpoint 做 Train-dev 数值和成本重放检查。
- [`analyze.py`](analyze.py)：在 Train-dev 上按已记录规则选择研究阈值，并重放 S1 的目标子集；离线标签与身份仅用于评估。
- [`health_revision1.json`](health_revision1.json)、[`analysis_extended_diagnostic.json`](analysis_extended_diagnostic.json)：末轮独立重放和阈值/子集数据。
- `test_core.py`、`test_stream.py`：状态、发放、梯度、分块一致性、边缘、空窗口和尾部规则。

从项目根目录复核 S2 测试：

```powershell
& 'D:\software\anaconda\envs\evuav\python.exe' -m unittest discover -s '3_ANN_to_SNN\s2' -p 'test_*.py' -v
```

初版完整训练在 SNN 第一轮出现数值健康问题，已停止并保留在 `runs/GEN2-S2-001/`；完整复训采用 [`PROTOCOL.md`](PROTOCOL.md) 记录的初始化修订。训练运行目录为追加式，已有目录不会覆盖；新实验先在 S0 登记新编号。`--limit-train-sequences` 和 `--limit-dev-sequences` 仅供运行先导检查，其输出标记为 `pilot_not_stage_result`，不能代替完整 S2。评价与成本来自缓存后的 Refiner 回放；其 CPU 时间不包含冻结 Proposal 的在线特征构建和窗口采集等待。
