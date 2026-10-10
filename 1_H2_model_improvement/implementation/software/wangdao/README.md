# wangdao Python 包

正式 H2-pro 流式推理入口是 `h2_pro/stream_state.py::CausalPipeline`，其特征、运动估计与协议在 `h2_pro/`。它调用冻结的 `h2/` Proposal 内核及 36→64→32→4 exact-int8 Refiner。按序列重置，并由 watermark 关闭 100 ms 窗口。

`a4_2_runtime.py` 是 H2-plus/A4-2 历史兼容模块，已退出正式推理路径。其余 `h2_research.py`、`causal.py`、学生模型、训练和指标模块用于历史复现及研究。

