# H2-pro 正式实现

最终模型 H2-pro-Causal-ANN-v1 使用 H2 Proposal 与 C1 历史 Proposal 运动 Refiner，36→64→32→4，seed42 epoch10，exact-int8，阈值 0.35。以首事件为锚点，按 100 ms watermark 发布完整窗口；未闭合尾部事件不发布。输入仅使用当前与已结束窗口的数据。

```bash
python -m pip install -e .
python software/scripts/run_h2_pro.py path/to/input.npz --output runs/h2_pro_predictions.npz
```

输出含闭合窗口事件概率、预测、候选事件下标、阈值、模型 ID 和未发布尾部事件数。流式 API 为 `wangdao.h2_pro.stream_state.CausalPipeline` 的 `reset`、`push`、`flush_completed_windows`、`finalize`。

正式权重在 `software/weights/`；`results/final_H2-pro/` 是冻结后 Validation/Test 同前缀 A/B 交付。第 5 板块保存正式表。Test 相对 H2-plus 的 IoU 增量置信区间跨 0，HOTA 更低，限制见 [MODEL_CARD.md](MODEL_CARD.md)。H2-plus 旧正式文件归入 `../seven_step_improvement_study/step7/intermediate_H2-plus/`；完整 H2-pro 开发证据在 `../seven_step_improvement_study/H2-pro/`。

