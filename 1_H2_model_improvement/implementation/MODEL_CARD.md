# H2-pro 模型卡

- 身份：H2-pro-Causal-ANN-v1；H2 Proposal 128→48→24→16，C1 Refiner 36→64→32→4。
- 特征：第 2 列由全序列相对时间替换为已闭合窗口 Proposal 热图运动；细节见 `software/wangdao/h2_pro/protocol.json`。状态按序列重置。
- 冻结：seed42、epoch10、阈值 0.35、uint8/int8/int32 exact-int8 软件路径。
- 输入/输出：时间升序 x/y/t/p 事件；只发布 100 ms 闭合窗口的概率和预测。
- 权重：`software/weights/H2_D1_seed42.npz` 与 `software/weights/H2-pro-Causal-ANN-v1_int8.npz`；float teacher 仅用于审计/转换。
- 入口：`software/scripts/run_h2_pro.py`；结果：`results/final_H2-pro/`。

Test IoU 87.1383%，对 H2-plus 增量 0.2737 个百分点，95% 配对 bootstrap 区间跨 0；HOTA 42.1332%，低于 H2-plus 的 43.1441%。高密度召回不是全面领先。H2/H2-plus 对照采用旧全序列预测裁剪，仍含非因果时间输入，不能当作同等流式模型性能证明。目前没有 KV260 板端速度、延迟、功耗实测。

模型和阈值由 Train-dev 选择；Validation/Test 在冻结后读取，不用于重选。原 Step5/Step7 历史结论保存在七步研究目录。
`../seven_step_improvement_study/H2-pro/frozen/` 中的冻结清单记录的是开封 Validation/Test 之前的状态，故其 `acceptance_status` 仍为 `not_yet_evaluated`；现行版本决定以 `results/final_H2-pro/deployment_decision.json` 为准，不改写原冻结证据。

