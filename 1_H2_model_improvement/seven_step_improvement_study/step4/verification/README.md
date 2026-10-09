# Step4本地工程验证

验证日期：2026-10-04。本目录只保留精简验证结论，不保留可再生成的候选缓存、smoke checkpoint或指标副本。

## 已通过

- `python -m unittest discover -s tests -v`：10/10通过。
- 本地真实数据preflight：通过；H2 SHA-256为`21019377617e429a599497a0a6b0f541d2c05bf7bc8fee83005deae72bb578b2`，索引SHA-256为`7968011347d9d6291a94e183b4268a84549301003523bca7b4b185a5ca36a397`；见`preflight_local.json`。
- CPU smoke：`status=smoke_complete`，依次完成A4-1、A4-1-null、A4-2、A4-2-null、A4-3、A4-3-control的加载、训练、逐轮float/int8评测、checkpoint保存/重载、指标导出和条件分支判断。
- A4-3 smoke没有公平工作点时，进度正确显示`iou=NA high_r=NA`，未复现Step3的`float(None)`崩溃。
- A4-2定向smoke确认历史可用性分组`0/1/2`、Train-main分组计数、全事件`bootstrap_event_scores.npz`以及训练/float评测/int8评测/checkpoint分项耗时均实际生成。
- 首批smoke不满足严格成对入口，A4-4/A4-5按协议跳过，说明条件编排生效。
- `run-all --resume`能够复用六个已完成smoke模型并生成最终`step4_decision.json`，不会重复训练。
- 最终`H2_step4_server_20261004.zip`独立解压后再次运行10项测试全部通过；ZIP内文件权限、清单和内容哈希通过构建器校验。最终SHA-256为`3ea9d158b2cc52459c8b06163b869b43317f48683e6711ad4a0a9a180a55cbf7`。

本地环境为CPU版PyTorch，以上smoke只证明工程链路可执行。此后Step4四卡CUDA正式结果已经回传，Step5已冻结A4-2 epoch 8/阈值0.4，Step6完成多seed复现与历史归因，Step7完成Validation/Test独立评测；最终按整体IoU、Precision和误报控制优先接受A4-2。
