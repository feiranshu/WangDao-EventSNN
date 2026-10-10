# H2-pro 安装、推理和复核

在 `implementation` 目录运行：

```bash
python -m pip install -e .
python software/scripts/run_h2_pro.py path/to/input.npz --output runs/h2_pro_predictions.npz
python software/scripts/test_evaluation_layout.py
python software/scripts/build_delivery_manifest.py
```

输入是包含 `ev` 结构化 x/y/t/p 数组的 EV-UAV NPZ，时间戳单位 ms，按时间升序。输出只覆盖完整的 100 ms 窗口，尾部事件数记在 `trailing_events`。脚本拒绝覆盖已有输出。

正式权重哈希和完整结果分别在 `software/weights/README.md`、`results/final_H2-pro/`；完整 C0/C1/C2 训练及服务器材料见 `../seven_step_improvement_study/H2-pro/`。本地匹配训练已经完成 10 轮，正式部署无需重新训练。数据集不发布，冻结索引在 `data/index.json`。

