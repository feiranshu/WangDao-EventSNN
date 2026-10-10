# 命令行脚本

- `run_h2_pro.py`：正式 H2-pro exact-int8 单序列推理；输入 EV-UAV NPZ，输出闭合窗口事件概率、预测、候选下标、尾部事件数。
- `rebuild_evaluation_tables.py`：从第 5 板块 H2/H2-pro 源记录重建正式 A/B CSV 和覆盖率。
- `test_evaluation_layout.py`：验证表格顺序、来源和重建确定性。
- `build_delivery_manifest.py`：重建正式交付 SHA-256 清单。

其他脚本用于 H2 基线和历史研究复核。旧 `run_a4_2.py` 已移入 Step7 中间模型目录；禁止用 Test 重新选择最终权重或阈值。

