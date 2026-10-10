# 测试目录

`test_h2_pro_runtime.py` 检查正式 exact-int8 权重哈希、协议加载和 100 ms 闭合窗口发布；`test_core.py`、`test_h2_research.py`、`test_h2_release.py`、`test_h2_students.py`、`test_causal.py`、`test_phase1_common.py`、`test_phase1_release.py` 用于保留的 H2 基线与历史研究回归。A4-2 专项旧测试已移入 Step7 中间模型材料。

正式专项验收：`python -m unittest software/tests/test_h2_pro_runtime.py -v`。完整回归：`python -m unittest discover -s software/tests -v`。

