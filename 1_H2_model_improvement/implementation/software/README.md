# 软件目录

正式推理链为 `scripts/run_h2_pro.py` → `wangdao/h2_pro/stream_state.py` → `wangdao/h2/`，使用 `weights/H2_D1_seed42.npz` 和 `weights/H2-pro-Causal-ANN-v1_int8.npz`。运行窗口固定 100 ms，阈值 0.35，发布闭合窗口事件结果。完整训练和冻结证据在 `../../seven_step_improvement_study/H2-pro/`。

`wangdao/a4_2_runtime.py` 与其他旧研究模块只用于历史复核，H2-plus 旧权重和旧正式脚本已移入 Step7 中间模型目录。

