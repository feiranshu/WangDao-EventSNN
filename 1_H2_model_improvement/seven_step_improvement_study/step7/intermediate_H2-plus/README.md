# H2-plus/A4-2 中间模型

本目录保存 H2-pro 晋级前的 H2-plus/A4-2 正式文件，现仅用于历史复核，不是正式部署入口。

- `weights/`：原 seed42 epoch8 float/int8 权重。
- `final_A4-2/`：原 Step7 精简 Validation/Test 对比及当时的部署决策。该决策后来被 H2-pro 因果实时性版本决策取代，原记录不改写。
- `A_model_accuracy/`、`B_model_complexity/`：从第 5 板块移出的 H2-plus 原正式源记录及 H2 原全序列版本记录。新正式 A/B 表在第 5 板块按同一闭合窗口前缀重建。
- `run_a4_2_legacy.py`：可运行的原单序列历史推理脚本；自动从本目录取 A4-2 权重，并从正式实现取共享 H2 基线包和 Proposal 权重。

H2-pro 的 C0/C1/C2 研究与冻结证据位于 `../../H2-pro/`。原 Step5/Step7 实验编号、门槛和结果仍以各自历史目录为准。
