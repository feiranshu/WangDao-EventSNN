# A+B 完整性核查（2026-10-08）

> 历史审计：本文件仅反映 H2-plus 尚在正式注册表时的状态。现行 H2/H2-pro 完整性见 `COMPLETENESS_AUDIT_20261010.md`。

## H2 与 H2-plus

逐一检查 A 的 44 个 Validation/Test 指标 CSV 和 B 的 10 个 Common/Validation/Test 指标 CSV：

- `h2__software_exact_int8`：54/54 行均存在、`status=complete`、`value` 非空。
- `h2_plus__software_exact_int8`：54/54 行均存在、`status=complete`、`value` 非空。

两者无需补测。

## EV-SpSegNet

EV-SpSegNet 已改为 `formal` 软件模型。论文表 2可直接录入：Test A02 Event IoU、A04 ACC、A06 Pd、A07 Fa，以及 B01 约 4.0M 参数。它们标记为 `paper_reported`，不会伪装成本项目同口径实测结果。

其余 A/B 项必须通过 `EVSpSegNet_full_ab_server_20261008.zip` 在官方 CUDA 环境实测。服务器回传包通过内置导入器导入后，完整性校验应由 `incomplete` 变为 `complete`。
