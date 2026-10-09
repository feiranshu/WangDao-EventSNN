# A+B 完整性核查（2026-10-08）

## H2 与 H2-plus

逐一检查 A 的 44 个 Validation/Test 指标 CSV 和 B 的 10 个 Common/Validation/Test 指标 CSV：

- `h2__software_exact_int8`：54/54 行均存在、`status=complete`、`value` 非空。
- `h2_plus__software_exact_int8`：54/54 行均存在、`status=complete`、`value` 非空。

两者无需补测。

## EV-SpSegNet

EV-SpSegNet 已改为 `formal` 软件模型。论文表 2可直接录入：Test A02 Event IoU、A04 ACC、A06 Pd、A07 Fa，以及 B01 约 4.0M 参数。它们标记为 `paper_reported`，不会伪装成本项目同口径实测结果。

其余 A/B 项必须通过 `EVSpSegNet_full_ab_server_20261008.zip` 在官方 CUDA 环境实测。服务器回传包通过内置导入器导入后，完整性校验应由 `incomplete` 变为 `complete`。

## EMS-YOLO

EMS-YOLO 已作为 `formal/software` 模型注册。现有实验报告给出的是框级 mAP、Recall 和 F1 工作点，不能替代本项目的 Event AP、事件 Pd/Fa 或可观测事件轨迹 HOTA；报告值不写入 A01–A22。B01–B04c 也缺少冻结权重同口径实测。

已建立 `EMS_YOLO_full_ab_server_20261009.zip`：严格核对冻结权重与配置，覆盖 Validation/Test 各 24 条完整序列；阈值固定 0.407。输入按原 250 ms 窗、100 ms 步长和 5 时间步无标签扫描，禁止复用只保留含目标窗口的 GT 筛选，再用滑窗框到事件适配器导出逐事件概率。服务器回传并导入前，EMS-YOLO 的 A/B 完整性保持 `incomplete`。
