# A 模型准确度

`01_metric_comparisons/01_event_level/` 保存 A01–A13，`02_target_track_level/` 保存 A14–A22。每项分别有 `validation.csv` 与 `test.csv`，二者不得交叉引用源数据。

`02_model_records/formal/<evaluation_id>/<split>/` 保存正式模型事件级与目标轨迹级源结果。A 对软件正式模型和板端正式模型都是必测项。完整定义、事件窗口和可观测事件轨迹 HOTA 协议见 `03_rules_and_protocols/`。

EV-SpSegNet 已作为软件正式模型注册。ICCV 2025 论文表 2直接给出的 Test Event IoU、ACC、Pd、Fa 以 `paper_reported` 保存，仅用于说明论文公开结果，不替代本项目 Validation/Test 同口径测量，也不计为完整性通过。官方 CUDA 服务器包会输出 A01–A22 所需全部源文件，回传导入后由生成器改为 `complete`。

EMS-YOLO 也作为软件正式模型注册。模型按原 250 ms 窗、100 ms 步长和 5 时间步扫描完整序列；不复用原先仅保存含目标窗口的 GT 筛选。固定适配器把每个事件赋为所有覆盖滑窗预测框的最大置信度，未覆盖事件为 0，再计算相同的 A01–A22。窗口选择和适配器都不读取 `label/name`。固定阈值 0.407 来自入选前 Validation F1 峰值，Validation 结果必须标注选择相关性，Test 用作主要无偏比较。实验报告的目标框 `mAP@0.5` 不等于 A01 Event AP，不写入正式指标 CSV。
