# A 类复现入口

在项目根目录运行：

```powershell
python 1_H2_model_improvement/implementation/software/scripts/generate_accuracy_reports.py
python 1_H2_model_improvement/implementation/software/scripts/validate_evaluation_coverage.py
```

第一条命令从 `02_model_records/` 中按 `model_registry.csv` 发现模型，并重建 A01–A22 各自的 `validation.csv`、`test.csv`。它不重新推理、不调参、不生成图表。第二条命令按软件 A+B、板端 A+C 规则更新并检查 `evaluation_coverage.csv`；正式缺项时返回失败。

重新计算 H2/H2-plus 可观测事件轨迹指标时运行：

```powershell
python 1_H2_model_improvement/implementation/software/scripts/evaluate_target_tracks.py --split all
```

该入口校验冻结权重哈希，只在 Train-dev 复现后处理选择，并写入对应稳定 `evaluation_id` 下的 `validation/test/target_track_level/`。Validation/Test 不参与调参。

EV-SpSegNet 在 CUDA 服务器运行 `EVSpSegNet_full_ab_server_20261008.zip` 后，将回传 ZIP 下载到本地，再运行：

```powershell
python 1_H2_model_improvement/implementation/software/scripts/import_ev_spsegnet_results.py <EVSpSegNet_full_ab_return_*.zip>
```

导入器验证清单与 SHA-256，一次写入 A/B 正式源记录并重建比较表；已有导入需显式加 `--replace`，旧结果会先进入 archive。

EMS-YOLO 在已跑通其 EV-UAV 工程的 CUDA 服务器运行 `EMS_YOLO_full_ab_server_20261009.zip`。先按包内说明检查原始数据、预处理样本和冻结权重，再下载回传 ZIP 并运行：

```powershell
python 1_H2_model_improvement/implementation/software/scripts/import_ems_yolo_results.py <EMS_YOLO_full_ab_return_*.zip>
```

导入器要求 Validation/Test 各 24 条冻结序列、固定阈值 0.407、指定 checkpoint SHA-256，并同时导入 A01–A22 与 B01–B04c。重复导入同样需要显式 `--replace`。
