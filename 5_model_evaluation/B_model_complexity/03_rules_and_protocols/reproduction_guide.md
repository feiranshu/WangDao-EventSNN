# B 类复现入口

只用现有模型源记录重建分指标比较表：

```powershell
python 1_H2_model_improvement/implementation/software/scripts/generate_model_complexity_report.py --tables-only
python 1_H2_model_improvement/implementation/software/scripts/validate_evaluation_coverage.py
```

完整重算 H2/H2-plus 算法复杂度：

```powershell
python 1_H2_model_improvement/implementation/software/scripts/generate_model_complexity_report.py --split all
```

完整重算会逐序列核对 A 类事件数与候选数，把 Common、Validation、Test 分别写入稳定 `evaluation_id` 的模型记录，再生成 B01/B02 Common 与 B03/B04a/B04b/B04c 的 Validation/Test CSV。B04c 必须严格等于同模型同划分的 B04a+B04b，并作为与 B03 稠密 MAC 比较的正式口径。

EV-SpSegNet 使用 A 类目录中同一个 `EVSpSegNet_full_ab_server_20261008.zip` 完成官方推理和 B01–B04 统计。回传后运行：

```powershell
python 1_H2_model_improvement/implementation/software/scripts/import_ev_spsegnet_results.py <EVSpSegNet_full_ab_return_*.zip>
```

导入器会同时写入 A 与 B，验证 B04c 恒等式，并更新正式权重 SHA-256。
