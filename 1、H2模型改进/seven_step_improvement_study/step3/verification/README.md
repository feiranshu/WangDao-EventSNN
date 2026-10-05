# 步骤三本地验证

本目录只保留紧凑验证摘要，不保留候选缓存、本地checkpoint或预测结果。

- `local_preflight.json`：79条Train-main、10条Train-dev、冻结索引和H2权重哈希检查通过。
- `failed_smoke_zero_update.json`：第一次smoke只取前20条Train-main，全部属于普通密度，A3-2正确拒绝空的高密度层。该失败促使smoke改为18条普通密度加2条真实高密度序列。
- `zero_update_smoke_summary.json`：修复后A3-1/A3-2两条zero-update路径全部通过；参数、概率、浮点AP及int8 AP保存前后保持一致。
- `d3_0_smoke_summary.json`：五种探针、两个seed的单轮执行验证完成；smoke数值不构成性能结论。
- `smoke_pipeline_summary.json`：完整CPU smoke状态为`smoke_complete`，A3-1/A3-2均完成int8导出和重新加载，Validation/Test均未使用。
- `step3_fix_verification.json`：2026-10-04针对恢复模式隔离、D3-0状态校验、续训结果收集和逐轮饱和率日志完成13项源码测试及服务器包独立解压测试。

本地运行环境为Python 3.14、PyTorch 2.11 CPU、NumPy 2.4；服务器包仍以项目冻结的Python 3.10、PyTorch 1.9.1、NumPy 1.24.3为兼容目标。正式CUDA结果必须由服务器回传。
