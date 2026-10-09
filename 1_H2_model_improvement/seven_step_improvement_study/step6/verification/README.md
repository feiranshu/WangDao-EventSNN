# Step6本地工程验证

验证日期：2026-10-05。本目录只保留精简结论，不保留可再生成的候选缓存、smoke checkpoint或指标副本。

## 已通过

- Step5服务器结果复核：冻结checkpoint、120组合审计、两套决策、H2指标、A4-2关键int8表格及一致性报告与本地结论一致；差异仅来自NPZ/SVG重新序列化、绝对路径和约`1e-16`级浮点表示。
- 本地preflight：H2/索引/Step5 checkpoint哈希、A4-2-null epoch8/0.40匹配点、79条Train-main与10条Train-dev全部通过。
- 初版单元测试：6项测试覆盖分层抽样、带分数并列的精确bootstrap AP、匹配null归因、三seed门槛与README单行命令。
- 四任务CPU smoke：A4-2/A4-2-null的seed43/44全部完成加载、单批训练、逐轮输出、checkpoint保存/重载、float/int8评测、固定阈值导出和历史可用性分组。
- 中断恢复：完整任务被复用；未完成任务先保留到`failed_attempts/`再重跑，没有覆盖证据。
- 真实Step5事件分数集成测试：密度分层配对bootstrap和六组敏感性均成功重算，完整集ΔIoU与Step5点估计一致。
- 初版服务器包独立解压后的6项测试全部通过，SHA-256为`dabbe24db29cfc75dacc8d1081a2c64ae7b5ac1a75888cba4faca611e78033b6`。
- 正式服务器首次运行暴露出密度参数使用浮点逐位比较的可移植性缺陷，发生在训练任务启动之前。修正版增加3项回归测试：接受不改变uint8编码的浮点尾差、拒绝会改变编码的参数差异、拒绝候选数量差异。
- 初版包单行就地热修复已在初版ZIP的独立解压副本上实际执行并通过语法编译；命令能唯一命中旧检查且不会删除已有preflight或缓存。
- `H2_step6_server_20261005_hotfix1.zip`已经通过构建器的POSIX权限、逐文件SHA和内置manifest校验，并通过Python语法编译；SHA-256为`99022495bd29264c0dc3adb686d9ed25f70f38dcaac4d57a48e30f03be4c88bf`。新增完整单元测试需在带PyTorch的服务器环境执行。
- 10,000次AP bootstrap在CUDA环境自动使用float64 GPU批处理，避免CPU统计阶段成为瓶颈。

smoke只证明工程链路，不能用于判断seed43/44效果。正式结论必须来自四卡CUDA的10轮运行、固定epoch8/阈值0.40和10,000次bootstrap；Validation/Test仍未使用。
