# Step5验证记录

- `selection_smoke/`：使用正式Step4摘要完成的选型冒烟，预期选择epoch8、阈值0.40；不冻结模型。
- 正式结果中的`verification/artifact_consistency.json`：检查源checkpoint、复制checkpoint、冻结清单、int8导出、指标和阈值的一致性；当前全部通过，int8回放最大绝对误差为0。
- 单元测试覆盖注册组合枚举、资格优先、高密度Fa、缺失指标、确定性排序及README单行命令。
- 源码和最终服务器包独立解压后的单元测试均为8/8通过。
- 最终服务器包已确认先删除占位数据目录再建立软链接；ZIP SHA-256为`1f755c64171874ac08ceaaf991dcf62f961b4fa59365fbbc36adeee5a6f5f3ee`。

Validation/Test均未读取。
