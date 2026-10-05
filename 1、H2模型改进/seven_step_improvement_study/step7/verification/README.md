# Step7本地验证记录

此目录只保存本地工程验证说明，不保存正式Validation/Test指标。正式数据结果必须由服务器包一次性运行后回传。

已完成以下本地验证：

- 5个Python文件通过语法编译；
- 7项单元测试通过，覆盖协议锁、角色隔离、密度分层bootstrap、事件指标重算、Fa保护门槛、缺失分组和README单行命令；
- 从最终ZIP完整解压后再次运行7项测试，全部通过；
- 解压包内冻结checkpoint完成7条合成36维特征exact-int8冒烟，概率有限且位于0—1，Validation/Test均未读取；
- ZIP构建器逐文件验证SHA-256、字节数和POSIX权限，最终包含37个受清单保护的文件；
- 最终服务器包SHA-256为`a8458f27106f90d43f437f68fdc0f5c9a4022d23f5addf9ff6b1031fdbecf073`。

本地没有生成正式Validation/Test指标。正式数据只由服务器包的一次`run-all`生成。

