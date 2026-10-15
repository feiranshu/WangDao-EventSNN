# 部署源记录

正式配置使用 formal/<evaluation_id>/<common|validation|test>/。C07 推荐单独 runtime.json（完整输入），其他项 summary.json（同评分前缀）；均用 evaluation-record-v2。原始 CSV/日志/清单与记录放在一起并计算哈希，测量边界和模板见 [C 类协议](../03_rules_and_protocols/README.md)。未测量不创建伪记录，重建表格保持 pending。
