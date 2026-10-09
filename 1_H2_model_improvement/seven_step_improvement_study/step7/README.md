# Step7：A4-2补充独立验收

状态：正式运行已于2026-10-05完成。本目录保留冻结协议、评测代码、服务器复现包和完整结果；最终部署入口已经迁入`../../implementation/`。

## 当前定位

Step7固定使用Step5冻结并经Step6三seed验证的部署候选：A4-2 seed42、epoch 8、阈值0.40、exact-int8。epoch 8是按“先过保护门槛、再最大化整体IoU”的预注册规则选出的检查点；它不是所有单项指标的绝对最大值，但拥有完整的Step5冻结、Step6三seed稳定性和matched-null因果证据链，因此不在Step7改选epoch 10。

Step6原协议因为seed44普通密度Fa相对升幅6.36%超过5%而未正式晋级，该结论保留不变。根据用户明确决定，本步骤建立一个在读取Validation/Test之前冻结的补充独立验收协议；它不反向修改Step6，只回答A4-2能否在独立Validation和Test上保持相对H2的收益。

## 固定评测对象

- A4-2：seed42、epoch 8、阈值0.40，exact-int8为正式结果，float为量化影响参考；
- H2-F：原始阈值0.28；
- H2-R：Train-dev阶段已经冻结的阈值0.30；
- 密度边界：20,000 events/s；
- 数据顺序：先Validation，再Test，两者均只用于评测；
- 禁止训练、继续训练、改模型、改epoch、改阈值、改量化或改后处理。

## 输出内容

每个数据角色都会生成完整指标JSON、PR与阈值曲线CSV/SVG、固定工作点比较CSV/SVG、逐序列对比、候选/H2的bootstrap分数档案、10,000次配对序列bootstrap和门槛结论。正式主比较为A4-2 exact-int8对H2-R，H2-F用于保留原始阈值与Fa/Pd保护参照。

最终`step7_decision.json`同时报告：

- Validation与Test的固定协议结果；
- 原有安全门槛是否通过；
- Test上的AP与IoU是否改善；
- Test整体IoU差值的bootstrap 95%下界是否大于0；
- “独立确认替代”“有提升但保护门槛/置信度未全过”或“未确认替代”三类结论。

## 正式结果与最终解释

Validation固定协议完整执行，所有安全门槛通过。Test上，A4-2 exact-int8相对H2-R的AP提高0.1712个百分点、整体IoU提高0.9431个百分点、Precision提高1.2933个百分点、F1提高0.5427个百分点、总体Fa下降6.11%；高密度IoU提高3.0945个百分点、高密度Fa下降13.49%。Test整体IoU差值的10,000次配对序列bootstrap 95%区间为+0.0565至+2.2878个百分点。

唯一未通过的Test安全项是高密度Recall：由77.6897%降至76.4759%，下降1.2138个百分点，超过预冻结的0.3个百分点保护线。因此原始`step7_decision.json`正确保留`passes_all_independent_acceptance_checks=false`。

用户在完整知悉该取舍后，明确选择以整体IoU、Precision和误报控制作为部署优先级，正式接受A4-2。这个工程决策记录在`../../implementation/results/final_A4-2/deployment_decision.json`，不会覆盖或重写本目录的原始科研判定。

## 文件目录

```text
step7/
├─ README.md                         本地研究说明与目录
├─ README_SERVER.md                  服务器单行操作命令
├─ step7_protocol.json               打开Validation/Test前冻结的正式协议
├─ step7_common.py                   数据隔离、门槛、差值、bootstrap与图表公共函数
├─ run_step7.py                      前置检查、合成冒烟、正式评测、恢复与结果校验入口
├─ collect_step7_results.py          排除缓存后打包回传正式结果
├─ build_server_package.py           构建并校验Linux服务器上传包
├─ requirements.txt                  服务器环境版本基线
├─ run_server.sh                     Linux入口包装
├─ tests/
│  └─ test_step7.py                  协议、隔离、bootstrap、Fa门槛与README格式测试
├─ verification/
│  └─ README.md                      本地验证记录
├─ upload/
   ├─ H2_step7_server_20261005.zip
   ├─ H2_step7_server_20261005.zip.sha256
   └─ H2_step7_server_20261005.manifest.json
└─ step7_results_20261005/           服务器回传的完整正式结果
   ├─ preflight.json                 环境、哈希、冻结输入和数据隔离检查
   ├─ step7_protocol.lock.json       正式运行时锁定的协议副本
   ├─ step7_decision.json            严格门槛结论与各检查状态
   ├─ RESULT_MANIFEST.json           回传结果文件清单和完整性信息
   ├─ validation/                    Validation指标、PR图、逐序列表和bootstrap
   └─ test/                          Test指标、PR图、逐序列表和bootstrap
```

服务器操作以[README_SERVER.md](README_SERVER.md)为唯一命令入口。所有命令均为可整条复制的一行形式。

