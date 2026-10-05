# A4-2模型卡

## 模型身份

- 模型：A4-2；
- 架构：H2 Proposal `128→48→24→16`，A4-2 Refiner `36→64→32→4`；
- 新增信息：前两个100 ms窗口的4维因果历史特征；
- 正式权重：seed42、epoch 8；
- 全局阈值：0.40；
- 数值路径：uint8激活、int8权重、int32乘加的exact-int8软件回放。

## 选择理由

A4-2在独立Validation和Test上都提高整体IoU，两个分区的IoU配对bootstrap 95%下界均大于0。Test上同时提高AP、Precision和F1，降低总体及高密度Fa。用户明确以整体IoU、Precision和误报控制为主要部署目标，因此接受A4-2作为本轮最终模型。

## 已知限制

Test高密度Recall相对H2-R下降1.2138个百分点，高密度Pd下降0.6920个百分点。高密度IoU提高3.0945个百分点、Precision提高5.5901个百分点、Fa下降13.49%，表现为更保守的精度—召回取舍。高密度Test仅3条序列，不能据此宣称所有高密度场景都稳定泛化。

不得根据已经查看的Test结果重新选择epoch或阈值。若未来必须恢复高密度Recall，需要建立新的开发数据和新的封存测试集。

## 文件与证据

- int8部署权重：`software/weights/A4-2_seed42_epoch008_int8.npz`；
- 训练checkpoint：`software/weights/A4-2_seed42_epoch008.pt`；
- 推理实现：`software/wangdao/a4_2_runtime.py`；
- 单序列入口：`software/scripts/run_a4_2.py`；
- 独立结果：`results/final_A4-2/`；
- 完整研究记录：`../seven_step_improvement_study/`。

