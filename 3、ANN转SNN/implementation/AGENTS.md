# MLP转SNN子项目协作规则

## 范围

本目录负责LIF/SNN、S0/S1/S2/S3、STC、定点化、PoT、LOA、位精确模型、计算代理与RTL向量。H2 ANN准确性改进不在这里进行；如需新的ANN教师，从`../../../1、H2_S1模型改进研究计划/`取得冻结版本并记录哈希。

## 数据和证据

- Train-main训练；Train-dev选择；HW-calibration选择位宽和近似规则。
- Validation只在阶段冻结后运行一次；Test只用于最终冻结模型。
- 浮点SNN、定点软件、RTL仿真、综合结果和板级实测必须分开表述。
- 软件运算计数不是功耗，综合估算不是板测。

## 转换纪律

- 每次只转换一个阶段或引入一种数值变化。
- ANN教师、SNN学生、阈值、时间步、复位、读出和数值格式必须版本化。
- 完整SNN只有在单级转换对照通过后才启动。
- 近似计算必须先通过exact对齐；PoT和LOA首轮不得叠加。
- Python黄金模型与RTL的操作顺序、位宽、饱和和复位规则必须一致。

## 文件与运行

- `software/wangdao/h2/`和`H2_D1_seed42.npz`是转换所需的只读ANN参考副本，不在本目录继续优化。
- 正式结果放`results/`；训练过程、缓存、checkpoint和预测放`results/runs/`。
- 数据集目录联接不进入上传包。
- 修改实现或结论后同步更新README、PLAN和MIGRATION_MANIFEST。
- 当前本地回归为51项；正式运行前必须全部通过。
