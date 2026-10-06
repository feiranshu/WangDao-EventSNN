# H2 统一 int8 评测

这是与 H2-plus 使用同一套协议的冻结 H2 基线，不是重新训练模型。

- 数值路径：`exact_int8_replay`。
- 量化形式：uint8 激活、int8 权重、int32 乘加。
- 指标协议：`evuav-33e2aff-official-compatible-v1+step3-density-20000`。
- 工作点：全局阈值 0.30。
- Train-dev：AP 0.9274106331，IoU 0.7805328724。
- Validation：AP 0.9610006356，IoU 0.8615992017。
- Test：AP 0.9675966378，IoU 0.8595001275。

目录结构：

- `int8/`：Train-dev 统一基线。
- `validation/int8/`、`test/int8/`：独立 Validation/Test 验收基线。

H2-F 与 H2-R 是同一模型的不同阈值工作点，PR 曲线相同。当前正式展示名称统一写作 `H2`。
