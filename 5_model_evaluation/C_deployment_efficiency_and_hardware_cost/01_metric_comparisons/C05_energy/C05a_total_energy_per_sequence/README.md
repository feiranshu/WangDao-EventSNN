# C05a total energy per sequence

本目录只比较 `C05a`，单位为 `J/sequence`；数据划分规则为 `validation/test`。正式数值只读取本目录 CSV，模型顺序由根目录 `model_registry.csv` 决定。

缺值必须保留空白，并用 `status=not_reported`、`pending` 或 `not_applicable` 解释，禁止填 0。`source_file` 必须指向同一数据划分的模型源记录。
