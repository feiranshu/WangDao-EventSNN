# C06d DSP

本目录只比较 `C06d`，单位为 `count`；数据划分规则为 `common`。正式数值只读取本目录 CSV，模型顺序由根目录 `model_registry.csv` 决定。

缺值必须保留空白，并用 `status=not_reported`、`pending` 或 `not_applicable` 解释，禁止填 0。`source_file` 必须指向同一数据划分的模型源记录。
