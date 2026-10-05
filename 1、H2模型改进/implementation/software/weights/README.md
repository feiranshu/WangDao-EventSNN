# 冻结权重

| 文件 | 作用 | SHA-256 |
| --- | --- | --- |
| `H2_D1_seed42.npz` | 原H2 Proposal及基线Refiner；A4-2部署复用其中Proposal | `21019377617e429a599497a0a6b0f541d2c05bf7bc8fee83005deae72bb578b2` |
| `A4-2_seed42_epoch008_int8.npz` | 最终36→64→32→4 exact-int8 Refiner | `d7d42a3876c37497bdcd5206e6f35a5f949b0230dde0675fd128f989b70d05c4` |
| `A4-2_seed42_epoch008.pt` | 对应PyTorch checkpoint，供复现和审计，不是部署必需 | `a9c71311dbc01c6142ed826697b466546f3404aeb76798e9ede1461a35d3119f` |

正式阈值0.40记录在int8权重元数据和`results/final_A4-2/deployment_decision.json`中。不要用Test结果替换这些文件。

