# 正式冻结权重

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `H2_D1_seed42.npz` | 冻结 H2 Proposal 和基线 Refiner | `21019377617e429a599497a0a6b0f541d2c05bf7bc8fee83005deae72bb578b2` |
| `H2-pro-Causal-ANN-v1_int8.npz` | 正式 C1 因果 Refiner，阈值 0.35 | `6e9221829459825413c58604a40da3582137a555aaa508d84d9da8325aa5368d` |
| `H2-pro-Causal-ANN-v1_float.pt` | 对应 float teacher，供审计与转换 | `dd4b81fb5cef1b5e5f120afbc5e30c9b7d3b0e5f48cd275a0caf0534e2112c91` |

旧 H2-plus/A4-2 权重已归入 `../../../seven_step_improvement_study/step7/intermediate_H2-plus/weights/`，不属于正式部署权重。

