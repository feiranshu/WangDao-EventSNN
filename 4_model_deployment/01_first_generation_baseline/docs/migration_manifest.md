# 第一代目录迁移记录

## 迁移结论

原 `1_first_generation_SNN_model` 已一次性迁移为 `01_first_generation_baseline`。不保留旧路径兼容层。

- 模型代码与权重：`software/` → `model_runtime/` 与 `host_tools/`。
- PL 设计：`hardware/rtl|tb|vectors|vivado` → `pl_design/`。
- KV260 部署：原平铺目录拆分为 `firmware`、`pc_automation`、`board_runtime`、`smoke_tests` 和 `measurement`。
- 正式证据：`results/` → `verification_evidence/`。
- 已完成板端运行：迁入 `experiments/board_runs/train_dev10/`。
- 可再生成旧包和备份：迁入根级本地`../../../archive/4_model_deployment/01_first_generation_baseline/`；`__pycache__` 与 `.pyc` 已清理。

## 冻结制品

| 文件 | Bytes | SHA256 |
|---|---:|---|
| `wangdao_pl_flat.bit.bin` | 7,797,692 | `3bf7473f26f5ddcd1157c2513706d179982eb131e8cb38880601d30e105dee23` |
| `wangdao_pl_flat.dtbo` | 1,986 | `0a1c9eb846b6929129b315b5e3ff510a06e332a74744dc048a9bc46c47ca2706` |
| `shell.json` | 50 | `0377daaf2050b88df14de9a79d120144cf49381d00fa552402395a322cb7b1a3` |

完整迁移前后校验结果位于上级 `migration_records/2026-10-08/`。两代数据集仅通过 Junction 使用 E 盘唯一实体副本。目录迁移不构成新的板端实测。

