# 第二代目录迁移记录

## 迁移结论

原 `2_second_generation_SNN_model` 已一次性迁移为 `02_second_generation_ps_software_optimized`，不保留旧路径兼容层。第二代仍是完整独立副本，但 PL 证据明确标记为继承第一代。

- 冻结模型、PL 设计、部署工具、测试和文档按新的语义目录重新分类。
- 第一代复制来的仿真、综合和联调证据迁入 `verification_evidence/inherited_first_gen_pl_baseline/`。
- 第二代新增计时、功耗与运行证据统一进入 `verification_evidence/runtime_measurements/`。
- 旧备份和旧生成包进入根级本地`../../../archive/4_model_deployment/02_second_generation_ps_software_optimized/`；缓存已清理。

## 复用的冻结制品

| 文件 | Bytes | SHA256 |
|---|---:|---|
| `wangdao_pl_flat.bit.bin` | 7,797,692 | `3bf7473f26f5ddcd1157c2513706d179982eb131e8cb38880601d30e105dee23` |
| `wangdao_pl_flat.dtbo` | 1,986 | `0a1c9eb846b6929129b315b5e3ff510a06e332a74744dc048a9bc46c47ca2706` |
| `shell.json` | 50 | `0377daaf2050b88df14de9a79d120144cf49381d00fa552402395a322cb7b1a3` |

旧清单中的 2024-byte DTBO 哈希已废止。完整迁移校验位于上级 `migration_records/2026-10-08/`；两代数据集仅通过 Junction 使用 E 盘唯一实体副本；本次没有重新生成 bitstream 或重新运行开发板。

