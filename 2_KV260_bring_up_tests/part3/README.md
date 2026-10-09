# Part 3：冻结版 SNN 上板与真实样本对拍

本目录对应路线 C 第三部分。目标是用已验证的冻结 Proposal SNN 完成 AXI-Lite 双 patch 冒烟、输入布局解析、reset 语义验证和真实样本 16 路 Q12 raw logits 位精确对拍。

## 当前入口与安全边界

- board_working_2026-09-19/：板端回收的已验证工作版、日志和原 DTBO 备份；恢复硬件包时以这里为准。
- real_sample_tools/：PC 导出/对拍脚本、板端安全 runner、协议 probe 和上传包；先读其 README.md。
- real_samples/sample_000/：第一个真实 patch 的特征、目标、golden 和 24 种 byte-lane 候选。
- 整理前顶层旧包已移到根级本地`../../archive/2_KV260_bring_up_tests/legacy_snn_package_before_cleanup/`，不得覆盖这里的已验证 DTBO/脚本。
- sample_000_meta.json 当前标记 board_ready: false；协议与 reset 未确认前，不应宣称真实样本已上板通过。
- 板端 runner 必须在结束或异常时恢复 k26-starter-kits；风扇/温度未闭环前只做短时测试。

## 生成文件命名

- *_words.txt：128 个 32-bit 输入 word。
- *_expected*.txt：候选排列下的 16 个软件 golden raw logits。
- order_0123 等：四个时间步到 32-bit byte lane 的候选排列。
- b0123_l0213 等：time-major 测试中的 block 顺序和 lane 顺序候选。
- *_manifest.json / SHA256SUMS.json：生成参数、候选集合和完整性校验。
- upload_bundle/：板端上传用自包含副本；与源目录重复是有意设计。

## 逐文件索引

下表覆盖本目录除本 README 之外的全部文件；重复 probe 也逐项列出。

| 文件 | 大小 | 作用 |
|---|---:|---|
| `board_working_2026-09-19\devmem-wrapper` | 733 B | 板端 devmem 兼容包装器，供 AXI-Lite 寄存器读写脚本调用。 |
| `board_working_2026-09-19\wangdao_board_package\artifact_preflight.sh` | 1.2 KiB | 加载前强制校验 bitstream、DTBO 与 shell.json 的字节数和 SHA-256。 |
| `board_working_2026-09-19\wangdao_board_package\board_run.sh` | 1.3 KiB | 板端基础安装与双 patch 脚本；正式测试优先使用带恢复逻辑的安全 runner。 |
| `board_working_2026-09-19\wangdao_board_package\patch0_expected.txt` | 192 B | 全零 patch 的 16 个预期 Q12 raw logits。 |
| `board_working_2026-09-19\wangdao_board_package\patch0_words.txt` | 1.5 KiB | 全零 patch 的 128 个 32 位输入 word。 |
| `board_working_2026-09-19\wangdao_board_package\patch1_expected.txt` | 192 B | 全 0xFFFFFFFF patch 的 16 个预期 Q12 raw logits。 |
| `board_working_2026-09-19\wangdao_board_package\patch1_words.txt` | 1.5 KiB | 全 0xFFFFFFFF patch 的 128 个 32 位输入 word。 |
| `board_working_2026-09-19\wangdao_board_package\safe_board_run.sh` | 3.6 KiB | 带产物预检、活动应用保护和 trap 的安全双 patch runner，结束或失败时恢复默认 PL。 |
| `board_working_2026-09-19\wangdao_board_package\shell.json` | 50 B | xmutil PL_FLAT 应用描述，声明单槽位 shell。 |
| `board_working_2026-09-19\wangdao_board_package\wangdao_axi_smoke.sh` | 3.4 KiB | 通过 AXI-Lite 写入 128 words、启动 SNN 并校验 16 logits。 |
| `board_working_2026-09-19\wangdao_board_package\wangdao_pl_flat.bit.bin` | 7.4 MiB | 冻结 Proposal SNN 的已验证 PL bitstream。 |
| `board_working_2026-09-19\wangdao_board_package\wangdao_pl_flat.dtbo` | 1.9 KiB | 与冻结 bitstream 配套、已修正并验证可加载的 Device Tree Overlay。 |
| `board_working_2026-09-19\wangdao_dtbo_backup\wangdao_pl_flat.original.dtbo` | 2.0 KiB | 修正前 DTBO 备份，仅用于差异追溯。 |
| `board_working_2026-09-19\wangdao_route_c_fixed_dtbo_run.log` | 2.1 KiB | 修正 DTBO 后的一次路线 C 运行日志。 |
| `board_working_2026-09-19\wangdao_route_c_run.log` | 2.0 KiB | 早期路线 C 运行日志，用于问题追溯。 |
| `board_working_2026-09-19\wangdao_route_c_run2.log` | 2.1 KiB | 第二次路线 C 运行日志，用于问题追溯。 |
| `board_working_2026-09-19\wangdao_route_c_run3.log` | 2.1 KiB | 第三次路线 C 运行日志，用于问题追溯。 |
| `board_working_2026-09-19\wangdao_route_c_snn_smoke_final.log` | 383.0 KiB | 冻结 SNN 冒烟测试的阶段性最终日志。 |
| `board_working_2026-09-19\wangdao_route_c_snn_smoke_fixed_devmem.log` | 2.3 KiB | 修复 devmem 调用后的 SNN 冒烟日志。 |
| `board_working_2026-09-19\wangdao_route_c_snn_smoke_fixed_wrapper.log` | 9.3 KiB | 修正版 wrapper 的成功基线日志，是主要 PASS 证据。 |
| `real_sample_tools\board\safe_lane_probe.sh` | 1.7 KiB | 安全加载 PL、运行 lane probe、保存捕获并恢复默认 PL。 |
| `real_sample_tools\board\safe_one_sample.sh` | 844 B | 安全运行一个 128-word/16-logit 样本并逐位比较。 |
| `real_sample_tools\board\safe_real_sample_resolver.sh` | 1.2 KiB | 依次运行少量真实样本候选以解析输入排列。 |
| `real_sample_tools\board\safe_reset_probe.sh` | 1.0 KiB | 按 patch1→patch0→patch1 验证 START 是否清空 SNN 状态。 |
| `real_sample_tools\board\safe_time_major_probe.sh` | 1.9 KiB | 安全运行 time-major probe，解析 block/time 与 lane 布局。 |
| `real_sample_tools\board\wangdao_axi_capture.sh` | 3.3 KiB | 写入一个输入并捕获 16 个实际输出，不预先比较 golden。 |
| `real_sample_tools\board\wangdao_axi_smoke.sh` | 3.7 KiB | 通过 AXI-Lite 写入 128 words、启动 SNN 并校验 16 logits。 |
| `real_sample_tools\board\artifact_preflight.sh` | 1.2 KiB | 真实样本测试加载前使用的部署产物强制校验脚本。 |
| `real_sample_tools\board\wangdao_safe_lib.sh` | 2.6 KiB | 带产物预检、活动应用保护和默认 PL 恢复的共享函数库。 |
| `real_sample_tools\common.py` | 8.5 KiB | 真实样本与 probe 工具的共用函数：模型、量化推理、打包、哈希和校验。 |
| `real_sample_tools\compare_lane_capture.py` | 2.5 KiB | 将板端捕获与 24 种 byte-lane 候选比较，识别真实排列。 |
| `real_sample_tools\export_real_patch.py` | 8.9 KiB | 从 EV-UAV 数据导出一个真实 Proposal patch、候选 words 和 golden logits。 |
| `real_sample_tools\finalize_sample.ps1` | 1.7 KiB | 按已确认排列固定正式样本文件并更新元数据和校验。 |
| `real_sample_tools\generate_protocol_probes.py` | 6.3 KiB | 生成可区分的 byte-lane 协议 probe 及候选 golden。 |
| `real_sample_tools\generate_time_major_probes.py` | 5.1 KiB | 生成用于确认 four-block/time-major 输入布局的 probe。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_0123.txt` | 176 B | lane probe 00 在 byte-lane 排列 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_0132.txt` | 176 B | lane probe 00 在 byte-lane 排列 0132 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_0213.txt` | 176 B | lane probe 00 在 byte-lane 排列 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_0231.txt` | 176 B | lane probe 00 在 byte-lane 排列 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_0312.txt` | 176 B | lane probe 00 在 byte-lane 排列 0312 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_0321.txt` | 176 B | lane probe 00 在 byte-lane 排列 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_1023.txt` | 176 B | lane probe 00 在 byte-lane 排列 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_1032.txt` | 176 B | lane probe 00 在 byte-lane 排列 1032 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_1203.txt` | 176 B | lane probe 00 在 byte-lane 排列 1203 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_1230.txt` | 176 B | lane probe 00 在 byte-lane 排列 1230 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_1302.txt` | 176 B | lane probe 00 在 byte-lane 排列 1302 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_1320.txt` | 176 B | lane probe 00 在 byte-lane 排列 1320 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_2013.txt` | 176 B | lane probe 00 在 byte-lane 排列 2013 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_2031.txt` | 176 B | lane probe 00 在 byte-lane 排列 2031 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_2103.txt` | 176 B | lane probe 00 在 byte-lane 排列 2103 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_2130.txt` | 176 B | lane probe 00 在 byte-lane 排列 2130 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_2301.txt` | 176 B | lane probe 00 在 byte-lane 排列 2301 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_2310.txt` | 176 B | lane probe 00 在 byte-lane 排列 2310 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_3012.txt` | 176 B | lane probe 00 在 byte-lane 排列 3012 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_3021.txt` | 176 B | lane probe 00 在 byte-lane 排列 3021 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_3102.txt` | 176 B | lane probe 00 在 byte-lane 排列 3102 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_3120.txt` | 176 B | lane probe 00 在 byte-lane 排列 3120 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_3201.txt` | 176 B | lane probe 00 在 byte-lane 排列 3201 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_expected_order_3210.txt` | 176 B | lane probe 00 在 byte-lane 排列 3210 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_lane_values.npy` | 640 B | lane probe 00 的原始四时间步/byte-lane 数值矩阵。 |
| `real_sample_tools\generated\lane_probe\lane_probe_00_words.txt` | 1.4 KiB | lane probe 00 的 128 个板端输入 word。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_0123.txt` | 176 B | lane probe 01 在 byte-lane 排列 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_0132.txt` | 176 B | lane probe 01 在 byte-lane 排列 0132 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_0213.txt` | 176 B | lane probe 01 在 byte-lane 排列 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_0231.txt` | 176 B | lane probe 01 在 byte-lane 排列 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_0312.txt` | 176 B | lane probe 01 在 byte-lane 排列 0312 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_0321.txt` | 176 B | lane probe 01 在 byte-lane 排列 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_1023.txt` | 176 B | lane probe 01 在 byte-lane 排列 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_1032.txt` | 176 B | lane probe 01 在 byte-lane 排列 1032 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_1203.txt` | 176 B | lane probe 01 在 byte-lane 排列 1203 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_1230.txt` | 176 B | lane probe 01 在 byte-lane 排列 1230 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_1302.txt` | 176 B | lane probe 01 在 byte-lane 排列 1302 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_1320.txt` | 176 B | lane probe 01 在 byte-lane 排列 1320 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_2013.txt` | 176 B | lane probe 01 在 byte-lane 排列 2013 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_2031.txt` | 176 B | lane probe 01 在 byte-lane 排列 2031 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_2103.txt` | 176 B | lane probe 01 在 byte-lane 排列 2103 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_2130.txt` | 176 B | lane probe 01 在 byte-lane 排列 2130 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_2301.txt` | 176 B | lane probe 01 在 byte-lane 排列 2301 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_2310.txt` | 176 B | lane probe 01 在 byte-lane 排列 2310 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_3012.txt` | 176 B | lane probe 01 在 byte-lane 排列 3012 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_3021.txt` | 176 B | lane probe 01 在 byte-lane 排列 3021 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_3102.txt` | 176 B | lane probe 01 在 byte-lane 排列 3102 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_3120.txt` | 176 B | lane probe 01 在 byte-lane 排列 3120 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_3201.txt` | 176 B | lane probe 01 在 byte-lane 排列 3201 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_expected_order_3210.txt` | 176 B | lane probe 01 在 byte-lane 排列 3210 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_lane_values.npy` | 640 B | lane probe 01 的原始四时间步/byte-lane 数值矩阵。 |
| `real_sample_tools\generated\lane_probe\lane_probe_01_words.txt` | 1.4 KiB | lane probe 01 的 128 个板端输入 word。 |
| `real_sample_tools\generated\lane_probe\lane_probe_manifest.json` | 9.4 KiB | lane probe 候选排列、文件、哈希和生成参数清单。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0123_l0123.txt` | 176 B | time-major probe 00 在 block 顺序 0123、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0123_l0213.txt` | 176 B | time-major probe 00 在 block 顺序 0123、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0123_l1023.txt` | 176 B | time-major probe 00 在 block 顺序 0123、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0132_l0123.txt` | 176 B | time-major probe 00 在 block 顺序 0132、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0132_l2103.txt` | 176 B | time-major probe 00 在 block 顺序 0132、lane 顺序 2103 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0213_l0123.txt` | 176 B | time-major probe 00 在 block 顺序 0213、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0213_l0213.txt` | 176 B | time-major probe 00 在 block 顺序 0213、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0213_l0231.txt` | 176 B | time-major probe 00 在 block 顺序 0213、lane 顺序 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0213_l0321.txt` | 176 B | time-major probe 00 在 block 顺序 0213、lane 顺序 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0213_l1023.txt` | 176 B | time-major probe 00 在 block 顺序 0213、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0231_l0123.txt` | 176 B | time-major probe 00 在 block 顺序 0231、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0231_l0213.txt` | 176 B | time-major probe 00 在 block 顺序 0231、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0231_l0231.txt` | 176 B | time-major probe 00 在 block 顺序 0231、lane 顺序 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0231_l0321.txt` | 176 B | time-major probe 00 在 block 顺序 0231、lane 顺序 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0231_l1023.txt` | 176 B | time-major probe 00 在 block 顺序 0231、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0321_l0123.txt` | 176 B | time-major probe 00 在 block 顺序 0321、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_expected_b0321_l0213.txt` | 176 B | time-major probe 00 在 block 顺序 0321、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_00_words.txt` | 1.4 KiB | time-major probe 00 的 128 个板端输入 word。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0123_l0123.txt` | 176 B | time-major probe 01 在 block 顺序 0123、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0123_l0213.txt` | 176 B | time-major probe 01 在 block 顺序 0123、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0123_l1023.txt` | 176 B | time-major probe 01 在 block 顺序 0123、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0132_l0123.txt` | 176 B | time-major probe 01 在 block 顺序 0132、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0132_l2103.txt` | 176 B | time-major probe 01 在 block 顺序 0132、lane 顺序 2103 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0213_l0123.txt` | 176 B | time-major probe 01 在 block 顺序 0213、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0213_l0213.txt` | 176 B | time-major probe 01 在 block 顺序 0213、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0213_l0231.txt` | 176 B | time-major probe 01 在 block 顺序 0213、lane 顺序 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0213_l0321.txt` | 176 B | time-major probe 01 在 block 顺序 0213、lane 顺序 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0213_l1023.txt` | 176 B | time-major probe 01 在 block 顺序 0213、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0231_l0123.txt` | 176 B | time-major probe 01 在 block 顺序 0231、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0231_l0213.txt` | 176 B | time-major probe 01 在 block 顺序 0231、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0231_l0231.txt` | 176 B | time-major probe 01 在 block 顺序 0231、lane 顺序 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0231_l0321.txt` | 176 B | time-major probe 01 在 block 顺序 0231、lane 顺序 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0231_l1023.txt` | 176 B | time-major probe 01 在 block 顺序 0231、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0321_l0123.txt` | 176 B | time-major probe 01 在 block 顺序 0321、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_expected_b0321_l0213.txt` | 176 B | time-major probe 01 在 block 顺序 0321、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_01_words.txt` | 1.4 KiB | time-major probe 01 的 128 个板端输入 word。 |
| `real_sample_tools\generated\time_major_probe\time_major_probe_manifest.json` | 4.7 KiB | time-major probe 的候选 block/lane 布局、文件与生成参数清单。 |
| `real_sample_tools\prepare_upload_bundle.ps1` | 2.9 KiB | 汇总板端所需硬件包、脚本、probe 与校验文件。 |
| `real_sample_tools\README.md` | 5.8 KiB | 真实样本位精确对拍工程的操作说明、当前状态与验收闸门。 |
| `real_sample_tools\real_sample_resolver\sample_000_expected_logits.txt` | 176 B | 真实样本解析使用的 16 个软件 golden logits。 |
| `real_sample_tools\real_sample_resolver\sample_000_input_words_order_1032.txt` | 1.4 KiB | 真实样本在排列 1032 下的 128 个输入 word。 |
| `real_sample_tools\real_sample_resolver\sample_000_input_words_order_1302.txt` | 1.4 KiB | 真实样本在排列 1302 下的 128 个输入 word。 |
| `real_sample_tools\real_sample_resolver\sample_000_input_words_order_1320.txt` | 1.4 KiB | 真实样本在排列 1320 下的 128 个输入 word。 |
| `real_sample_tools\real_sample_resolver\SHA256SUMS.json` | 661 B | 真实样本解析候选的 SHA-256 完整性清单。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_0123.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_0132.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 0132 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_0213.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_0231.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_0312.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 0312 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_0321.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_1023.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_1032.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 1032 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_1203.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 1203 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_1230.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 1230 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_1302.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 1302 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_1320.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 1320 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_2013.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 2013 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_2031.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 2031 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_2103.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 2103 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_2130.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 2130 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_2301.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 2301 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_2310.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 2310 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_3012.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 3012 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_3021.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 3021 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_3102.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 3102 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_3120.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 3120 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_3201.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 3201 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_expected_order_3210.txt` | 176 B | 板端上传包副本：lane probe 00 在 byte-lane 排列 3210 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_00_words.txt` | 1.4 KiB | 板端上传包副本：lane probe 00 的 128 个板端输入 word。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_0123.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_0132.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 0132 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_0213.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_0231.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_0312.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 0312 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_0321.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_1023.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_1032.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 1032 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_1203.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 1203 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_1230.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 1230 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_1302.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 1302 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_1320.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 1320 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_2013.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 2013 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_2031.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 2031 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_2103.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 2103 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_2130.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 2130 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_2301.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 2301 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_2310.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 2310 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_3012.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 3012 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_3021.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 3021 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_3102.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 3102 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_3120.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 3120 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_3201.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 3201 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_expected_order_3210.txt` | 176 B | 板端上传包副本：lane probe 01 在 byte-lane 排列 3210 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_01_words.txt` | 1.4 KiB | 板端上传包副本：lane probe 01 的 128 个板端输入 word。 |
| `real_sample_tools\upload_bundle\lane_probe\lane_probe_manifest.json` | 9.4 KiB | 板端上传包副本：lane probe 候选排列、文件、哈希和生成参数清单。 |
| `real_sample_tools\upload_bundle\patch0_expected.txt` | 192 B | 板端上传包副本：全零 patch 的 16 个预期 Q12 raw logits。 |
| `real_sample_tools\upload_bundle\patch0_words.txt` | 1.5 KiB | 板端上传包副本：全零 patch 的 128 个 32 位输入 word。 |
| `real_sample_tools\upload_bundle\patch1_expected.txt` | 192 B | 板端上传包副本：全 0xFFFFFFFF patch 的 16 个预期 Q12 raw logits。 |
| `real_sample_tools\upload_bundle\patch1_words.txt` | 1.5 KiB | 板端上传包副本：全 0xFFFFFFFF patch 的 128 个 32 位输入 word。 |
| `real_sample_tools\upload_bundle\README_PC.md` | 5.8 KiB | 板端上传包副本：随板端上传包携带的 PC 侧操作说明副本。 |
| `real_sample_tools\upload_bundle\safe_lane_probe.sh` | 1.7 KiB | 板端上传包副本：安全加载 PL、运行 lane probe、保存捕获并恢复默认 PL。 |
| `real_sample_tools\upload_bundle\safe_one_sample.sh` | 844 B | 板端上传包副本：安全运行一个 128-word/16-logit 样本并逐位比较。 |
| `real_sample_tools\upload_bundle\safe_reset_probe.sh` | 1.0 KiB | 板端上传包副本：按 patch1→patch0→patch1 验证 START 是否清空 SNN 状态。 |
| `real_sample_tools\upload_bundle\safe_time_major_probe.sh` | 1.9 KiB | 板端上传包副本：安全运行 time-major probe，解析 block/time 与 lane 布局。 |
| `real_sample_tools\upload_bundle\SHA256SUMS.json` | 17.9 KiB | 板端上传包副本：目录内文件的 SHA-256 与字节数清单。 |
| `real_sample_tools\upload_bundle\artifact_preflight.sh` | 1.2 KiB | 板端上传包副本：加载前强制校验已验证部署产物。 |
| `real_sample_tools\upload_bundle\shell.json` | 50 B | 板端上传包副本：xmutil PL_FLAT 应用描述，声明单槽位 shell。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0123_l0123.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0123、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0123_l0213.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0123、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0123_l1023.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0123、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0132_l0123.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0132、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0132_l2103.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0132、lane 顺序 2103 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0213_l0123.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0213、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0213_l0213.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0213、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0213_l0231.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0213、lane 顺序 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0213_l0321.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0213、lane 顺序 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0213_l1023.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0213、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0231_l0123.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0231、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0231_l0213.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0231、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0231_l0231.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0231、lane 顺序 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0231_l0321.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0231、lane 顺序 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0231_l1023.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0231、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0321_l0123.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0321、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_expected_b0321_l0213.txt` | 176 B | 板端上传包副本：time-major probe 00 在 block 顺序 0321、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_00_words.txt` | 1.4 KiB | 板端上传包副本：time-major probe 00 的 128 个板端输入 word。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0123_l0123.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0123、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0123_l0213.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0123、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0123_l1023.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0123、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0132_l0123.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0132、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0132_l2103.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0132、lane 顺序 2103 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0213_l0123.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0213、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0213_l0213.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0213、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0213_l0231.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0213、lane 顺序 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0213_l0321.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0213、lane 顺序 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0213_l1023.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0213、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0231_l0123.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0231、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0231_l0213.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0231、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0231_l0231.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0231、lane 顺序 0231 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0231_l0321.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0231、lane 顺序 0321 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0231_l1023.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0231、lane 顺序 1023 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0321_l0123.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0321、lane 顺序 0123 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_expected_b0321_l0213.txt` | 176 B | 板端上传包副本：time-major probe 01 在 block 顺序 0321、lane 顺序 0213 假设下的 16 个软件 golden 输出。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_01_words.txt` | 1.4 KiB | 板端上传包副本：time-major probe 01 的 128 个板端输入 word。 |
| `real_sample_tools\upload_bundle\time_major_probe\time_major_probe_manifest.json` | 4.7 KiB | 板端上传包副本：time-major probe 的候选 block/lane 布局、文件与生成参数清单。 |
| `real_sample_tools\upload_bundle\wangdao_axi_capture.sh` | 3.3 KiB | 板端上传包副本：写入一个输入并捕获 16 个实际输出，不预先比较 golden。 |
| `real_sample_tools\upload_bundle\wangdao_axi_smoke.sh` | 3.7 KiB | 板端上传包副本：通过 AXI-Lite 写入 128 words、启动 SNN 并校验 16 logits。 |
| `real_sample_tools\upload_bundle\wangdao_pl_flat.bit.bin` | 7.4 MiB | 板端上传包副本：冻结 Proposal SNN 的已验证 PL bitstream。 |
| `real_sample_tools\upload_bundle\wangdao_pl_flat.dtbo` | 1.9 KiB | 板端上传包副本：与冻结 bitstream 配套、已修正并验证可加载的 Device Tree Overlay。 |
| `real_sample_tools\upload_bundle\wangdao_safe_lib.sh` | 2.6 KiB | 板端上传包副本：带预检、活动应用保护和默认 PL 恢复的共享函数库。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_0123.txt` | 1.4 KiB | sample_000 在排列 0123 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_0132.txt` | 1.4 KiB | sample_000 在排列 0132 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_0213.txt` | 1.4 KiB | sample_000 在排列 0213 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_0231.txt` | 1.4 KiB | sample_000 在排列 0231 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_0312.txt` | 1.4 KiB | sample_000 在排列 0312 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_0321.txt` | 1.4 KiB | sample_000 在排列 0321 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_1023.txt` | 1.4 KiB | sample_000 在排列 1023 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_1032.txt` | 1.4 KiB | sample_000 在排列 1032 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_1203.txt` | 1.4 KiB | sample_000 在排列 1203 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_1230.txt` | 1.4 KiB | sample_000 在排列 1230 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_1302.txt` | 1.4 KiB | sample_000 在排列 1302 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_1320.txt` | 1.4 KiB | sample_000 在排列 1320 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_2013.txt` | 1.4 KiB | sample_000 在排列 2013 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_2031.txt` | 1.4 KiB | sample_000 在排列 2031 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_2103.txt` | 1.4 KiB | sample_000 在排列 2103 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_2130.txt` | 1.4 KiB | sample_000 在排列 2130 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_2301.txt` | 1.4 KiB | sample_000 在排列 2301 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_2310.txt` | 1.4 KiB | sample_000 在排列 2310 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_3012.txt` | 1.4 KiB | sample_000 在排列 3012 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_3021.txt` | 1.4 KiB | sample_000 在排列 3021 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_3102.txt` | 1.4 KiB | sample_000 在排列 3102 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_3120.txt` | 1.4 KiB | sample_000 在排列 3120 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_3201.txt` | 1.4 KiB | sample_000 在排列 3201 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\input_candidates\sample_000_input_words_order_3210.txt` | 1.4 KiB | sample_000 在排列 3210 下的 128 个输入 word，待协议确认后选定。 |
| `real_samples\sample_000\sample_000_expected_logits.txt` | 176 B | sample_000 的 16 个位精确软件 golden logits。 |
| `real_samples\sample_000\sample_000_meta.json` | 8.5 KiB | sample_000 的来源、形状、编码、候选排列、哈希与板端状态。 |
| `real_samples\sample_000\sample_000_target_cells.npy` | 144 B | sample_000 对应的 4×4 Proposal 目标 cell。 |
| `real_samples\sample_000\sample_000_temporal_uint8.npy` | 640 B | sample_000 的原始 [4,128] uint8 时间特征。 |

