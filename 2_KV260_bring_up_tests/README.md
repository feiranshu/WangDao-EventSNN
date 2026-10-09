# KV260 启动测试目录说明

本目录以 `plan/README_route_C.md` 为基准，只在活动目录中保留与路线 C 有关的项目文件；整理日期：2026-10-02。凡与路线 C 的环境、数据通路、冻结 SNN、真实样本、安装恢复或 KV260 官方实现有关的内容均保留，不区分已完成或未完成。

## 目录结构

```text
2_KV260_bring_up_tests/
├── README.md
├── part1/                       # 第一部分：板卡、系统、网络与构建环境基线
├── part2/                       # 第二部分：AXI DMA / vadd 数据通路实验
├── part3/                       # 第三部分：冻结版 SNN 上板与真实样本对拍
├── references_and_installation/          # 当前安装镜像/固件和 KV260 参考资料
└── plan/                        # 各保留一份路线 A 与路线 C
```

## 三部分对应关系

| 目录 | 路线 C 对应内容 | 主要内容 |
|---|---|---|
| `part1` | 第一部分：前期准备 | 刷写前状态、系统/网络/XRT 检查、CMA 记录、外部内核模块构建验证 |
| `part2` | 第二部分：AXI DMA 的 PS DDR ↔ PL 实验 | vadd host/kernel、SD 数据生成器、TCP 客户端和原构建配置 |
| `part3` | 第三部分：冻结版 SNN 的 AXI-Lite 上板实验 | 已验证工作包、真实样本工具、协议 probe、上传包和 sample_000 |

## 参考资料与计划

- `references_and_installation/installation_images_and_boot_firmware/`：路线 C 当前使用的 K26 Boot Firmware 与 Generic EDF 2025.1 镜像。
- `references_and_installation/KV260_references/`：官方硬件资料、教程、参考实现和中文指南；其中本地克隆的第三方官方仓库仅供离线参考，不嵌入本项目 Git 历史。
- `plan/README_route_C.md`：当前路线 C 执行基准。
- `plan/README_route_A_XRT.md`：XRT 路线 A 参考方案。

`plan/` 当前只保留上述两份文档；`references_and_installation/` 当前只保留上述两个子文件夹。

## 历史归档

历史材料已经统一迁入根级本地目录`../archive/2_KV260_bring_up_tests/`，包括早于板端修正版的旧 SNN 包、Python 缓存留档、原空目录标记，以及 2026-10-02 路线 C 筛选时移出的重复计划、旧恢复镜像和迁移记录。根归档不上传 GitHub。

## 使用入口

1. 查看当前执行基准：读 `plan/README_route_C.md`。
2. 查看板卡与系统基线：读 `part1/README.md`。
3. 查看 DMA/vadd 工作：读 `part2/README.md`。
4. 继续冻结 SNN 真实样本对拍：读 `part3/README.md`，再读 `part3/real_sample_tools/README.md`。
5. 需要刷卡、恢复或查官方资料时，进入 `references_and_installation/`。

## 本次筛选结果

`part1/`、`part2/`、`part3/` 以及其余顶层目录结构未改变。旧恢复镜像、重复计划和迁移日志均按原来源层级迁入根归档`../archive/2_KV260_bring_up_tests/route_C_screening_archive_2026-10-02/`，没有删除文件。

