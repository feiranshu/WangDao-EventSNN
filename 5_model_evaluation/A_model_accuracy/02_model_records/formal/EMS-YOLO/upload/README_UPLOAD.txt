EMS-YOLO 正式 A+B 补测包

1. 上传并解压 EMS_YOLO_full_ab_server_20261009.zip。
2. 激活已经跑通 EMS-YOLO 的虚拟环境。
3. 按包内 README_SERVER.md 先运行 inspect_server.sh，再运行 run_all.sh。
4. 下载 EMS_YOLO_full_ab_return_*.zip 及同名 .sha256。
5. 本地用 import_ems_yolo_results.py 导入；不要手工抄录结果。

当前正式状态：已注册但待服务器实测。实验报告的 bbox mAP 不写入 A01。
