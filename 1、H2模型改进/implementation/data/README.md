# 数据目录

## 目录结构

```text
data/
├─ README.md                         本说明
├─ index.json                        冻结角色、文件大小、事件数和SHA-256
└─ EV-UAV-dataset/                   用户自行放置的数据集；已被.gitignore排除
   ├─ train/                         Train-main、Train-dev和校准序列
   ├─ val/                           24条Validation序列
   └─ test/                          24条Test序列
```

`index.json`是唯一数据身份来源。真实数据集不随GitHub仓库提交；复现实验时必须保持文件名和SHA不变。Validation/Test已经用于最终独立验收，后续不得用它们调参后继续引用原Step7结论。

