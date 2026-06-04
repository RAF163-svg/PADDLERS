# PaddleRS Release Handoff

这个 `release/` 目录用于把当前仓库整理成一个“别人接手后能更快跑起来”的交付包。

它不是完整源码仓库的替代品。推荐的交付方式是：

1. 交付整个 `PaddleRS` 仓库源码。
2. 同时保留这个 `release/` 目录。
3. 按 `release/docs/` 里的步骤准备环境、数据和运行命令。

## 当前主交付线

- 当前最完整、最建议对外移交的实验线是 `paddlers/unet_data_split`
- 这条线已经支持：
  - `single-head UNet`
  - `4-head routed UNet`
  - 统一数据读取、日志、评估、checkpoint
- `release/configs/` 里提供的是可移交版配置，不再依赖固定的本机仓库绝对路径

## 这个目录里有什么

- `docs/`
  运行说明、数据布局说明、依赖审计说明
- `configs/`
  可移交版 single-head / multi-head 配置
- `scripts/`
  可直接调用的启动与检查脚本
- `data_specs/`
  轻量元数据快照，不包含大体量训练数据
- `env/`
  当前机器导出的环境文件
- `custom_code/`
  自定义源码快照，方便对照和二次移植
- `manifests/`
  打包建议、排除清单、绝对路径审计结果

## 当前最小外部依赖

要跑 `paddlers/unet_data_split`，仓库之外仍然至少需要：

- 一个可访问的 GID15 正式语义分割数据根目录
- 已安装好的 Paddle / PaddleRS 运行环境
- 可选的官方 UNet 预训练权重下载能力

当前 release 已经把下面这些“小而关键”的元信息带进来了：

- `labels.txt`
- `dataset_summary.json`
- `cluster_summary.json`
- `sample_cluster_mapping.relative.csv`

因此，当前 release 不再强依赖整套 `gid15_clustered_dataset` 图像与 mask，只需要它的路由元信息。

## 快速开始

先看：

- [RUN_TRAIN.md](docs/RUN_TRAIN.md)
- [RUN_INFER.md](docs/RUN_INFER.md)
- [DATA_LAYOUT.md](docs/DATA_LAYOUT.md)
- [DEPENDENCY_AUDIT.md](docs/DEPENDENCY_AUDIT.md)
