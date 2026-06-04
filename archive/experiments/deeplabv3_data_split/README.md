# DeepLabV3 Data Split

这套工程为 `GID15 + cluster-aware multi-head DeepLabV3` 的训练骨架。

当前口径固定为：

- 分割监督类别：GID15 原始 16 类（含 `background`）
- 聚类路由类别：4 个 cluster
- 模型形态：共享 encoder + 共享 ASPP / trunk + 4 个 segmentation heads
- 路由方式：样本读取 `cluster_id` 后，只走对应 head；监督依然使用该样本原始 mask

本目录当前只完成：

- 训练目录骨架
- 配置文件
- cluster-aware 数据读取与校验逻辑
- 多 head DeepLabV3 模型定义
- output 目录准备
- dry-run 配置快照输出

本目录当前不会自动启动训练。

## 目录说明

- `train.py`: 训练入口；默认只做 dry-run 校验和快照写出
- `train_config.json`: 默认训练配置
- `cluster_routed_dataset.py`: 读取 cluster manifest，并为样本附加 `cluster_id`
- `multihead_deeplabv3.py`: 共享主干 + 4 个 head 的模型骨架
- `metrics_utils.py`: 后续评估所需的 OA / per-class Acc / IoU / confusion matrix 工具
- `start_train_tmux.sh`: 后续正式启动训练的 tmux 脚本骨架
- `start_train_nohup.sh`: 后续正式启动训练的 nohup 脚本骨架
- `output/`: 训练输出目录骨架

## 数据来源

- 标准随机分割数据集：`${GID15_DATASET_ROOT}/gid15_dataset`
- 聚类数据集：`${GID15_DATASET_ROOT}/gid15_clustered_dataset`
- cluster 映射文件：`manifests/sample_cluster_mapping.csv`

## 关键口径

- `num_heads = 4`，严格等于当前最终选定的 `K = 4`
- `num_classes = 16`，对应 `background + 15` 个 GID15 语义类别
- cluster 只决定路由 head，不改变 mask 语义
- 后续评估仍按 GID15 原始 16 类输出：
  `per-class Accuracy`, `OA`, `per-class IoU`, `mIoU`, `confusion matrix`
