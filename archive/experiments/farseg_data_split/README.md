# FarSeg Data Split

这套工程用于承接 `GID15 + cluster-routed 4-head FarSeg` 实验。

当前固定口径：

- 聚类路由数：`K = 4`
- head 数量：`num_heads = 4`
- 每个 head 的输出空间：`background + 15 个 GID15 原始语义类 = 16 类`
- `cluster_id` 只决定样本路由，不参与像素监督
- 原始 segmentation mask 继续作为唯一监督信号
- 最终评估以 GID15 原始语义标签空间为准，不以 cluster 为评估类别

当前这轮只完成工程骨架，不会自动训练，不会做 smoke test，不会启动后台任务。

## 目录说明

- `train.py`: 训练入口骨架；当前默认只做 prepare 和配置快照输出
- `eval_best_model.py`: best model 评估入口骨架
- `farseg_routed_model.py`: 共享 FarSeg 主干 + 4 个 segmentation heads
- `cluster_routed_dataset.py`: 读取 cluster manifest，并为样本附加 `cluster_id`
- `metrics_utils.py`: OA / FG_OA / mIoU / per-class Accuracy / per-class IoU 工具
- `train_config_multihead.json`: 4-head FarSeg 默认配置
- `output/`: 训练输出目录骨架

## 数据口径

- 标准语义标签：`${GID15_DATASET_ROOT}`
- 聚类路由数据：`${GID15_CLUSTERED_DATASET_ROOT}`
- cluster 映射文件：`${GID15_CLUSTERED_DATASET_ROOT}/manifests/sample_cluster_mapping.csv`

默认配置会从 manifest 读取 `cluster_id`，并使用 `output_image/output_mask` 作为样本路径。

## 训练日志设计

正式训练阶段将使用统一 logger，而不是 TeeStream 重定向：

- 每条日志只写一次到 stdout
- 同时由单独的 `FileHandler` 写入 `output/logs/train.log`
- 不会把 stdout/stderr 再双写回同一个日志文件

日志格式预留为：

```text
[start] model_type=ClusterRoutedFarSeg num_heads=4 total_epochs=40 train_dataset_size=... val_dataset_size=... steps_per_epoch=...
[path] best_model_dir=...
[path] latest_model_dir=...
[path] checkpoint_dir=...
[path] metrics_dir=...
[train] epoch=1/40 step=20/1270 global_step=20 batch=8 loss=0.823451 lr=0.004991
[val] epoch=1/40 OA=0.756432 FG_OA=0.781205 mIoU=0.507881
```

其中：

- `OA` 为 16 类整体像素精度
- `FG_OA` 为去掉 background 后的 15 类整体像素精度
- `mIoU` 默认按 foreground 15 类汇总
