# FastSCNN Data Split

这套工程用于承接 `GID15 + cluster-routed FastSCNN` 的 single-head / multi-head 实验骨架。

当前固定口径：

- 聚类路由数：`K = 4`
- `cluster_id` 只决定样本路由，不进入像素监督
- 原始 segmentation mask 仍然是唯一监督信号
- 每个 segmentation head 的输出空间固定为 `16` 类：
  `background + 15 个 GID15 原始语义类`
- 最终评估固定在原始 GID15 语义空间上导出：
  `OA`、`Mean Acc`、`per-class Accuracy`、`per-class IoU`、`mIoU`
- cluster 不参与最终指标计算

当前这轮只完成工程骨架：

- 不训练
- 不做 smoke test
- 不启动后台任务
- `train.py` 默认只做 prepare、路径校验、配置快照和日志骨架输出

## 目录说明

- `train.py`: prepare-only 训练入口骨架
- `eval_best_model.py`: best model 评估脚本骨架
- `infer_best_model.py`: 推理导出脚本骨架
- `train_config_singlehead_baseline.json`: 正式公平 single-head 基线配置骨架
- `train_config_multihead.json`: `K=4` multi-head 配置骨架
- `cluster_routed_dataset.py`: 读取 cluster manifest，并为样本附加 `cluster_id`
- `metrics_utils.py`: confusion matrix、OA、Mean Acc、IoU、日志格式工具
- `fastscnn_routed_model.py`: 共享 FastSCNN trunk + routed heads
- `output/`: 输出目录骨架

## 数据口径

- 标准分割数据集：`${GID15_DATASET_ROOT}`
- 聚类映射数据集：`${GID15_CLUSTERED_DATASET_ROOT}`
- cluster 映射文件：
  `${GID15_CLUSTERED_DATASET_ROOT}/manifests/sample_cluster_mapping.csv`

默认样本图像与 mask 走：

- `input_image`
- `input_mask`

也就是仍然使用标准分割样本，只附加 `cluster_id` 做 head 路由。

## 模型结构

### Multi-head

- 共享 `LearningToDownsample`
- 共享 `GlobalFeatureExtractor`
- 共享 `FeatureFusion`
- `4` 个主 head
- 如果启用辅助损失，则也对应 `4` 个 aux head
- 每个 head 输出 `16` 类 logits

### Single-head

- 与 multi-head 共用同一套代码框架
- 差异只保留在：
  `num_heads=1`
- single-head 配置会把 manifest 中的原始 `cluster_id` 折叠为 `0`
- 路由语义等价于所有样本都走同一个共享 head

## 日志设计

训练日志从骨架阶段就固定为单写结构：

- stdout 只由 `logging.StreamHandler` 写一次
- 文件日志只由 `logging.FileHandler` 写一次
- 不使用 TeeStream
- 不把 stdout/stderr 再次重定向回同一个日志文件

预留日志格式：

```text
[start] total_epochs=40 train_dataset_size=10164 val_dataset_size=2244 steps_per_epoch=1271 model_type=ClusterRoutedFastSCNN num_heads=4
[path] best_model_dir=...
[path] latest_model_dir=...
[path] checkpoint_dir=...
[path] metrics_dir=...
[path] prediction_samples_dir=...
[train] epoch=1/40 step=20/1271 global_step=20 batch=8 loss=0.823451 lr=0.004991
[val] epoch=1/40 OA=0.756432 FG_OA=0.781205 mIoU=0.507881
```

其中：

- `OA` 是全部 `16` 类整体像素精度
- `FG_OA` 是去掉 background 后的 `15` 类整体像素精度
- `mIoU` 默认按全部 `16` 类语义空间汇总

