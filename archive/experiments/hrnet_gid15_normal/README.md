# HRNet GID-15 Normal

这个目录提供一套面向 GID15 的 HRNet 语义分割实验，目标是兼顾效果、训练稳定性和结果可复查性。

## 目录结构

- `train.py`: HRNet 训练入口，按验证集 `mIoU` 保存 `best_model`
- `train_config.json`: 数据集、模型、训练和评估配置
- `eval_best_model.py`: 重新评估 `best_model` 并导出 OA、mIoU、Mean Accuracy、per-class accuracy
- `infer_best_model.py`: 导出 `best_model` 的预测结果与可视化结果
- `utils_metrics.py`: confusion matrix、OA、per-class accuracy、预测导出工具
- `resume_from_latest.sh`: 自动寻找最近 checkpoint 并续训
- `start_train_tmux.sh`: 用 `tmux` 启动稳定训练
- `start_train_nohup.sh`: 用 `nohup + setsid` 启动稳定训练
- `output/`: 训练日志、checkpoint、best_model 和评估产物

## 数据集配置

主要配置在 `train_config.json`：

- `dataset_root`
- `train_file_list`
- `val_file_list`
- `test_file_list`
- `label_list`
- `class_names`
- `num_classes`

默认优先使用：

- `${GID15_DATASET_ROOT}/gid15_seed20260320_tile640_stride448`
- `${GID15_DATASET_ROOT}/paddlers_gid15_benchmark_20260320/data/gid15_seed20260320_tile640_stride448`
- `${GID15_DATASET_ROOT}/gid-15`

## 前台训练

```bash
python paddlers/hrnet_gid15_normal/train.py
```

如果要指定配置或输出目录：

```bash
python paddlers/hrnet_gid15_normal/train.py \
  --config paddlers/hrnet_gid15_normal/train_config.json \
  --save-dir paddlers/hrnet_gid15_normal/output
```

## tmux 启动

```bash
bash paddlers/hrnet_gid15_normal/start_train_tmux.sh
```

默认 session 名：

```bash
hrnet_gid15_train
```

查看和进入：

```bash
tmux ls
tmux attach -t hrnet_gid15_train
tail -f paddlers/hrnet_gid15_normal/output/train_tmux.log
```

## nohup 启动

```bash
bash paddlers/hrnet_gid15_normal/start_train_nohup.sh
```

查看：

```bash
tail -f paddlers/hrnet_gid15_normal/output/train_nohup.log
cat paddlers/hrnet_gid15_normal/output/train.pid
ps -fp "$(cat paddlers/hrnet_gid15_normal/output/train.pid)"
```

## 续训

自动从最近 checkpoint 恢复：

```bash
bash paddlers/hrnet_gid15_normal/resume_from_latest.sh
```

指定 GPU：

```bash
CUDA_DEVICE=1 bash paddlers/hrnet_gid15_normal/resume_from_latest.sh
```

## best_model 评估与推理

重新评估 `best_model`：

```bash
python paddlers/hrnet_gid15_normal/eval_best_model.py
```

如果只评估验证集：

```bash
python paddlers/hrnet_gid15_normal/eval_best_model.py --final-eval-split val
```

导出 `best_model` 预测与可视化：

```bash
python paddlers/hrnet_gid15_normal/infer_best_model.py --split test
```

## output/ 产物说明

- `output/best_model/`: 训练过程中按验证集 `mIoU` 自动保存的最佳模型
- `output/best_model_metrics.json`: 结构化评估结果，包含 OA、mIoU、Mean Accuracy、per-class accuracy、confusion matrix
- `output/best_model_metrics.txt`: 文本版评估摘要
- `output/per_class_metrics.csv`: 每类指标表，至少包含 `class_id`、`class_name`、`class_accuracy`、`iou`
- `output/confusion_matrix.npy`: confusion matrix 原始矩阵
- `output/pred/`: 预测标签图
- `output/vis/`: 可视化叠加结果
- `output/train.log`: 训练主日志
- `output/eval.log`: best_model 评估日志摘要
- `output/train_tmux.log`: tmux 启动日志
- `output/train_nohup.log`: nohup 启动日志
- `output/train_resume.log`: 自动续训日志
- `output/train_config.json`: 训练实际生效配置快照
- `output/dataset_report.json`: 数据集类别分布报告
- `output/oversample_report.json`: 针对弱类的训练列表重复采样记录
- `output/pred_manifest.json`: 导出的预测文件索引

## 如何查看每类准确率和 OA

- `best_model_metrics.txt` 里有 OA、mIoU、Mean Accuracy 和每类指标
- `per_class_metrics.csv` 里有结构化表格，`class_accuracy` 是基于 confusion matrix 计算的每类准确率
- `best_model_metrics.json` 里保留了结构化 summary 和 confusion matrix

## HRNet 的合理预期

- HRNet 一般比 FastSCNN 更重，显存和训练耗时更高
- HRNet 更适合保留高分辨率特征，对边界和细粒度类别通常更友好
- 在 GID15 上，它有希望比 FastSCNN 更稳地改善弱类和易混类别
- 但它不一定必然超过当前最强的 FastSeg，最终还是要以验证集 `best_model` 为准

## 这次的保守优化

- 继承了当前仓库里更稳的 `512 crop + val mIoU 选 best_model + 训练后自动评估导出` 骨架
- 默认使用 `CITYSCAPES` 预训练，不走无预训练硬训
- 保留 `CE + Lovasz` mixed loss，但默认不开启 class weights，避免对整体 OA / mIoU 造成过强扰动
- 对 `garden_plot`、`artificial_grassland`、`pond` 做轻量重复采样，默认只重复 `2` 次
- `save_interval_epochs` 固定为 `1`，减少长任务中断时的进度损失
- best_model 导出不依赖 `load_model()` 反序列化，而是按当前配置重建 HRNet 后手动加载 `model.pdparams`，提升复查稳定性
