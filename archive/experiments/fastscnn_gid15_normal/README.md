# FastSCNN GID-15 Normal

目录名使用 `fastscnn_gid15_normal`，而不是带 `-` 的 `Fast-SCNN_gid15_normal`，因为这样更稳妥：脚本路径、后续 Python 工程复用、shell 变量展开都更友好，也不需要依赖带连字符的包导入。

## 目录结构

- `train.py`: FastSCNN 训练入口，按验证集 `mIoU` 保存 `best_model`
- `train_config.json`: 数据、训练、评估、采样和启动相关配置
- `eval_best_model.py`: 重新评估 `best_model` 并导出 OA、mIoU、Mean Accuracy、per-class accuracy
- `infer_best_model.py`: 导出 `best_model` 的预测结果与可视化结果
- `utils_metrics.py`: confusion matrix、OA、per-class metrics、pred/vis 导出工具
- `resume_from_latest.sh`: 自动定位最新 checkpoint 并续训
- `start_train_tmux.sh`: 用固定 tmux session 后台启动训练
- `start_train_nohup.sh`: 用 `nohup + setsid + bash -lc` 脱离终端启动训练
- `output/`: 训练日志、checkpoint、best_model 和评估产物

## 数据集配置

数据路径和类别配置在 `train_config.json`：

- `dataset_root`
- `train_file_list`
- `val_file_list`
- `test_file_list`
- `label_list`
- `class_names`
- `num_classes`

默认会优先查找当前仓库里已有的 GID15 数据路径候选。

## 训练命令

前台训练：

```bash
python paddlers/fastscnn_gid15_normal/train.py
```

指定配置或输出目录：

```bash
python paddlers/fastscnn_gid15_normal/train.py \
  --config paddlers/fastscnn_gid15_normal/train_config.json \
  --save-dir paddlers/fastscnn_gid15_normal/output
```

## 持久化启动

tmux 启动：

```bash
bash paddlers/fastscnn_gid15_normal/start_train_tmux.sh
```

nohup 启动：

```bash
bash paddlers/fastscnn_gid15_normal/start_train_nohup.sh
```

指定 GPU：

```bash
CUDA_DEVICE=1 bash paddlers/fastscnn_gid15_normal/start_train_tmux.sh
CUDA_DEVICE=1 bash paddlers/fastscnn_gid15_normal/start_train_nohup.sh
```

## 续训

自动从最新 checkpoint 续训：

```bash
bash paddlers/fastscnn_gid15_normal/resume_from_latest.sh
```

指定 GPU：

```bash
CUDA_DEVICE=1 bash paddlers/fastscnn_gid15_normal/resume_from_latest.sh
```

也可以直接传给训练脚本：

```bash
python paddlers/fastscnn_gid15_normal/train.py \
  --resume-checkpoint paddlers/fastscnn_gid15_normal/output/epoch_12
```

## best model 评估与导出

重新评估 `best_model`：

```bash
python paddlers/fastscnn_gid15_normal/eval_best_model.py
```

强制评估验证集：

```bash
python paddlers/fastscnn_gid15_normal/eval_best_model.py --final-eval-split val
```

单独导出预测与可视化：

```bash
python paddlers/fastscnn_gid15_normal/infer_best_model.py --split test
```

## output 产物

- `output/best_model/`: 按验证集 `mIoU` 自动保存的最佳模型
- `output/best_model_metrics.json`: 结构化评估结果，包含 OA、mIoU、Mean Accuracy、每类 accuracy、confusion matrix
- `output/best_model_metrics.txt`: 文本版指标摘要
- `output/per_class_metrics.csv`: 每类指标表，包含 `class_accuracy` 和 `iou`
- `output/confusion_matrix.npy`: confusion matrix
- `output/pred/`: 预测标签图
- `output/vis/`: 预测可视化图
- `output/train.log`: 训练主日志
- `output/eval.log`: best model 评估摘要
- `output/train_tmux.log`: tmux 启动日志
- `output/train_nohup.log`: nohup 启动日志
- `output/train_resume.log`: 自动续训日志
- `output/train_config.json`: 实际解析后的运行配置
- `output/dataset_report.json`: 当前 GID15 划分与类别分布统计
- `output/oversample_report.json`: 针对弱类的训练列表重复采样记录
- `output/run_summary.json`: 主要产物索引

## 如何查看 OA 和每类准确率

- `best_model_metrics.txt` 里有 OA、mIoU、Mean Accuracy 和每类指标
- `per_class_metrics.csv` 里有 `class_id`、`class_name`、`class_accuracy`、`iou`
- `best_model_metrics.json` 里保留了结构化 summary 和 confusion matrix

这里的 `class_accuracy` 是按 confusion matrix 的行归一化计算的，不是 IoU。

## FastSCNN 的合理预期

- FastSCNN 是轻量模型，训练和推理更快，显存压力更小
- 在 GID15 这类多类遥感语义分割任务上，精度通常不应被假定一定优于更重的模型
- 这套实验的目标是让 FastSCNN 在当前仓库和数据条件下尽量接近它的合理上限，同时保持训练稳定、可续训、可评估、可复查

## 这次做的保守优化

- 继承了当前仓库里更稳的 `512 crop + val mIoU 选 best_model + 训练后自动评估导出` 骨架
- 默认使用 `CITYSCAPES` 预训练，不走无预训练硬训
- 保留 `CE + Lovasz` mixed loss 和 FastSCNN 自带辅助分支损失权重
- 只做了轻量的弱类样本重复采样，默认聚焦 `garden_plot`、`artificial_grassland`、`pond`
- 重复采样只设为 `2`，避免为了抬弱类把整体 OA / mIoU 拉崩
- 没有默认叠加更激进的 rare-aware crop、复杂损失组合或大幅改动主优化器
