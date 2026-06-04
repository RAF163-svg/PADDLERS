# FarSeg GID-15 Normal

这套实验以 [deeplabv3_gid_nromal](<PROJECT_ROOT>/paddlers/deeplabv3_gid_nromal) 为主骨架重建，目标是在当前仓库条件下提供一套更完整的 `FarSeg + GID-15` 训练与最佳模型评估流程。

## 设计原则

- 训练主流程、数据检查、best model 逻辑优先继承 `deeplabv3_gid_nromal`
- 只在 FarSeg 的模型接入方式上参考现有 FarSeg/FactSeg 相关实现
- 默认保留更稳妥的 GID-15 经验：`512 crop`、`mIoU` 选最佳、`CE + Lovasz` mixed loss、轻量翻转与颜色扰动
- 不默认继承 `fastseg_gid15_nromal` 的稀有类裁剪和更复杂的优化器/增强组合

## 训练

```bash
python paddlers/farseg_gid15_normal/train.py
```

如需指定别的配置文件或输出目录：

```bash
python paddlers/farseg_gid15_normal/train.py \
  --config paddlers/farseg_gid15_normal/train_config.json \
  --save-dir paddlers/farseg_gid15_normal/output
```

## 单独评估 best model

```bash
python paddlers/farseg_gid15_normal/eval_best_model.py
```

也可以强制评估验证集：

```bash
python paddlers/farseg_gid15_normal/eval_best_model.py --final-eval-split val
```

## 主要输出

- `output/best_model/`: 训练过程中按验证集 `mIoU` 自动保存的最佳模型
- `output/best_model_metrics.json`: 结构化评估结果，包含 OA、mIoU、Mean Accuracy、per-class accuracy、confusion matrix
- `output/best_model_metrics.txt`: 文本版评估摘要
- `output/per_class_metrics.csv`: 每类指标表
- `output/confusion_matrix.npy`: 混淆矩阵
- `output/pred/`: 最佳模型在最终评估 split 上的预测标签图
- `output/vis/`: 可视化结果
- `output/dataset_report.json`: GID-15 当前划分和类别分布统计
- `output/oversample_report.json`: 训练列表过采样记录
- `output/train_config.json`: 解析后的实际运行配置
- `output/run_summary.json`: 本次运行的产物索引
