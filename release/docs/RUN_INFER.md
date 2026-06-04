# RUN_INFER

## 1. 适用范围

这里的“推理/评估”基于 `paddlers/unet_data_split` 已训练好的 `best_model/` 目录。

脚本入口：

- `paddlers/unet_data_split/eval_best_model.py`
- `paddlers/unet_data_split/infer_best_model.py`

## 2. 先决条件

- 仓库源码存在
- `release/configs/*.json` 存在
- `GID15_DATASET_ROOT` 已设置
- `save_dir` 下已有：
  - `best_model/model.pdparams`
  - `best_model/model.pdopt`
  - `best_model/meta.json`

## 3. 评估 best model

single-head:

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
python paddlers/unet_data_split/eval_best_model.py \
  --config release/configs/unet_singlehead_portable.json \
  --save-dir release/runs/unet_singlehead \
  --device gpu
```

multi-head:

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
python paddlers/unet_data_split/eval_best_model.py \
  --config release/configs/unet_multihead_portable.json \
  --save-dir release/runs/unet_multihead \
  --device gpu
```

## 4. 导出预测样例

single-head:

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
python paddlers/unet_data_split/infer_best_model.py \
  --config release/configs/unet_singlehead_portable.json \
  --save-dir release/runs/unet_singlehead \
  --split test \
  --count 12 \
  --device gpu
```

multi-head:

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
python paddlers/unet_data_split/infer_best_model.py \
  --config release/configs/unet_multihead_portable.json \
  --save-dir release/runs/unet_multihead \
  --split test \
  --count 12 \
  --device gpu
```

## 5. 结果位置

评估后会在对应 `save_dir` 落盘：

- `best_model_metrics.json`
- `best_model_metrics.txt`
- `per_class_metrics.csv`
- `confusion_matrix.csv`
- `metrics/best_model_*`
- `prediction_samples/`

## 6. 如果只有模型文件、没有完整训练目录

最省事的方式是手动补成下面结构：

```text
<save_dir>/
  best_model/
    model.pdparams
    model.pdopt
    meta.json
```

然后用上面的 `--save-dir` 指过去即可。
