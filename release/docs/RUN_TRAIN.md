# RUN_TRAIN

## 1. 前提

- 交付对象拿到的是整个 `PaddleRS` 仓库源码，而不是只拿 `release/`
- 当前主训练入口使用仓库内源码：
  - `paddlers/unet_data_split/train.py`
  - `paddlers/unet_data_split/start_train_nohup.sh`
- 当前 release 里的配置使用：
  - `release/configs/unet_singlehead_portable.json`
  - `release/configs/unet_multihead_portable.json`

## 2. 环境准备

推荐直接复用 `release/env/environment.yml`。

```bash
conda env create -f release/env/environment.yml
conda activate paddlers
```

如果对方机器已经有相近环境，也可以只把 `release/env/requirements-freeze.txt` 当审计参考。

注意：

- `requirements-freeze.txt` 包含本机构建痕迹和 editable git 依赖
- 可移交复现时，优先使用 `environment.yml`

## 3. 数据准备

必须提供一个正式 GID15 数据根目录，并设置环境变量：

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
```

这个目录至少应包含：

- `images/`
- `masks/`

当前 routed 训练不再要求你额外准备完整的 `gid15_clustered_dataset` 图像与 mask；release 已自带路由元数据。

说明：

- 当前 portable config 也已指向 `release/data_specs/gid15_dataset/labels.txt`
- 所以外部数据根目录即使不带 `labels.txt`，只要图像与 mask 结构完整，也可以运行

## 4. 交付包自检

先跑一次轻量检查：

```bash
bash release/scripts/check_release.sh
```

这个检查会验证：

- 关键文件是否存在
- `paddlers/unet_data_split` 是否能 `py_compile`
- portable config 是否能做 prepare-only 解析

## 5. single-head 持久化训练

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
CUDA_DEVICE=0 bash release/scripts/start_unet_train_nohup.sh singlehead
```

如需自定义输出目录：

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
CUDA_DEVICE=0 bash release/scripts/start_unet_train_nohup.sh \
  singlehead \
  /abs/path/to/save_dir
```

## 6. multi-head 持久化训练

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
CUDA_DEVICE=0 bash release/scripts/start_unet_train_nohup.sh multihead
```

## 7. 直接前台 smoke / dry-run

single-head:

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
python paddlers/unet_data_split/train.py \
  --config release/configs/unet_singlehead_portable.json \
  --run-smoke-test \
  --device gpu
```

multi-head:

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
python paddlers/unet_data_split/train.py \
  --config release/configs/unet_multihead_portable.json \
  --run-smoke-test \
  --device gpu
```

## 8. 日志位置

single-head 默认输出：

- `release/runs/unet_singlehead/logs/train.log`
- `release/runs/unet_singlehead/logs/eval.log`
- `release/runs/unet_singlehead/logs/smoke_test.log`

multi-head 默认输出：

- `release/runs/unet_multihead/logs/train.log`
- `release/runs/unet_multihead/logs/eval.log`
- `release/runs/unet_multihead/logs/smoke_test.log`
