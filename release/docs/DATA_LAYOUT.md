# DATA_LAYOUT

## 1. 当前 release 自带了什么

release 自带的是“轻量元数据”，不是完整训练数据。

已内置：

- `release/data_specs/gid15_dataset/labels.txt`
- `release/data_specs/gid15_dataset/stats/dataset_summary.json`
- `release/data_specs/gid15_dataset/lists/train.txt`
- `release/data_specs/gid15_dataset/lists/val.txt`
- `release/data_specs/gid15_dataset/lists/test.txt`
- `release/data_specs/gid15_clustered_dataset/stats/cluster_summary.json`
- `release/data_specs/gid15_clustered_dataset/manifests/sample_cluster_mapping.relative.csv`
- `release/data_specs/gid15_clustered_dataset/manifests/pair_checks.json`

## 2. 外部仍需准备的数据

必须准备正式语义分割数据根目录，并通过环境变量指定：

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
```

建议目录结构：

```text
gid15_dataset/
  images/
    train/
    val/
    test/
  masks/
    train/
    val/
    test/
  lists/
    train.txt
    val.txt
    test.txt
  stats/
    dataset_summary.json
```

说明：

- `labels.txt`、`lists/*.txt`、`stats/dataset_summary.json` 最好一起保留
- 但当前 release 训练主线真正强依赖的是图像、mask，以及 release 内自带的 routed metadata 与 labels 副本

## 3. routed UNet 当前实际使用的数据口径

- 图像与 mask 主体：`GID15_DATASET_ROOT`
- cluster 路由元信息：`release/data_specs/gid15_clustered_dataset`
- `sample_path_source=input`
- `cluster_id` 只用于 head 路由，不改变原始 16 类语义

## 4. 为什么不需要完整 clustered 图像目录

当前 `paddlers/unet_data_split` 训练线用的是：

- 原始 `input_image`
- 原始 `input_mask`
- 额外读取 `cluster_id`

也就是说，训练时不读 clustered 输出图像和 clustered 输出 mask。

因此，当前 release 只打包了 routed manifest 与 cluster summary，没有打包整套 `gid15_clustered_dataset` 图像与 mask。

## 5. sample_cluster_mapping.relative.csv 的设计

这个 manifest 已从本机绝对路径改成相对路径字段，便于移交：

- `input_image_rel`
- `input_mask_rel`
- `output_image_rel`
- `output_mask_rel`

同时保留原有字段名，但值也已改成相对路径，供可移交版 `cluster_routed_dataset.py` 解析。
