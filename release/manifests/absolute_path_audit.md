# Absolute Path Audit

下面列的是本轮扫描时对“会影响别人运行”的关键绝对路径命中，优先记录源码与配置，不展开历史 output。

## 已处理

| 文件 | 行号 | 命中内容 | 处理结果 |
|---|---:|---|---|
| `paddlers/unet_data_split/start_train_nohup.sh` | 10 | `<PROJECT_ROOT>` | 已改成脚本相对定位 `PROJECT_ROOT` |
| `paddlers/unet_data_split/start_train_nohup.sh` | 32 | `${HOME}/miniconda3/etc/profile.d/conda.sh` | 已改成 `CONDA_SH` 可配置 |
| `paddlers/unet_data_split/train.py` | 路径解析逻辑 | 相对路径原本依赖脚本目录 | 已改成按 config 文件目录解析，并支持环境变量展开 |
| `paddlers/unet_data_split/cluster_routed_dataset.py` | manifest 路径解析 | 原本默认绝对路径行值 | 已支持 release 内相对 manifest |

## 已被 release 配置替代，但原文件未改

| 文件 | 行号 | 命中内容 | 当前状态 |
|---|---:|---|---|
| `paddlers/unet_data_split/train_config_singlehead_baseline.json` | 2 | `${GID15_DATASET_ROOT}` | 本机正式配置，release 用 `release/configs/unet_singlehead_portable.json` 替代 |
| `paddlers/unet_data_split/train_config_singlehead_baseline.json` | 3 | `${GID15_CLUSTERED_DATASET_ROOT}` | 同上 |
| `paddlers/unet_data_split/train_config_multihead.json` | 2 | `${GID15_DATASET_ROOT}` | 本机正式配置，release 用 `release/configs/unet_multihead_portable.json` 替代 |
| `paddlers/unet_data_split/train_config_multihead.json` | 3 | `${GID15_CLUSTERED_DATASET_ROOT}` | 同上 |
| `paddlers/unet_data_split/README.md` | 7,8,63,65 | `${GID15_DATASET_ROOT}/...` `${HOME}/...` | 原 README 保留本机实验说明，release 文档已提供可移交版说明 |

## 保留为历史实验或辅助脚本，未统一处理

| 文件 | 行号 | 命中内容 | 当前状态 |
|---|---:|---|---|
| `paddlers/fastscnn_data_split/train_config_singlehead_baseline.json` | 2,3 | `${GID15_DATASET_ROOT}/...` | 历史对照实验，未作为当前主交付线修整 |
| `paddlers/fastscnn_data_split/train_config_multihead.json` | 2,3 | `${GID15_DATASET_ROOT}/...` | 同上 |
| `paddlers/hrnet_data_split/train_config_multihead.json` | 2,3 | `${GID15_DATASET_ROOT}/...` | 同上 |
| `paddlers/farseg_data_split/train_config_multihead.json` | 2,3 | `${GID15_DATASET_ROOT}/...` | 同上 |
| `paddlers/deeplabv3_data_split/train_config.json` | 2,3 | `${GID15_DATASET_ROOT}/...` | 同上 |
| `paddlers/deeplabv3_data_split/start_train_nohup.sh` | 13 | `${HOME}/miniconda3/...` | 同上 |
| `paddlers/deeplabv3_data_split/start_train_tmux.sh` | 21,22 | `<PROJECT_ROOT>` | 同上 |
| `paddlers/farseg_data_split/launch_formal_gpu1.sh` | 22,23,26 | `${HOME}/...` | 同上 |
| `paddlers/hrnet_data_split/start_train_nohup.sh` | 13 | `${HOME}/miniconda3/...` | 同上 |

## 明显是旧目录或本机专项实验，默认不纳入主交付

| 文件 | 行号 | 命中内容 | 当前状态 |
|---|---:|---|---|
| `paddlers/fastscnn_gid15_normal/train.py` | 42-44 | `${HOME}/...` `${GID15_DATASET_ROOT}/...` `${GID15_DATASET_ROOT}/...` | 旧实验 |
| `paddlers/hrnet_gid15_normal/train.py` | 43-45 | `${HOME}/...` `${GID15_DATASET_ROOT}/...` `${GID15_DATASET_ROOT}/...` | 旧实验 |
| `paddlers/farseg_gid15_normal/train.py` | 39-41 | `${HOME}/...` `${GID15_DATASET_ROOT}/...` `${GID15_DATASET_ROOT}/...` | 旧实验 |
| `paddlers/deeplabv3_gid_nromal/train.py` | 28-30 | `${HOME}/...` `${GID15_DATASET_ROOT}/...` `${GID15_DATASET_ROOT}/...` | 旧实验 |
| `paddlers/fastseg_gid15_nromal/train.py` | 30-32 | `${HOME}/...` `${GID15_DATASET_ROOT}/...` `${GID15_DATASET_ROOT}/...` | 旧实验 |

## DORGM 默认路径

| 文件 | 行号 | 命中内容 | 当前状态 |
|---|---:|---|---|
| `DORGM/build_gid15_dataset.py` | 41,44 | `${GID15_DATASET_ROOT}/...` `${GID15_DATASET_ROOT}/...` | 作为数据构建脚本保留默认值，未做统一重构 |
| `DORGM/build_gid15_clustered_dataset.py` | 39,40 | `${GID15_DATASET_ROOT}/...` | 同上 |
| `DORGM/build_gid15_split_dataset.py` | 24,29,30,33,659 | `${GID15_DATASET_ROOT}/...` `${HOME}/...` `${GID15_DATASET_ROOT}/...` | 同上 |
| `DORGM/audit_gid15_split_dataset.py` | 19,20,22,23,187,188,640 | `${GID15_DATASET_ROOT}/...` `${GID15_DATASET_ROOT}/...` `${HOME}/...` | 同上 |
