# DEPENDENCY_AUDIT

## 当前结论

当前仓库要“发给别人还能跑”，最需要解决的不是算法本身，而是三类本机耦合：

1. 绝对路径
2. 外部数据根目录
3. 输出物与环境快照混杂在源码目录里

这次 release 已重点把 `paddlers/unet_data_split` 整理成当前主交付线。

## 必须打包

- 仓库源码本身
- `paddlers/unet_data_split/*.py`
- `release/configs/*.json`
- `release/docs/*`
- `release/env/environment.yml`
- `release/env/requirements-freeze.txt`
- `release/data_specs/gid15_dataset/labels.txt`
- `release/data_specs/gid15_dataset/stats/dataset_summary.json`
- `release/data_specs/gid15_clustered_dataset/stats/cluster_summary.json`
- `release/data_specs/gid15_clustered_dataset/manifests/sample_cluster_mapping.relative.csv`
- `release/scripts/start_unet_train_nohup.sh`
- `release/scripts/check_release.sh`

## 建议打包

- `DORGM/*.py`
- `DORGM/README.md`
- `paddlers/fastscnn_data_split/*.py`
- `paddlers/hrnet_data_split/*.py`
- `paddlers/farseg_data_split/*.py`
- `paddlers/deeplabv3_data_split/*.py`
- 如果要支持开箱即用推理：
  - 选定的 `best_model/`
  - 对应的 `best_model_metrics.*`

## 不建议打包

- `.git/`
- `__pycache__/`
- `output/`
- `**/output/`
- 中间 checkpoint
- 临时日志
- 历史试验快照
- 明显只属于本机运行历史的 `train.pid`、`nohup` 日志、`vdl_log/`

## 当前已处理的可移交问题

- `paddlers/unet_data_split/start_train_nohup.sh`
  - 已去掉对固定 `<PROJECT_ROOT>` 的依赖
  - 已改成脚本相对定位项目根目录
  - conda 初始化路径改成可配置
- `paddlers/unet_data_split/train.py`
  - 路径解析已支持环境变量展开
  - 配置里的相对路径改为相对 config 文件本身解析
- `paddlers/unet_data_split/cluster_routed_dataset.py`
  - 已支持 release 内的相对 routed manifest

## 当前没有统一处理的部分

这些仍然保留本机默认路径，只做审计，不作为当前主交付线：

- `paddlers/fastscnn_data_split`
- `paddlers/hrnet_data_split`
- `paddlers/farseg_data_split`
- `paddlers/deeplabv3_data_split`
- 各类 `*_gid15_normal` / `*_gid15_nromal` 历史目录
- `DORGM` 里的默认输入输出根目录常量

详细文件与行号见：

- [absolute_path_audit.md](<PROJECT_ROOT>/release/manifests/absolute_path_audit.md)
