# PaddleRS-GID15 遥感语义分割实验项目

本仓库是一个基于 [PaddleRS](https://github.com/PaddlePaddle/PaddleRS) 的 GID15 遥感语义分割实验项目。它不是原始 PaddleRS 官方 README 的简单镜像，而是把我在 GID15 数据集上做过的模型实验、routed UNet 主线代码、可移植配置和实验结果整理成一个适合复现和继续开发的代码仓库。

当前主线代码位于：

```text
paddlers/unet_data_split/
```

完整实验结果见：[docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md)

机器可读结果表见：[docs/experiment_results_summary.tsv](docs/experiment_results_summary.tsv)

## 1. 项目简介

本项目围绕 GID15 遥感语义分割任务，整理并对比了多组模型实验，包括：

- UNet
- DeepLabV3 / DeepLabV3+
- FarSeg
- FastSCNN
- FastSeg
- HRNet
- GID15 normal baseline
- cluster-routed / data-split 实验
- single-head routed UNet
- multi-head routed UNet

当前主要交付线是 `paddlers/unet_data_split/` 下的 routed UNet。它包含 single-head baseline 和 multi-head routed 两种设置，配套了训练、评估、推理、指标统计和 portable config。

## 2. 我在原 PaddleRS 基础上做了什么

- 新增 GID15 数据集实验流程。
- 新增 cluster-routed 数据划分与加载逻辑。
- 新增 single-head / multi-head routed UNet 模型。
- 新增训练、评估、推理脚本。
- 整理 UNet、DeepLabV3+、FarSeg、FastSCNN、FastSeg、HRNet 等多组模型实验结果。
- 提供 portable config，避免运行时依赖本机绝对路径。
- 将模型权重、checkpoint 和训练输出从代码仓库中剥离，方便 GitHub 代码管理。

## 3. 仓库内容

```text
PaddleRS-GID15/
├── paddlers/unet_data_split/      # 当前主线实验代码
├── release/configs/               # 可移植训练配置
├── release/scripts/               # 训练和检查脚本
├── release/docs/                  # 运行说明
├── docs/EXPERIMENT_RESULTS.md     # 完整实验结果
├── docs/experiment_results_summary.tsv
├── paddlers/                      # PaddleRS 核心源码
├── tools/                         # 工具脚本
├── tests/                         # 测试代码
└── README.md
```

## 4. 实验结果总览

下面只放 README 首页摘要。所有数值均来自 `docs/experiment_results_summary.tsv`，没有补填或估算。更多历史实验、smoke/probe 记录和指标来源请看 [完整实验结果](docs/EXPERIMENT_RESULTS.md)。

| 模型 | 实验类型 | mIoU | Accuracy | Kappa | best model | 备注 |
|---|---|---:|---:|---:|---|---|
| UNet | multi-head routed | 0.383720 | 0.723863 | 0.626882 | 是 | 当前主线 |
| UNet | single-head routed | 0.378758 | 0.724935 | 0.630498 | 是 | 当前主线 |
| DeepLabV3+ | GID15 normal baseline | 0.526867 | 0.786168 | 0.732505 | 是 | 历史对照 |
| FarSeg | GID15 normal baseline | 0.507928 | 0.757180 | 0.697622 | 是 | 历史对照 |
| FastSCNN | GID15 normal baseline | 0.469304 | 0.736937 | 0.668483 | 是 | 历史对照 |
| HRNet | GID15 normal baseline | 0.430103 | 0.722239 | 0.649383 | 是 | 历史对照 |
| HRNet | stable baseline | 0.474284 | 0.714009 | 0.617053 | 是 | 历史对照 |
| FastSeg | GID15 normal baseline | 0.129983 | 0.465505 | 0.188082 | 是 | 历史对照 |

说明：

- README 只展示 8 条主要结果，避免首页过长。
- `当前主线` 指本仓库当前最完整、最建议继续维护的 `paddlers/unet_data_split/` 代码线。
- 完整表格中还有 cluster-routed data-split、archive、smoke/probe 等记录；这些结果请以 [docs/EXPERIMENT_RESULTS.md](docs/EXPERIMENT_RESULTS.md) 中的来源说明为准。

## 5. 当前主线：Routed UNet

当前主要代码在：

```text
paddlers/unet_data_split/
```

核心文件：

- `train.py`：训练入口，支持 single-head / multi-head 配置。
- `eval_best_model.py`：评估 `best_model`。
- `infer_best_model.py`：导出推理样例。
- `cluster_routed_dataset.py`：cluster-routed 数据集加载逻辑。
- `unet_routed_model.py`：single-head / multi-head routed UNet。
- `metrics_utils.py`：指标计算、日志和结果导出工具。

配套 release 文件：

- `release/configs/unet_singlehead_portable.json`
- `release/configs/unet_multihead_portable.json`
- `release/scripts/check_release.sh`
- `release/scripts/start_unet_train_nohup.sh`
- `release/docs/RUN_TRAIN.md`
- `release/docs/RUN_INFER.md`

## 6. 模型权重说明

本 GitHub 仓库不包含训练好的模型权重和训练输出。

未上传的内容包括：

- `*.pdparams`
- `*.pdopt`
- `*.pdstates`
- `*.pth`
- `*.onnx`
- `best_model/`
- `latest_model/`
- `checkpoints/`
- `output/`
- `runs/`

原因很简单：这些文件体积较大，不适合直接放进普通 GitHub 代码仓库。代码、配置和结果摘要放在 GitHub；模型资产应该单独发布。

后续模型权重可以单独整理并发布到：

- GitHub Release
- Hugging Face
- 网盘
- 服务器下载链接

下载权重后，建议放置到：

```text
release/runs/unet_singlehead/best_model/
release/runs/unet_multihead/best_model/
```

## 7. 环境安装

推荐使用 conda 环境文件：

```bash
git clone https://github.com/RAF163-svg/PADDLERS.git
cd PADDLERS

conda env create -f release/env/environment.yml
conda activate paddlers
pip install -e .
```

如果已经有可用的 Paddle/PaddleRS 环境，也可以用基础方式安装：

```bash
pip install -r requirements.txt
pip install -e .
```

环境说明：

- `release/env/environment.yml` 是当前 clean 版本保留的可复现环境参考。
- GPU、CUDA、PaddlePaddle 版本需要和你的机器匹配。
- `release/env/requirements-freeze.txt` 更适合作为依赖审计参考，不一定适合直接跨机器安装。

## 8. 数据准备

训练、评估和推理都需要准备 GID15 语义分割数据集，并设置环境变量：

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
```

数据根目录至少应包含：

```text
gid15_dataset/
├── images/
└── masks/
```

当前 release 已带有轻量元数据和标签说明，例如：

- `release/data_specs/gid15_dataset/labels.txt`
- cluster routing 相关元数据

因此，运行主线 routed UNet 时不需要把大体量数据复制进仓库，只需要在本机准备好正式数据目录并设置 `GID15_DATASET_ROOT`。

## 9. 运行训练

先做一次 release 自检：

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
bash release/scripts/check_release.sh
```

single-head 训练：

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
CUDA_DEVICE=0 bash release/scripts/start_unet_train_nohup.sh singlehead
```

multi-head 训练：

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
CUDA_DEVICE=0 bash release/scripts/start_unet_train_nohup.sh multihead
```

前台 smoke test：

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
python paddlers/unet_data_split/train.py \
  --config release/configs/unet_singlehead_portable.json \
  --run-smoke-test \
  --device gpu
```

multi-head smoke test 时把配置换成：

```text
release/configs/unet_multihead_portable.json
```

更完整的训练说明见：[release/docs/RUN_TRAIN.md](release/docs/RUN_TRAIN.md)

## 10. 运行评估和推理

评估 single-head `best_model`：

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
python paddlers/unet_data_split/eval_best_model.py \
  --config release/configs/unet_singlehead_portable.json \
  --save-dir release/runs/unet_singlehead \
  --device gpu
```

推理 single-head 样例：

```bash
export GID15_DATASET_ROOT=/path/to/gid15_dataset
python paddlers/unet_data_split/infer_best_model.py \
  --config release/configs/unet_singlehead_portable.json \
  --save-dir release/runs/unet_singlehead \
  --split test \
  --count 12 \
  --device gpu
```

multi-head 评估和推理时使用：

```text
release/configs/unet_multihead_portable.json
release/runs/unet_multihead
```

更完整的评估和推理说明见：[release/docs/RUN_INFER.md](release/docs/RUN_INFER.md)

## 11. 复现实验需要准备什么

如果要复现 README 中的主线 routed UNet 实验，至少需要：

- 本仓库代码。
- 可用的 Paddle/PaddleRS 环境。
- GID15 数据集，并设置 `GID15_DATASET_ROOT`。
- `release/configs/` 中的 portable config。
- 如需直接评估或推理，还需要单独下载并放置 `best_model/` 权重。

如果要复现历史对照实验，例如 DeepLabV3+、FarSeg、FastSCNN、FastSeg、HRNet，需要参考 [docs/experiment_results_summary.tsv](docs/experiment_results_summary.tsv) 中记录的配置来源、实验类型、指标来源和备注。部分历史目录只保留了结果摘要或混淆矩阵，未必能仅凭本仓库完全恢复训练过程。

## 12. PaddleRS 来源说明

本项目基于 PaddleRS 进行二次实验开发。PaddleRS 是百度飞桨生态中的遥感影像智能解译开发套件，原项目支持图像分割、目标检测、场景分类、变化检测、图像复原等遥感任务。

本仓库重点不是替代 PaddleRS 官方仓库，而是整理 GID15 语义分割实验代码和结果。PaddleRS 原始项目请参考：

- [PaddlePaddle/PaddleRS](https://github.com/PaddlePaddle/PaddleRS)

## 13. License

本仓库保留 PaddleRS 原项目的 Apache 2.0 许可证。详见 [LICENSE](LICENSE)。
