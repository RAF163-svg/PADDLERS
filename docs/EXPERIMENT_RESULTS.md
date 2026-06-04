# 实验结果汇总

## 1. 项目说明

本报告汇总原始工作目录中已经运行过的 GID15 语义分割实验结果。代码仓库只保留可公开的代码与说明文档；训练输出目录、模型权重、优化器状态、预测样例和大文件不进入 GitHub。

本次扫描只读取实验目录中的配置、状态、指标 JSON/CSV 和模型说明文件，没有复制任何权重文件。报告中的路径均为相对于原始项目根目录的相对路径。

## 2. 实验目录说明

| 实验目录 | 模型族 | 识别到的实验行数 | 指标情况 |
| --- | --- | ---: | --- |
| `paddlers/unet_data_split` | UNet | 2 | 有 |
| `paddlers/deeplabv3_data_split` | DeepLabV3+ | 1 | 有 |
| `paddlers/farseg_data_split` | FarSeg | 24 | 有 |
| `paddlers/fastscnn_data_split` | FastSCNN | 11 | 有 |
| `paddlers/hrnet_data_split` | HRNet | 1 | 有 |
| `paddlers/deeplabv3_gid_nromal` | DeepLabV3+ | 2 | 有 |
| `paddlers/farseg_gid15_normal` | FarSeg | 1 | 有 |
| `paddlers/fastscnn_gid15_normal` | FastSCNN | 1 | 有 |
| `paddlers/fastseg_gid15_nromal` | FactSeg/FastSeg | 2 | 有 |
| `paddlers/hrnet_gid15_normal` | HRNet | 2 | 有 |

## 3. 模型实验总览

- 扫描目录数：10
- 识别实验行数：47
- 模型族数量：6（DeepLabV3+, FactSeg/FastSeg, FarSeg, FastSCNN, HRNet, UNet）
- 有可提取整体指标的实验：15
- 暂未找到整体指标的实验：32
- 建议作为公开结果引用的实验：8

| 实验 | 模型族 | 类型 | mIoU | OA/accuracy | Kappa | 建议 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| deeplabv3_gid_nromal/output | DeepLabV3+ | GID15 normal baseline | 0.526867 | 0.786168 | 0.732505 | yes |
| fastseg_gid15_nromal/output | FactSeg/FastSeg | GID15 normal baseline | 0.129983 | 0.465505 | 0.188082 | yes |
| farseg_gid15_normal/output | FarSeg | GID15 normal baseline | 0.507928 | 0.757180 | 0.697622 | yes |
| fastscnn_gid15_normal/output | FastSCNN | GID15 normal baseline | 0.469304 | 0.736937 | 0.668483 | yes |
| hrnet_gid15_normal/output | HRNet | GID15 normal baseline | 0.430103 | 0.722239 | 0.649383 | yes |
| hrnet_gid15_normal/output_stable | HRNet | GID15 normal baseline | 0.474284 | 0.714009 | 0.617053 | yes |
| unet_data_split/multihead | UNet | cluster-routed multihead | 0.383720 | 0.723863 | 0.626882 | yes |
| unet_data_split/singlehead | UNet | cluster-routed singlehead | 0.378758 | 0.724935 | 0.630498 | yes |
| deeplabv3_data_split/output | DeepLabV3+ | cluster-routed data_split | 0.025525 | 0.025073 | - | review |
| deeplabv3_gid_nromal/archive_oversample_mixloss_20260325_165404 | DeepLabV3+ | archive baseline | 0.451574 | 0.712147 | 0.615652 | review |
| fastseg_gid15_nromal/archive_restart_20260326_161955 | FactSeg/FastSeg | archive baseline | 0.169774 | 0.436000 | 0.149374 | review |
| farseg_data_split/output | FarSeg | cluster-routed data_split | 0.029716 | 0.109487 | - | review |
| farseg_data_split/gpu_smoke | FarSeg | smoke/probe | 0.095989 | 0.359648 | - | review |
| fastscnn_data_split/acceptance_k4_20260405_174500 | FastSCNN | formal routed experiment | 0.064097 | 0.320308 | - | review |
| hrnet_data_split/output | HRNet | cluster-routed data_split | 0.434319 | 0.756966 | 0.674375 | review |
| farseg_data_split/batch_probes_20260405_1139/bs02 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_1139/bs04 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_1139/bs06 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_1139/bs08 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_1139/bs10 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_1139/bs12 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_1139/bs14 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_1139/bs16 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_1139/bs18 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_1139/bs20 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_115054/bs22 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_115054/bs24 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_115054/bs26 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_115054/bs28 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/batch_probes_20260405_115054/bs30 | FarSeg | smoke/probe | - | - | - | no |
| farseg_data_split/formal_gpu1_bs20_20260405_122442 | FarSeg | formal routed experiment | - | - | - | no |
| farseg_data_split/formal_gpu1_bs22_20260405_120445 | FarSeg | formal routed experiment | - | - | - | no |
| farseg_data_split/formal_gpu1_bs22_20260405_120731 | FarSeg | formal routed experiment | - | - | - | no |
| farseg_data_split/stability_bs20_20260405_121138 | FarSeg | stability probe | - | - | - | no |
| farseg_data_split/stability_bs22_20260405_120140 | FarSeg | stability probe | - | - | - | no |
| farseg_data_split/stability_bs24_20260405_115919 | FarSeg | stability probe | - | - | - | no |
| farseg_data_split/stability_bs26_20260405_115737 | FarSeg | stability probe | - | - | - | no |
| fastscnn_data_split/output | FastSCNN | cluster-routed data_split | - | - | - | no |
| fastscnn_data_split/formal_gpu1_bs32_20260405_191241 | FastSCNN | formal routed experiment | - | - | - | no |
| fastscnn_data_split/gpu1_mem_probe_20260405_190210/probe_bs04 | FastSCNN | smoke/probe | - | - | - | no |
| fastscnn_data_split/gpu1_mem_probe_20260405_190210/probe_bs08 | FastSCNN | smoke/probe | - | - | - | no |
| fastscnn_data_split/gpu1_mem_probe_20260405_190210/probe_bs12 | FastSCNN | smoke/probe | - | - | - | no |
| fastscnn_data_split/gpu1_mem_probe_20260405_190210/probe_bs16 | FastSCNN | smoke/probe | - | - | - | no |
| fastscnn_data_split/gpu1_mem_probe_20260405_190210/probe_bs24 | FastSCNN | smoke/probe | - | - | - | no |
| fastscnn_data_split/gpu1_mem_probe_20260405_190210/probe_bs32 | FastSCNN | smoke/probe | - | - | - | no |
| fastscnn_data_split/gpu1_mem_probe_20260405_190210/probe_bs36_boundary | FastSCNN | smoke/probe | - | - | - | no |
| fastscnn_data_split/gpu1_mem_probe_20260405_190210/probe_bs40_boundary | FastSCNN | smoke/probe | - | - | - | no |

完整机器可读汇总见 [`experiment_results_summary.tsv`](./experiment_results_summary.tsv)。

## 4. 主要结果对比

以下只列出建议公开引用或可人工复核的主要结果；`review` 表示可作为历史对照线索，但不建议直接作为主结果。

| 实验 | 模型族 | 类型 | mIoU | OA/accuracy | Kappa | 指标来源 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| deeplabv3_gid_nromal/output | DeepLabV3+ | GID15 normal baseline | 0.526867 | 0.786168 | 0.732505 | `paddlers/deeplabv3_gid_nromal/output/best_model/eval_details.json` |
| fastseg_gid15_nromal/output | FactSeg/FastSeg | GID15 normal baseline | 0.129983 | 0.465505 | 0.188082 | `paddlers/fastseg_gid15_nromal/output/best_model_metrics.json` |
| farseg_gid15_normal/output | FarSeg | GID15 normal baseline | 0.507928 | 0.757180 | 0.697622 | `paddlers/farseg_gid15_normal/output/best_model/eval_details.json` |
| fastscnn_gid15_normal/output | FastSCNN | GID15 normal baseline | 0.469304 | 0.736937 | 0.668483 | `paddlers/fastscnn_gid15_normal/output/best_model/eval_details.json` |
| hrnet_gid15_normal/output | HRNet | GID15 normal baseline | 0.430103 | 0.722239 | 0.649383 | `paddlers/hrnet_gid15_normal/output/best_model/eval_details.json` |
| hrnet_gid15_normal/output_stable | HRNet | GID15 normal baseline | 0.474284 | 0.714009 | 0.617053 | `paddlers/hrnet_gid15_normal/output_stable/best_model_metrics.json` |
| unet_data_split/multihead | UNet | cluster-routed multihead | 0.383720 | 0.723863 | 0.626882 | `paddlers/unet_data_split/output/multihead/best_model_metrics.json` |
| unet_data_split/singlehead | UNet | cluster-routed singlehead | 0.378758 | 0.724935 | 0.630498 | `paddlers/unet_data_split/output/singlehead/best_model_metrics.json` |
| deeplabv3_data_split/output | DeepLabV3+ | cluster-routed data_split | 0.025525 | 0.025073 | - | `paddlers/deeplabv3_data_split/output/metrics/smoke_test_metrics.json` |
| deeplabv3_gid_nromal/archive_oversample_mixloss_20260325_165404 | DeepLabV3+ | archive baseline | 0.451574 | 0.712147 | 0.615652 | `paddlers/deeplabv3_gid_nromal/output/archive_oversample_mixloss_20260325_165404/best_model_metrics.json` |
| fastseg_gid15_nromal/archive_restart_20260326_161955 | FactSeg/FastSeg | archive baseline | 0.169774 | 0.436000 | 0.149374 | `paddlers/fastseg_gid15_nromal/output/archive_restart_20260326_161955/best_model_metrics.json` |
| farseg_data_split/output | FarSeg | cluster-routed data_split | 0.029716 | 0.109487 | - | `paddlers/farseg_data_split/output/metrics/smoke_test_metrics.json` |
| farseg_data_split/gpu_smoke | FarSeg | smoke/probe | 0.095989 | 0.359648 | - | `paddlers/farseg_data_split/output/gpu_smoke/metrics/smoke_test_metrics.json` |
| fastscnn_data_split/acceptance_k4_20260405_174500 | FastSCNN | formal routed experiment | 0.064097 | 0.320308 | - | `paddlers/fastscnn_data_split/output/acceptance_k4_20260405_174500/metrics/smoke_test_metrics.json` |
| hrnet_data_split/output | HRNet | cluster-routed data_split | 0.434319 | 0.756966 | 0.674375 | `paddlers/hrnet_data_split/output/metrics/val_epoch_035.json` |

## 5. UNet Routed 主线实验

UNet 目录是当前 clean 仓库保留的主线代码来源，包含 singlehead baseline 与 multihead routed 两条训练输出记录。模型权重未进入仓库，指标来源于原始输出中的 `best_model_metrics.json` 或 `metrics/best_model_val_metrics.json`。

- `paddlers/unet_data_split/output/multihead`：UNet，cluster-routed multihead，mIoU=0.383720，OA=0.723863，来源 `paddlers/unet_data_split/output/multihead/best_model_metrics.json`。
- `paddlers/unet_data_split/output/singlehead`：UNet，cluster-routed singlehead，mIoU=0.378758，OA=0.724935，来源 `paddlers/unet_data_split/output/singlehead/best_model_metrics.json`。

## 6. 历史对照实验

以下模型族作为历史对照或实验参考保留结果摘要。包含 `archive`、`stability`、`smoke/probe` 标记的行不建议直接作为论文/README 主结果，除非后续人工确认训练条件一致。

- `paddlers/deeplabv3_gid_nromal/output`：DeepLabV3+，GID15 normal baseline，mIoU=0.526867，OA=0.786168，来源 `paddlers/deeplabv3_gid_nromal/output/best_model/eval_details.json`。
- `paddlers/fastseg_gid15_nromal/output`：FactSeg/FastSeg，GID15 normal baseline，mIoU=0.129983，OA=0.465505，来源 `paddlers/fastseg_gid15_nromal/output/best_model_metrics.json`。
- `paddlers/farseg_gid15_normal/output`：FarSeg，GID15 normal baseline，mIoU=0.507928，OA=0.757180，来源 `paddlers/farseg_gid15_normal/output/best_model/eval_details.json`。
- `paddlers/fastscnn_gid15_normal/output`：FastSCNN，GID15 normal baseline，mIoU=0.469304，OA=0.736937，来源 `paddlers/fastscnn_gid15_normal/output/best_model/eval_details.json`。
- `paddlers/hrnet_gid15_normal/output`：HRNet，GID15 normal baseline，mIoU=0.430103，OA=0.722239，来源 `paddlers/hrnet_gid15_normal/output/best_model/eval_details.json`。
- `paddlers/hrnet_gid15_normal/output_stable`：HRNet，GID15 normal baseline，mIoU=0.474284，OA=0.714009，来源 `paddlers/hrnet_gid15_normal/output_stable/best_model_metrics.json`。
- `paddlers/deeplabv3_data_split/output`：DeepLabV3+，cluster-routed data_split，mIoU=0.025525，OA=0.025073，来源 `paddlers/deeplabv3_data_split/output/metrics/smoke_test_metrics.json`。
- `paddlers/deeplabv3_gid_nromal/output/archive_oversample_mixloss_20260325_165404`：DeepLabV3+，archive baseline，mIoU=0.451574，OA=0.712147，来源 `paddlers/deeplabv3_gid_nromal/output/archive_oversample_mixloss_20260325_165404/best_model_metrics.json`。
- `paddlers/fastseg_gid15_nromal/output/archive_restart_20260326_161955`：FactSeg/FastSeg，archive baseline，mIoU=0.169774，OA=0.436000，来源 `paddlers/fastseg_gid15_nromal/output/archive_restart_20260326_161955/best_model_metrics.json`。
- `paddlers/farseg_data_split/output`：FarSeg，cluster-routed data_split，mIoU=0.029716，OA=0.109487，来源 `paddlers/farseg_data_split/output/metrics/smoke_test_metrics.json`。
- `paddlers/farseg_data_split/output/gpu_smoke`：FarSeg，smoke/probe，mIoU=0.095989，OA=0.359648，来源 `paddlers/farseg_data_split/output/gpu_smoke/metrics/smoke_test_metrics.json`。
- `paddlers/fastscnn_data_split/output/acceptance_k4_20260405_174500`：FastSCNN，formal routed experiment，mIoU=0.064097，OA=0.320308，来源 `paddlers/fastscnn_data_split/output/acceptance_k4_20260405_174500/metrics/smoke_test_metrics.json`。
- `paddlers/hrnet_data_split/output`：HRNet，cluster-routed data_split，mIoU=0.434319，OA=0.756966，来源 `paddlers/hrnet_data_split/output/metrics/val_epoch_035.json`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_1139/bs02`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_1139/bs04`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_1139/bs06`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_1139/bs08`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_1139/bs10`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_1139/bs12`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_1139/bs14`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_1139/bs16`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_1139/bs18`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_1139/bs20`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_115054/bs22`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_115054/bs24`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_115054/bs26`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_115054/bs28`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/batch_probes_20260405_115054/bs30`：FarSeg，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/formal_gpu1_bs20_20260405_122442`：FarSeg，formal routed experiment，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/formal_gpu1_bs22_20260405_120445`：FarSeg，formal routed experiment，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/formal_gpu1_bs22_20260405_120731`：FarSeg，formal routed experiment，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/stability_bs20_20260405_121138`：FarSeg，stability probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/stability_bs22_20260405_120140`：FarSeg，stability probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/stability_bs24_20260405_115919`：FarSeg，stability probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/farseg_data_split/output/stability_bs26_20260405_115737`：FarSeg，stability probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/fastscnn_data_split/output`：FastSCNN，cluster-routed data_split，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/fastscnn_data_split/output/formal_gpu1_bs32_20260405_191241`：FastSCNN，formal routed experiment，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/fastscnn_data_split/output/gpu1_mem_probe_20260405_190210/probe_bs04`：FastSCNN，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/fastscnn_data_split/output/gpu1_mem_probe_20260405_190210/probe_bs08`：FastSCNN，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/fastscnn_data_split/output/gpu1_mem_probe_20260405_190210/probe_bs12`：FastSCNN，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/fastscnn_data_split/output/gpu1_mem_probe_20260405_190210/probe_bs16`：FastSCNN，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/fastscnn_data_split/output/gpu1_mem_probe_20260405_190210/probe_bs24`：FastSCNN，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/fastscnn_data_split/output/gpu1_mem_probe_20260405_190210/probe_bs32`：FastSCNN，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/fastscnn_data_split/output/gpu1_mem_probe_20260405_190210/probe_bs36_boundary`：FastSCNN，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。
- `paddlers/fastscnn_data_split/output/gpu1_mem_probe_20260405_190210/probe_bs40_boundary`：FastSCNN，smoke/probe，mIoU=未记录，OA=未记录，来源 `未找到`。

## 7. 模型权重说明

本 GitHub 仓库不包含训练好的模型权重，也不包含 optimizer state、checkpoint、`best_model` 或 `latest_model` 目录。

模型权重应通过 Release、网盘或其他资产渠道单独发布。下载后按需要放置到：

- `release/runs/unet_singlehead/best_model/`
- `release/runs/unet_multihead/best_model/`

放置权重后，再运行 `infer_best_model.py` 或对应 release 脚本进行推理。

## 8. 如何复现实验

建议复现实验时使用 clean 仓库中的 release 主线代码，并准备 GID15 数据集与单独发布的模型权重。训练类复现实验需要根据 `docs/experiment_results_summary.tsv` 中记录的配置来源，对照原始 `train_config*.json` 或 `config_snapshot/resolved_train_config.json` 设置 batch size、epoch、crop size、路由 head 数等参数。

已发布代码仓库不包含原始训练输出目录，因此不能仅靠 GitHub 仓库恢复 checkpoint 历史；需要从独立模型资产包或原始训练环境补齐权重。

## 9. 结果来源与可信度说明

- 直接指标：优先来自 `best_model_metrics.json`、`accuracy_summary.json`、`metrics/best_model_val_metrics.json`、`training_status.json` 中的整体指标。
- 计算指标：部分历史 baseline 只保存 `eval_details.json` 的混淆矩阵，报告据此计算 OA、mean IoU、Kappa、macro F1、macro precision、macro recall。
- 不确定项：没有整体指标且只有 smoke/probe 或失败状态的目录，已标记为 `no` 或 `review`，不建议作为公开主结果。
- 路径说明：报告仅保留相对路径，避免泄露本机数据路径。
