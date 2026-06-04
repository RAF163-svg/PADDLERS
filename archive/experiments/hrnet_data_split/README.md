# HRNet 4-Head Multi-Head GID15

本实验只做 multi-head，没有做 single-head baseline。

## 数据口径

- 语义分割图像与 mask 使用 `${GID15_DATASET_ROOT}`
- cluster 路由信息使用 `${GID15_CLUSTERED_DATASET_ROOT}`
- `selected_k = 4`
- `cluster_id` 只用于 head 路由，不改变原始 15 类语义
- 每个 head 输出 `16` 通道，即 `background + 原始 15 类`
- 最终评估统一基于原始 16 类预测图计算 `OA / Mean Accuracy / mIoU / Kappa / per-class Accuracy / per-class Recall / per-class IoU / confusion matrix`

## 结构

- 共享一个 `HRNet_W48` backbone
- 顶部 4 个 `FCNHead`
- forward 时按 `cluster_id` 从 4 个 head 里选出该样本的 routed logits
- loss 只对 routed 之后的 16 类 segmentation logits 计算

## 当前默认配置

- backbone: `HRNet_W48`
- pretrained: `CITYSCAPES`
- epochs: `40`
- train_batch_size: `4`
- eval_batch_size: `1`
- learning_rate: `0.005`
- crop_size: `512`

这样选的原因是优先保证 4090D 上的 4-head HRNet 稳定起跑，不先冒进把 batch size 推到高风险区；等正式线稳定后，再做更激进的吞吐优化。

## 运行方式

先做 smoke test：

```bash
cd <PROJECT_ROOT>/paddlers/hrnet_data_split
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate paddlers
CUDA_VISIBLE_DEVICES=1 python train.py --config train_config_multihead.json --run-smoke-test --device gpu
```

通过后启动正式训练：

```bash
cd <PROJECT_ROOT>/paddlers/hrnet_data_split
bash start_train_nohup.sh
```

## 日志与输出

- 训练日志: `output/logs/train.log`
- 验证/测试日志: `output/logs/eval.log`
- smoke test 日志: `output/logs/smoke_test.log`
- launcher stdout: `output/logs/launcher.stdout`
- best model 选模依据: `val mIoU`

输出目录固定在 `output/`，包含：

- `best_model/`
- `latest_model/`
- `checkpoints/`
- `logs/`
- `config_snapshot/`
- `metrics/`
- `prediction_samples/`

## 已知风险与后续优化

- 当前实现优先稳定，数据迭代走的是 Python 级别 batch 组织，没有先上更复杂的多进程 DataLoader。
- 当前 batch size 保守取 `4`；如果后续验证显存仍有明显余量，可以继续上探。
- 当前没有 single-head baseline，因此暂时不能严格证明 multi-head 相对提升，但这不是本实验的当前目标。
