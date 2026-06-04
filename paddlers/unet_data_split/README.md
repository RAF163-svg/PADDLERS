# UNet Data Split

这套工程用于在当前正式 GID15 口径上落地 `single-head UNet` 与 `4-head routed UNet`。

## 当前正式口径

- 语义分割主体数据：`${GID15_DATASET_ROOT}`
- cluster 路由数据：`${GID15_CLUSTERED_DATASET_ROOT}`
- 当前 `selected_k = 4`
- `sample_path_source = input`
- `cluster_id` 只用于 head 路由，不改变原始 `background + 15` 个 GID15 类的 mask 语义
- 每个 head 都输出 `16` 通道 logits

## single-head 与 multi-head

- `single-head`:
  `num_heads=1`，`cluster_id_handling=ignore`，所有样本都走同一个共享 head
- `multi-head`:
  `num_heads=4`，按 `cluster_id` 路由到 `cluster_0..cluster_3`
- 两者共用同一套：
  数据读取、训练入口、日志体系、评估指标、checkpoint 结构

## 模型结构

- 共享一个 `UNet Encoder`
- 共享一个 `UNet Decoder`
- 顶部扩成 `num_heads` 个轻量 segmentation heads
- 不复制 4 套完整 decoder
- 最终评估统一基于原始 16 类预测图，而不是 cluster

## best model 选模

- 默认按 `val mIoU` 选择 `best_model`

## 预训练加载

- 共享 encoder / decoder trunk 优先从 PaddleRS 官方 `UNet CITYSCAPES` 权重加载
- 单头 `cls` 权重会广播到 routed heads
- 无法匹配的 head 特有参数保留随机初始化，并在日志与快照里落盘

## 关键日志

- single-head:
  `output/singlehead/logs/train.log`
  `output/singlehead/logs/eval.log`
  `output/singlehead/logs/smoke_test.log`
- multi-head:
  `output/multihead/logs/train.log`
  `output/multihead/logs/eval.log`
  `output/multihead/logs/smoke_test.log`

日志格式固定为：

```text
YYYY-MM-DD HH:MM:SS [LEVEL] [STAGE] message
```

## 常用命令

先进入环境：

```bash
source ${HOME}/miniconda3/etc/profile.d/conda.sh
conda activate paddlers
cd <PROJECT_ROOT>
```

single-head smoke test:

```bash
python paddlers/unet_data_split/train.py \
  --config paddlers/unet_data_split/train_config_singlehead_baseline.json \
  --run-smoke-test
```

single-head 持久化训练:

```bash
bash paddlers/unet_data_split/start_train_nohup.sh \
  paddlers/unet_data_split/train_config_singlehead_baseline.json \
  paddlers/unet_data_split/output/singlehead
```

multi-head smoke test:

```bash
python paddlers/unet_data_split/train.py \
  --config paddlers/unet_data_split/train_config_multihead.json \
  --run-smoke-test
```

multi-head 持久化训练:

```bash
bash paddlers/unet_data_split/start_train_nohup.sh \
  paddlers/unet_data_split/train_config_multihead.json \
  paddlers/unet_data_split/output/multihead
```

## 当前已知风险 / 后续优化项

- routed UNet 第一版先追求稳妥，不引入更复杂的 decoder 专属路由结构
- 当前默认输入裁剪是 `512`，后续可再做更大 crop 或更大 batch 的显存探测
- 当前主要使用 `val mIoU` 选模；如果后续需要，也可以加 foreground-only 监控视图
