# custom_code

这里放的是“当前仓库里与交付最相关的自定义源码快照”。

用途：

- 方便审计哪些源码是本仓库二次开发新增的
- 方便对方快速定位需要重点接管的目录
- 方便在不翻完整仓库的情况下查看定制逻辑

注意：

- 这里不是独立可运行副本
- 真正运行时仍以仓库原始路径为准，例如：
  - `paddlers/unet_data_split/train.py`
  - `paddlers/unet_data_split/eval_best_model.py`
- 当前最建议接手的源码主线是：
  - `custom_code/paddlers/unet_data_split/`
  - `custom_code/DORGM/`
