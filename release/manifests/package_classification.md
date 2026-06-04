# Package Classification

## 必须打包

- `LICENSE`
- `setup.py`
- `requirements.txt`
- `paddlers/`
  说明：真实运行仍依赖 PaddleRS 主包源码
- `release/`
  说明：这是当前整理后的交付层
- `DORGM/*.py`
  说明：如果需要说明数据构建来源，建议一并带上

## 建议打包

- `paddlers/fastscnn_data_split/`
- `paddlers/hrnet_data_split/`
- `paddlers/farseg_data_split/`
- `paddlers/deeplabv3_data_split/`
- 选定实验的 `best_model/`
- 选定实验的 `best_model_metrics.*`

## 不建议打包

- `.git/`
- `__pycache__/`
- `output/`
- `**/output/`
- `**/*.log`
- `**/*.pdparams`
- `**/*.pdopt`
- `**/*.pdstates`
- `**/vdl_log/`
- `**/distributed_logs/`
- `release/runs/`
- `release/validation/`
