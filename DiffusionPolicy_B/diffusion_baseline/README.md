# Diffusion Policy Baseline

用途: 说明 `diffusion_baseline` 目录内基线工程的结构、数据格式和运行方式。

这是一个轻量 PyTorch Diffusion Policy 基线工程，所有代码和子目录均位于 `diffusion_baseline/` 下。

## 目录结构

```text
diffusion_baseline/
  data/sequence_dataset.py
  diffusion/scheduler.py
  evaluation/sample_policy.py
  models/encoder.py
  models/diffusion_net.py
  training/train_diffusion.py
  utils/checkpoint.py
  utils/config.py
  utils/seed.py
```

## 数据格式

训练脚本默认读取 `.npz` 文件，要求包含:

- `observations`: shape `[N, obs_horizon, obs_dim]`
- `actions`: shape `[N, pred_horizon, action_dim]`

如果没有传入 `--data-path`，脚本会使用合成数据跑通训练流程，便于先检查环境。

## 训练

在项目根目录运行:

```bash
python -m diffusion_baseline.training.train_diffusion
```

使用真实数据:

```bash
python -m diffusion_baseline.training.train_diffusion --data-path path/to/demo.npz
```

## 采样

```bash
python -m diffusion_baseline.evaluation.sample_policy \
  --checkpoint diffusion_baseline/runs/latest/checkpoint.pt \
  --obs path/to/obs.npy
```
