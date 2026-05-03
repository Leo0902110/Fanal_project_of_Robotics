# 用途: 定义训练脚本使用的配置数据结构，避免参数在模块间散落传递。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainConfig:
    data_path: Path | None = None
    run_dir: Path = Path("diffusion_baseline/runs/latest")
    epochs: int = 10
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 1e-6
    num_diffusion_steps: int = 100
    hidden_dim: int = 512
    cond_dim: int = 256
    num_blocks: int = 4
    dropout: float = 0.0
    val_ratio: float = 0.1
    seed: int = 42
    device: str = "cpu"
