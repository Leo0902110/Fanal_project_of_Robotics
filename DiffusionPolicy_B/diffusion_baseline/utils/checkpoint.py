# 用途: 保存和读取训练 checkpoint，包括模型、优化器、配置和归一化统计量。

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    config: Any,
    dataset: Any,
    epoch: int,
    val_loss: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": _to_serializable_dict(config),
            "epoch": epoch,
            "val_loss": val_loss,
            "obs_horizon": dataset.obs_horizon,
            "obs_dim": dataset.obs_dim,
            "pred_horizon": dataset.pred_horizon,
            "action_dim": dataset.action_dim,
            "obs_mean": dataset.stats.obs_mean.cpu(),
            "obs_std": dataset.stats.obs_std.cpu(),
            "action_mean": dataset.stats.action_mean.cpu(),
            "action_std": dataset.stats.action_std.cpu(),
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def _to_serializable_dict(config: Any) -> dict[str, Any]:
    if hasattr(config, "__dataclass_fields__"):
        raw = asdict(config)
    elif isinstance(config, dict):
        raw = dict(config)
    else:
        raw = vars(config)
    return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}
