# 用途: 加载观测-动作序列数据，并提供无需真实数据即可调试流程的合成数据集。

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DatasetStats:
    obs_mean: torch.Tensor
    obs_std: torch.Tensor
    action_mean: torch.Tensor
    action_std: torch.Tensor


class SequenceDataset(Dataset):
    """Dataset for fixed-horizon observation and action sequences."""

    def __init__(
        self,
        observations: np.ndarray,
        actions: np.ndarray,
        normalize: bool = True,
        stats: Optional[DatasetStats] = None,
    ) -> None:
        if observations.ndim != 3:
            raise ValueError("observations must have shape [N, obs_horizon, obs_dim]")
        if actions.ndim != 3:
            raise ValueError("actions must have shape [N, pred_horizon, action_dim]")
        if observations.shape[0] != actions.shape[0]:
            raise ValueError("observations and actions must have the same first dimension")

        self.observations = torch.as_tensor(observations, dtype=torch.float32)
        self.actions = torch.as_tensor(actions, dtype=torch.float32)
        self.normalize = normalize
        self.stats = stats or self._compute_stats()

    @classmethod
    def from_npz(
        cls,
        path: str | Path,
        normalize: bool = True,
        stats: Optional[DatasetStats] = None,
    ) -> "SequenceDataset":
        data = np.load(Path(path))
        if "observations" not in data or "actions" not in data:
            raise KeyError("npz file must contain 'observations' and 'actions'")
        return cls(data["observations"], data["actions"], normalize=normalize, stats=stats)

    @property
    def obs_horizon(self) -> int:
        return int(self.observations.shape[1])

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[2])

    @property
    def pred_horizon(self) -> int:
        return int(self.actions.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[2])

    def _compute_stats(self) -> DatasetStats:
        obs_std = self.observations.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        action_std = self.actions.std(dim=(0, 1), keepdim=True).clamp_min(1e-6)
        return DatasetStats(
            obs_mean=self.observations.mean(dim=(0, 1), keepdim=True),
            obs_std=obs_std,
            action_mean=self.actions.mean(dim=(0, 1), keepdim=True),
            action_std=action_std,
        )

    def __len__(self) -> int:
        return int(self.observations.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        obs = self.observations[idx]
        action = self.actions[idx]
        if self.normalize:
            obs = (obs - self.stats.obs_mean.squeeze(0)) / self.stats.obs_std.squeeze(0)
            action = (action - self.stats.action_mean.squeeze(0)) / self.stats.action_std.squeeze(0)
        return {"obs": obs, "action": action}


def make_synthetic_dataset(
    num_samples: int = 512,
    obs_horizon: int = 2,
    pred_horizon: int = 16,
    obs_dim: int = 8,
    action_dim: int = 4,
    seed: int = 0,
) -> SequenceDataset:
    """Create a small correlated dataset for smoke-testing the training loop."""

    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(num_samples, obs_horizon, obs_dim)).astype(np.float32)
    weights = rng.normal(size=(obs_dim, action_dim)).astype(np.float32) / np.sqrt(obs_dim)
    base_action = observations[:, -1] @ weights
    trend = np.linspace(0.0, 1.0, pred_horizon, dtype=np.float32)[None, :, None]
    noise = 0.05 * rng.normal(size=(num_samples, pred_horizon, action_dim)).astype(np.float32)
    actions = base_action[:, None, :] * (1.0 + trend) + noise
    return SequenceDataset(observations, actions)
