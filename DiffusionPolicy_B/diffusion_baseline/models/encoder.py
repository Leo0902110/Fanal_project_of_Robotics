# 用途: 将观测输入编码为固定长度表示，供扩散动作网络使用。
# Purpose: Encode observations into fixed-size representations for diffusion policy models.

from __future__ import annotations

import torch
from torch import nn


class ObservationEncoder(nn.Module):
    """MLP encoder for flattened observation histories."""

    def __init__(
        self,
        obs_horizon: int,
        obs_dim: int,
        hidden_dim: int = 256,
        cond_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.obs_horizon = obs_horizon
        self.obs_dim = obs_dim
        self.net = nn.Sequential(
            nn.Linear(obs_horizon * obs_dim, hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3:
            raise ValueError("obs must have shape [B, obs_horizon, obs_dim]")
        return self.net(obs.flatten(start_dim=1))


class CNNEncoder(nn.Module):
    """Small image encoder for NHWC image observations."""

    def __init__(self, in_channels: int = 3, repr_dim: int = 128) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.repr_dim = repr_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, repr_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError("images must have shape [B, H, W, C] or [B, C, H, W]")
        if images.dtype == torch.uint8:
            images = images.float() / 255.0
        else:
            images = images.float()
        if images.shape[-1] == self.in_channels:
            images = images.permute(0, 3, 1, 2).contiguous()
        if images.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} channels, got shape {tuple(images.shape)}")
        return self.net(images)
