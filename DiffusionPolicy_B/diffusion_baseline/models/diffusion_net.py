# 用途: 实现条件扩散动作去噪网络，输入带噪动作、时间步和观测条件后预测噪声。
# Purpose: Denoise noisy action trajectories conditioned on observation/state features and timesteps.

from __future__ import annotations

import math

import torch
from torch import nn

from diffusion_baseline.models.encoder import ObservationEncoder


class SinusoidalTimestepEmbedding(nn.Module):
    """Classic sinusoidal timestep embedding followed by a small MLP."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(
            half_dim, device=timesteps.device, dtype=torch.float32
        )
        exponent = exponent / max(half_dim - 1, 1)
        emb = timesteps.float()[:, None] * torch.exp(exponent)[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1))
        return self.proj(emb)


class ResidualMLPBlock(nn.Module):
    """Residual block for flattened action trajectories."""

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.cond = nn.Linear(cond_dim, dim)
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return x + self.net(x + self.cond(cond))


class ConditionalDiffusionMLP(nn.Module):
    """Diffusion Policy baseline that denoises an action horizon."""

    def __init__(
        self,
        obs_horizon: int,
        obs_dim: int,
        pred_horizon: int,
        action_dim: int,
        hidden_dim: int = 512,
        cond_dim: int = 256,
        num_blocks: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.encoder = ObservationEncoder(
            obs_horizon=obs_horizon,
            obs_dim=obs_dim,
            hidden_dim=cond_dim,
            cond_dim=cond_dim,
            dropout=dropout,
        )
        self.time_embedding = SinusoidalTimestepEmbedding(cond_dim)
        self.input = nn.Linear(pred_horizon * action_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualMLPBlock(hidden_dim, cond_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, pred_horizon * action_dim),
        )

    def forward(self, noisy_action: torch.Tensor, timesteps: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        if noisy_action.ndim != 3:
            raise ValueError("noisy_action must have shape [B, pred_horizon, action_dim]")
        cond = self.encoder(obs) + self.time_embedding(timesteps)
        x = self.input(noisy_action.flatten(start_dim=1))
        for block in self.blocks:
            x = block(x, cond)
        noise_pred = self.output(x)
        return noise_pred.reshape(noisy_action.shape)


class DiffusionPolicyNet(nn.Module):
    """Compact test-friendly diffusion policy network.

    The forward signature matches common robotics usage:
    encoded observation, low-dimensional state, noisy action horizon, and timestep.
    """

    def __init__(
        self,
        repr_dim: int,
        state_dim: int,
        action_dim: int,
        action_horizon: int,
        hidden_dim: int = 256,
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.repr_dim = repr_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.time_embedding = SinusoidalTimestepEmbedding(time_dim)
        input_dim = repr_dim + state_dim + action_horizon * action_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_horizon * action_dim),
        )

    def forward(
        self,
        encoded_obs: torch.Tensor,
        state: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if encoded_obs.ndim != 2:
            raise ValueError("encoded_obs must have shape [B, repr_dim]")
        if state.ndim != 2:
            raise ValueError("state must have shape [B, state_dim]")
        if noisy_actions.ndim != 3:
            raise ValueError("noisy_actions must have shape [B, action_horizon, action_dim]")
        batch_size = noisy_actions.shape[0]
        if encoded_obs.shape[0] != batch_size or state.shape[0] != batch_size:
            raise ValueError("encoded_obs, state, and noisy_actions must share the same batch size")
        if timesteps.shape[0] != batch_size:
            raise ValueError("timesteps must have shape [B]")

        features = torch.cat(
            [
                encoded_obs.float(),
                state.float(),
                noisy_actions.float().flatten(start_dim=1),
                self.time_embedding(timesteps),
            ],
            dim=-1,
        )
        return self.net(features).reshape_as(noisy_actions)
