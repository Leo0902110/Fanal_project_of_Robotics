from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DiffusionPolicyConfig:
    condition_dim: int
    action_dim: int
    horizon: int = 8
    hidden_dim: int = 256
    num_diffusion_steps: int = 50
    beta_start: float = 1e-4
    beta_end: float = 2e-2


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        frequencies = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device))
            * torch.arange(half, device=device, dtype=torch.float32)
            / max(half - 1, 1)
        )
        args = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ConditionalNoisePredictor(nn.Module):
    """MLP epsilon model for action-sequence diffusion."""

    def __init__(self, config: DiffusionPolicyConfig):
        super().__init__()
        self.config = config
        flat_action_dim = config.horizon * config.action_dim
        time_dim = max(32, config.hidden_dim // 2)
        if time_dim % 2:
            time_dim += 1
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.net = nn.Sequential(
            nn.Linear(flat_action_dim + config.condition_dim + time_dim, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.Mish(),
            nn.Linear(config.hidden_dim, flat_action_dim),
        )

    def forward(self, noisy_actions: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        batch = noisy_actions.shape[0]
        x = noisy_actions.reshape(batch, -1)
        t = self.time_embedding(timesteps)
        pred = self.net(torch.cat([x, condition, t], dim=-1))
        return pred.reshape(batch, self.config.horizon, self.config.action_dim)


class GaussianDiffusionPolicy(nn.Module):
    """Minimal conditional DDPM policy for short robot action sequences."""

    def __init__(self, config: DiffusionPolicyConfig):
        super().__init__()
        self.config = config
        self.model = ConditionalNoisePredictor(config)
        betas = torch.linspace(config.beta_start, config.beta_end, config.num_diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    def training_loss(
        self,
        actions: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = actions.shape[0]
        timesteps = torch.randint(0, self.config.num_diffusion_steps, (batch,), device=actions.device)
        noise = torch.randn_like(actions)
        noisy_actions = self.q_sample(actions, timesteps, noise)
        pred_noise = self.model(noisy_actions, timesteps, condition)
        loss = (pred_noise - noise) ** 2
        if mask is not None:
            loss = loss * mask[:, :, None]
            denom = mask.sum().clamp_min(1.0) * actions.shape[-1]
            return loss.sum() / denom
        return loss.mean()

    def q_sample(self, actions: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1)
        return sqrt_alpha * actions + sqrt_one_minus * noise

    @torch.no_grad()
    def sample(self, condition: torch.Tensor, *, action_low: torch.Tensor, action_high: torch.Tensor) -> torch.Tensor:
        batch = condition.shape[0]
        device = condition.device
        actions = torch.randn(batch, self.config.horizon, self.config.action_dim, device=device)
        for step in reversed(range(self.config.num_diffusion_steps)):
            timesteps = torch.full((batch,), step, device=device, dtype=torch.long)
            pred_noise = self.model(actions, timesteps, condition)
            alpha = self.alphas[step]
            alpha_bar = self.alphas_cumprod[step]
            beta = self.betas[step]
            actions = (actions - beta / torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
            if step > 0:
                actions = actions + torch.sqrt(beta) * torch.randn_like(actions)
        return torch.max(torch.min(actions, action_high.view(1, 1, -1)), action_low.view(1, 1, -1))
