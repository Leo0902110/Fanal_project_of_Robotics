# 用途: 提供测试和训练可复用的扩散前向加噪日程。
# Purpose: Provide a reusable diffusion q_sample schedule for tests and training code.

from __future__ import annotations

import torch


class DiffusionSchedule:
    """Minimal DDPM-style forward noising schedule."""

    def __init__(
        self,
        num_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_timesteps <= 0:
            raise ValueError("num_timesteps must be positive")
        self.num_timesteps = num_timesteps
        self.device = torch.device(device)
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_start.ndim < 2:
            raise ValueError("x_start must include batch and feature dimensions")
        if timesteps.ndim != 1 or timesteps.shape[0] != x_start.shape[0]:
            raise ValueError("timesteps must have shape [B]")
        if noise is None:
            noise = torch.randn_like(x_start)
        if noise.shape != x_start.shape:
            raise ValueError("noise must have the same shape as x_start")

        timesteps = timesteps.to(device=x_start.device, dtype=torch.long)
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )
        return sqrt_alpha * x_start + sqrt_one_minus_alpha * noise

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        values = values.to(timesteps.device)
        gathered = values.gather(0, timesteps)
        return gathered.reshape(timesteps.shape[0], *([1] * (len(shape) - 1)))


def q_sample(
    x_start: torch.Tensor,
    timesteps: torch.Tensor,
    noise: torch.Tensor | None = None,
    num_timesteps: int = 100,
) -> torch.Tensor:
    """Convenience function for one-off q_sample calls."""

    schedule = DiffusionSchedule(num_timesteps=num_timesteps, device=x_start.device)
    return schedule.q_sample(x_start, timesteps, noise)
