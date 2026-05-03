# 用途: 实现 DDPM 前向加噪和反向采样所需的 beta/alpha 调度。

from __future__ import annotations

import torch


class DDPMScheduler:
    """Minimal DDPM scheduler with epsilon prediction."""

    def __init__(
        self,
        num_train_timesteps: int = 100,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ) -> None:
        if num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive")
        self.num_train_timesteps = num_train_timesteps
        self.device = torch.device(device)

        self.betas = torch.linspace(beta_start, beta_end, num_train_timesteps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=self.device), self.alphas_cumprod[:-1]], dim=0
        )
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        ).clamp_min(1e-20)

    def to(self, device: torch.device | str) -> "DDPMScheduler":
        return DDPMScheduler(
            num_train_timesteps=self.num_train_timesteps,
            beta_start=float(self.betas[0].detach().cpu()),
            beta_end=float(self.betas[-1].detach().cpu()),
            device=device,
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, timesteps, original_samples.shape)
        sqrt_one_minus_alpha = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, original_samples.shape
        )
        return sqrt_alpha * original_samples + sqrt_one_minus_alpha * noise

    @torch.no_grad()
    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        t = torch.full((sample.shape[0],), timestep, dtype=torch.long, device=sample.device)
        beta_t = self._extract(self.betas, t, sample.shape)
        alpha_t = self._extract(self.alphas, t, sample.shape)
        alpha_cumprod_t = self._extract(self.alphas_cumprod, t, sample.shape)

        mean = (sample - beta_t * model_output / torch.sqrt(1.0 - alpha_cumprod_t)) / torch.sqrt(alpha_t)
        if timestep == 0:
            return mean

        variance = self._extract(self.posterior_variance, t, sample.shape)
        noise = torch.randn(sample.shape, device=sample.device, generator=generator)
        return mean + torch.sqrt(variance) * noise

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        out = values.to(timesteps.device).gather(0, timesteps)
        return out.reshape(timesteps.shape[0], *([1] * (len(shape) - 1)))
