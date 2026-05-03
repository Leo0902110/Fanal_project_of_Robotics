from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        exponent = -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        exponent = exponent / max(half - 1, 1)
        args = timesteps.float().unsqueeze(-1) * torch.exp(exponent).unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb


class TactileConditionEncoder(nn.Module):
    def __init__(self, vision_dim: int, tactile_dim: int, hidden_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(vision_dim + tactile_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, vision_features: torch.Tensor, tactile_features: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([vision_features, tactile_features], dim=-1))


class TactileDiffusionPolicy(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        tactile_dim: int,
        action_dim: int,
        action_horizon: int = 8,
        hidden_dim: int = 256,
        time_dim: int = 128,
        num_phase_classes: int = 6,
    ):
        super().__init__()
        self.vision_dim = vision_dim
        self.tactile_dim = tactile_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.num_phase_classes = num_phase_classes
        self.condition_encoder = TactileConditionEncoder(vision_dim, tactile_dim, hidden_dim)
        self.phase_head = nn.Linear(hidden_dim, num_phase_classes)
        self.phase_embedding = nn.Embedding(num_phase_classes, hidden_dim)
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        action_flat_dim = action_dim * action_horizon
        self.denoiser = nn.Sequential(
            nn.Linear(action_flat_dim + hidden_dim + hidden_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_flat_dim),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
        vision_features: torch.Tensor,
        tactile_features: torch.Tensor,
        phase_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = noisy_actions.shape[0]
        noisy_flat = noisy_actions.reshape(batch_size, -1)
        cond = self.condition_encoder(vision_features, tactile_features)
        time_emb = self.time_encoder(timesteps)
        if phase_labels is None:
            phase_labels = self.phase_head(cond).argmax(dim=-1)
        phase_emb = self.phase_embedding(phase_labels.long())
        pred = self.denoiser(torch.cat([noisy_flat, cond, time_emb, phase_emb], dim=-1))
        return pred.reshape(batch_size, self.action_horizon, self.action_dim)

    def predict_phase_logits(self, vision_features: torch.Tensor, tactile_features: torch.Tensor) -> torch.Tensor:
        return self.phase_head(self.condition_encoder(vision_features, tactile_features))


class DiffusionScheduler:
    def __init__(self, num_train_steps: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.num_train_steps = num_train_steps
        self.betas = torch.linspace(beta_start, beta_end, num_train_steps, dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def to(self, device: torch.device) -> "DiffusionScheduler":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        return self

    def add_noise(self, clean_actions: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_cumprod[timesteps].view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * clean_actions + torch.sqrt(1.0 - alpha_bar) * noise

    @torch.no_grad()
    def sample(
        self,
        model: TactileDiffusionPolicy,
        vision_features: torch.Tensor,
        tactile_features: torch.Tensor,
        sample_steps: int | None = None,
        phase_labels: torch.Tensor | None = None,
        stochastic: bool = False,
        init_noise_scale: float = 1.0,
    ) -> torch.Tensor:
        sample_steps = sample_steps or self.num_train_steps
        device = vision_features.device
        batch_size = vision_features.shape[0]
        actions = torch.randn(batch_size, model.action_horizon, model.action_dim, device=device) * init_noise_scale
        step_indices = torch.linspace(self.num_train_steps - 1, 0, sample_steps, device=device).long().unique_consecutive()
        for timestep in step_indices:
            t = timestep.repeat(batch_size)
            pred_noise = model(actions, t, vision_features, tactile_features, phase_labels=phase_labels)
            beta = self.betas[timestep]
            alpha = self.alphas[timestep]
            alpha_bar = self.alpha_cumprod[timestep]
            actions = (actions - beta / torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha)
            if stochastic and int(timestep.item()) > 0:
                actions = actions + torch.sqrt(beta) * torch.randn_like(actions)
        return actions