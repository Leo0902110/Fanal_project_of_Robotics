from __future__ import annotations

import torch
from torch import nn


class TactileFeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, tactile_features: torch.Tensor) -> torch.Tensor:
        return self.network(tactile_features)


class VisionTactileFusionMLP(nn.Module):
    def __init__(
        self,
        vision_dim: int,
        tactile_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 128,
    ):
        super().__init__()
        self.tactile_encoder = TactileFeatureEncoder(tactile_dim, hidden_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + hidden_dim + 1, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        vision_features: torch.Tensor,
        tactile_features: torch.Tensor,
        probe_flag: torch.Tensor,
    ) -> torch.Tensor:
        tactile_latent = self.tactile_encoder(tactile_features)
        if probe_flag.ndim == 1:
            probe_flag = probe_flag.unsqueeze(-1)
        fused = torch.cat([vision_features, tactile_latent, probe_flag], dim=-1)
        return self.fusion(fused)