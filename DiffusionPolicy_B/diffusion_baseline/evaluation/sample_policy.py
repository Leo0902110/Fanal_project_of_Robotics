# 用途: 从已训练 checkpoint 加载策略，并对给定观测序列采样动作轨迹。

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from diffusion_baseline.diffusion.scheduler import DDPMScheduler
from diffusion_baseline.models.diffusion_net import ConditionalDiffusionMLP
from diffusion_baseline.utils.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample actions from a trained Diffusion Policy baseline.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--obs", type=Path, required=True, help="Path to .npy obs with shape [obs_horizon, obs_dim].")
    parser.add_argument("--output", type=Path, default=Path("diffusion_baseline/runs/latest/sampled_action.npy"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def sample_action(
    model: ConditionalDiffusionMLP,
    scheduler: DDPMScheduler,
    obs: torch.Tensor,
    pred_horizon: int,
    action_dim: int,
    generator: torch.Generator,
) -> torch.Tensor:
    model.eval()
    action = torch.randn(
        (obs.shape[0], pred_horizon, action_dim),
        device=obs.device,
        generator=generator,
    )
    for timestep in reversed(range(scheduler.num_train_timesteps)):
        timesteps = torch.full((obs.shape[0],), timestep, device=obs.device, dtype=torch.long)
        noise_pred = model(action, timesteps, obs)
        action = scheduler.step(noise_pred, timestep, action, generator=generator)
    return action


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    model = ConditionalDiffusionMLP(
        obs_horizon=checkpoint["obs_horizon"],
        obs_dim=checkpoint["obs_dim"],
        pred_horizon=checkpoint["pred_horizon"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=config["hidden_dim"],
        cond_dim=config["cond_dim"],
        num_blocks=config["num_blocks"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_steps"],
        device=device,
    )

    obs = np.load(args.obs).astype(np.float32)
    obs_tensor = torch.as_tensor(obs, device=device).unsqueeze(0)
    obs_tensor = (obs_tensor - checkpoint["obs_mean"].to(device).squeeze(0)) / checkpoint[
        "obs_std"
    ].to(device).squeeze(0)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    action = sample_action(
        model,
        scheduler,
        obs_tensor,
        pred_horizon=checkpoint["pred_horizon"],
        action_dim=checkpoint["action_dim"],
        generator=generator,
    )
    action = action * checkpoint["action_std"].to(device).squeeze(0) + checkpoint["action_mean"].to(
        device
    ).squeeze(0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, action.squeeze(0).cpu().numpy())
    print(f"saved sampled action to {args.output}")


if __name__ == "__main__":
    main()
