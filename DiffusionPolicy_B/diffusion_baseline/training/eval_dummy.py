# 用途: 加载 dummy checkpoint，执行轻量反向采样，并打印动作 shape/min/max。
# Purpose: Load the dummy checkpoint, run lightweight reverse sampling, and print action shape/min/max.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch import nn


# Step 0: Make direct script execution work from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_baseline.models.diffusion_net import DiffusionPolicyNet
from diffusion_baseline.models.encoder import CNNEncoder
from diffusion_baseline.models.schedule import DiffusionSchedule


IMAGE_SHAPE = (64, 64, 3)
REPR_DIM = 128
STATE_DIM = 7
ACTION_HORIZON = 8
ACTION_DIM = 4
NUM_DIFFUSION_STEPS = 100
DEFAULT_CKPT_PATH = Path("diffusion_baseline/checkpoints/dummy_ckpt.pt")


class DummyPolicy(nn.Module):
    """Same model wrapper used by run_dummy_train.py."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = CNNEncoder(in_channels=3, repr_dim=REPR_DIM)
        self.diffusion_net = DiffusionPolicyNet(
            repr_dim=REPR_DIM,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            action_horizon=ACTION_HORIZON,
        )

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        noisy_actions: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        encoded = self.encoder(images)
        return self.diffusion_net(encoded, states, noisy_actions, timesteps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the dummy Diffusion Policy checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "[警告] 检测到 CUDA 不可用，已切换到 CPU，PyTorch 仍可正常运行"
            "（当前安装为 CUDA 版，兼容 CPU 计算）"
        )
        return torch.device("cpu")
    return torch.device(requested_device)


@torch.no_grad()
def simple_reverse_sample(
    model: DummyPolicy,
    schedule: DiffusionSchedule,
    images: torch.Tensor,
    states: torch.Tensor,
    sample_steps: int,
) -> torch.Tensor:
    # Step 1: Start from Gaussian action noise.
    actions = torch.randn(
        images.shape[0],
        ACTION_HORIZON,
        ACTION_DIM,
        device=images.device,
        dtype=torch.float32,
    )

    # Step 2: Use a short deterministic DDIM-like denoising loop for speed.
    timesteps = torch.linspace(
        schedule.num_timesteps - 1,
        0,
        steps=max(sample_steps, 1),
        device=images.device,
    ).long()
    for timestep in timesteps:
        t = torch.full((images.shape[0],), int(timestep.item()), device=images.device, dtype=torch.long)
        pred_noise = model(images, states, actions, t)
        alpha_bar = schedule.alphas_cumprod[t].reshape(-1, 1, 1).to(images.device)
        pred_x0 = (actions - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
        actions = pred_x0.clamp(-5.0, 5.0)
    return actions


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}")

    # Step 3: Load checkpoint and restore model weights.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = DummyPolicy().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = DiffusionSchedule(num_timesteps=NUM_DIFFUSION_STEPS, device=device)

    # Step 4: Create tiny random conditioning inputs and sample actions.
    images = torch.randint(0, 256, (args.batch_size, *IMAGE_SHAPE), dtype=torch.uint8, device=device)
    states = torch.randn(args.batch_size, STATE_DIM, dtype=torch.float32, device=device)
    actions = simple_reverse_sample(model, schedule, images, states, args.sample_steps)

    print(f"checkpoint_step={checkpoint['step']}")
    print(f"sampled_action_shape={tuple(actions.shape)}")
    print(f"sampled_action_min={actions.min().item():.6f}")
    print(f"sampled_action_max={actions.max().item():.6f}")
    print("DUMMY_EVAL_SUCCESS")


if __name__ == "__main__":
    main()
