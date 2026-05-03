# 用途: 运行一个极小 Diffusion Policy dummy 训练闭环，并保存 checkpoint。
# Purpose: Run a tiny Diffusion Policy dummy training loop and save a checkpoint.

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
from diffusion_baseline.utils.buffer import ReplayBuffer


IMAGE_SHAPE = (64, 64, 3)
REPR_DIM = 128
STATE_DIM = 7
ACTION_HORIZON = 8
ACTION_DIM = 4
BUFFER_SIZE = 1024
NUM_DIFFUSION_STEPS = 100
DEFAULT_CKPT_PATH = Path("diffusion_baseline/checkpoints/dummy_ckpt.pt")


class DummyPolicy(nn.Module):
    """Small wrapper that stores encoder and diffusion net in one checkpoint state_dict."""

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
    parser = argparse.ArgumentParser(description="Run a minimal dummy Diffusion Policy training loop.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT_PATH)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "[警告] 检测到 CUDA 不可用，已切换到 CPU，PyTorch 仍可正常运行"
            "（当前安装为 CUDA 版，兼容 CPU 计算）"
        )
        return torch.device("cpu")
    return torch.device(requested_device)


def build_random_buffer(size: int = BUFFER_SIZE) -> ReplayBuffer:
    # Step 1: Create a small random replay buffer that mimics image/state/action transitions.
    buffer = ReplayBuffer(capacity=size)
    for _ in range(size):
        buffer.push(
            image=torch.randint(0, 256, IMAGE_SHAPE, dtype=torch.uint8),
            state=torch.randn(STATE_DIM, dtype=torch.float32),
            action=torch.randn(ACTION_HORIZON, ACTION_DIM, dtype=torch.float32),
        )
    return buffer


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"device={device}")

    # Step 2: Build model, DDPM forward schedule, optimizer, and random buffer.
    model = DummyPolicy().to(device)
    schedule = DiffusionSchedule(num_timesteps=NUM_DIFFUSION_STEPS, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    buffer = build_random_buffer()
    print(f"buffer_size={len(buffer)} batch_size={args.batch_size} num_steps={args.num_steps}")

    # Step 3: Run a tiny train loop: sample batch, q_sample actions, predict noise, MSE, backward, step.
    last_loss = float("nan")
    for step in range(1, args.num_steps + 1):
        batch = move_batch_to_device(buffer.sample(args.batch_size), device)
        images = batch["image"]
        states = batch["state"].float()
        actions = batch["action"].float()
        noise = torch.randn_like(actions)
        timesteps = torch.randint(
            0,
            schedule.num_timesteps,
            (args.batch_size,),
            device=device,
            dtype=torch.long,
        )

        noisy_actions = schedule.q_sample(actions, timesteps, noise)
        pred_noise = model(images, states, noisy_actions, timesteps)
        loss = torch.nn.functional.mse_loss(pred_noise, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        last_loss = float(loss.detach().cpu())
        if step == 1 or step % 10 == 0 or step == args.num_steps:
            print(f"step={step:04d} loss={last_loss:.6f}")

    # Step 4: Save the exact minimal checkpoint format requested by the task.
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "step": args.num_steps}, args.checkpoint)
    ckpt_size = args.checkpoint.stat().st_size
    print(f"checkpoint_path={args.checkpoint}")
    print(f"checkpoint_size_bytes={ckpt_size}")
    print(f"DUMMY_TRAIN_SUCCESS final_loss={last_loss:.6f}")


if __name__ == "__main__":
    main()
