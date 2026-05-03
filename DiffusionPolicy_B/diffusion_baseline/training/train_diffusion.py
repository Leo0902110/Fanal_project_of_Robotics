# 用途: 提供 Diffusion Policy 基线训练入口，支持真实 npz 数据和合成数据 smoke test。

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from diffusion_baseline.data.sequence_dataset import SequenceDataset, make_synthetic_dataset
from diffusion_baseline.diffusion.scheduler import DDPMScheduler
from diffusion_baseline.models.diffusion_net import ConditionalDiffusionMLP
from diffusion_baseline.utils.checkpoint import save_checkpoint
from diffusion_baseline.utils.config import TrainConfig
from diffusion_baseline.utils.seed import seed_everything


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a lightweight Diffusion Policy baseline.")
    parser.add_argument("--data-path", type=Path, default=None, help="Path to npz with observations/actions.")
    parser.add_argument("--run-dir", type=Path, default=Path("diffusion_baseline/runs/latest"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--num-diffusion-steps", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


def build_dataset(config: TrainConfig) -> SequenceDataset:
    if config.data_path is None:
        return make_synthetic_dataset(seed=config.seed)
    return SequenceDataset.from_npz(config.data_path)


def train_one_epoch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="train", leave=False):
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)
        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (action.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noisy_action = scheduler.add_noise(action, noise, timesteps)
        noise_pred = model(noisy_action, timesteps, obs)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * action.shape[0]
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, scheduler: DDPMScheduler, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    for batch in tqdm(loader, desc="val", leave=False):
        obs = batch["obs"].to(device)
        action = batch["action"].to(device)
        noise = torch.randn_like(action)
        timesteps = torch.randint(
            0,
            scheduler.num_train_timesteps,
            (action.shape[0],),
            device=device,
            dtype=torch.long,
        )
        noisy_action = scheduler.add_noise(action, noise, timesteps)
        noise_pred = model(noisy_action, timesteps, obs)
        loss = torch.nn.functional.mse_loss(noise_pred, noise)
        total_loss += loss.item() * action.shape[0]
    return total_loss / len(loader.dataset)


def main() -> None:
    config = parse_args()
    seed_everything(config.seed)
    device = torch.device(config.device)

    dataset = build_dataset(config)
    val_size = max(1, int(len(dataset) * config.val_ratio))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed),
    )
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=0)

    model = ConditionalDiffusionMLP(
        obs_horizon=dataset.obs_horizon,
        obs_dim=dataset.obs_dim,
        pred_horizon=dataset.pred_horizon,
        action_dim=dataset.action_dim,
        hidden_dim=config.hidden_dim,
        cond_dim=config.cond_dim,
        num_blocks=config.num_blocks,
        dropout=config.dropout,
    ).to(device)
    scheduler = DDPMScheduler(num_train_timesteps=config.num_diffusion_steps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    config.run_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(model, scheduler, train_loader, optimizer, device)
        val_loss = evaluate(model, scheduler, val_loader, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(
                config.run_dir / "checkpoint.pt",
                model=model,
                optimizer=optimizer,
                config=config,
                dataset=dataset,
                epoch=epoch,
                val_loss=val_loss,
            )


if __name__ == "__main__":
    main()
