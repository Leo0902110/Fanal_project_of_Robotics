from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import TrajectoryWindow, TrajectoryWindowDataset
from src.models import DiffusionPolicyConfig, GaussianDiffusionPolicy


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def collate_windows(batch: list[TrajectoryWindow]) -> dict[str, torch.Tensor]:
    return {
        "condition": torch.from_numpy(np.stack([item.condition for item in batch]).astype(np.float32)),
        "actions": torch.from_numpy(np.stack([item.actions for item in batch]).astype(np.float32)),
        "mask": torch.from_numpy(np.stack([item.mask for item in batch]).astype(np.float32)),
    }


def evaluate(model: GaussianDiffusionPolicy, loader: DataLoader, device: torch.device) -> float:
    if len(loader.dataset) == 0:
        return 0.0
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            condition = batch["condition"].to(device)
            actions = batch["actions"].to(device)
            mask = batch["mask"].to(device)
            losses.append(float(model.training_loss(actions, condition, mask).item()))
    return float(np.mean(losses)) if losses else 0.0


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dataset = TrajectoryWindowDataset(
        args.demo_dir,
        horizon=args.horizon,
        only_success=not args.include_failed_demos,
        stride=args.stride,
    )
    val_size = max(1, int(len(dataset) * args.val_fraction)) if len(dataset) > 1 else 0
    train_size = len(dataset) - val_size
    if train_size <= 0:
        train_size, val_size = len(dataset), 0
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_windows)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_windows)

    config = DiffusionPolicyConfig(
        condition_dim=dataset.condition_dim,
        action_dim=dataset.action_dim,
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        num_diffusion_steps=args.diffusion_steps,
    )
    model = GaussianDiffusionPolicy(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            condition = batch["condition"].to(device)
            actions = batch["actions"].to(device)
            mask = batch["mask"].to(device)
            loss = model.training_loss(actions, condition, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(float(loss.item()))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": evaluate(model, val_loader, device),
        }
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps(row, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "diffusion_policy.pt"
    metrics_path = output_dir / "diffusion_metrics.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": vars(config),
            "args": vars(args),
            "feature_names": [
                "flattened_observation",
                "uncertainty",
                "boundary_confidence",
                "probe_state",
                "dominant_reason_one_hot",
                "probe_point_xy",
                "refined_grasp_target_xyz",
            ],
        },
        checkpoint_path,
    )
    metrics = {
        "num_windows": int(len(dataset)),
        "condition_dim": int(dataset.condition_dim),
        "action_dim": int(dataset.action_dim),
        "horizon": int(args.horizon),
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "final_train_loss": history[-1]["train_loss"] if history else 0.0,
        "final_val_loss": history[-1]["val_loss"] if history else 0.0,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved diffusion checkpoint: {checkpoint_path}")
    print(f"Saved diffusion metrics: {metrics_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal conditional Diffusion Policy.")
    parser.add_argument("--demo-dir", default="data/demos/pickcube_mvp")
    parser.add_argument("--output-dir", default="runs/dp_mvp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--diffusion-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--include-failed-demos", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
