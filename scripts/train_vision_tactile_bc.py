from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import TACTILE_FEATURE_NAMES, load_vision_tactile_npz


class VisionTactileBC(nn.Module):
    def __init__(self, tactile_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 192, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(12, 192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.policy_head = nn.Sequential(
            nn.Linear(192 + tactile_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, images: torch.Tensor, tactile: torch.Tensor) -> torch.Tensor:
        encoded = self.image_encoder(images)
        return self.policy_head(torch.cat([encoded, tactile], dim=-1))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(n: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    val_size = max(1, int(n * val_fraction)) if n > 1 else 0
    return indices[val_size:], indices[:val_size]


def normalize_tactile(tactile: np.ndarray, train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = tactile[train_indices].mean(axis=0, dtype=np.float64).astype(np.float32)
    std = tactile[train_indices].std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return ((tactile - mean) / std).astype(np.float32), mean, std


def make_loader(images, tactile, actions, indices, batch_size, shuffle):
    ds = TensorDataset(
        torch.from_numpy(images[indices].astype(np.float32)),
        torch.from_numpy(tactile[indices].astype(np.float32)),
        torch.from_numpy(actions[indices].astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=torch.cuda.is_available())


def evaluate(model, loader, device) -> float:
    if len(loader.dataset) == 0:
        return 0.0
    model.eval()
    losses = []
    loss_fn = nn.SmoothL1Loss()
    with torch.no_grad():
        for images, tactile, actions in loader:
            images = images.to(device, non_blocking=True)
            tactile = tactile.to(device, non_blocking=True)
            actions = actions.to(device, non_blocking=True)
            losses.append(float(loss_fn(model(images, tactile), actions).item()))
    return float(np.mean(losses)) if losses else 0.0


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    images, tactile, actions, metadata = load_vision_tactile_npz(args.demo_dir)
    train_idx, val_idx = split_indices(len(actions), args.val_fraction, args.seed)
    tactile, tactile_mean, tactile_std = normalize_tactile(tactile, train_idx)
    train_loader = make_loader(images, tactile, actions, train_idx, args.batch_size, True)
    val_loader = make_loader(images, tactile, actions, val_idx, args.batch_size, False)

    model = VisionTactileBC(tactile.shape[1], actions.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch_images, batch_tactile, batch_actions in train_loader:
            batch_images = batch_images.to(device, non_blocking=True)
            batch_tactile = batch_tactile.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp)):
                pred = model(batch_images, batch_tactile)
                loss = loss_fn(pred, batch_actions)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_loss": evaluate(model, val_loader, device),
        }
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps(row, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "vision_tactile_bc.pt"
    metrics_path = output_dir / "vision_tactile_bc_metrics.json"
    demo_meta = metadata[0] if metadata else {}
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "tactile_dim": tactile.shape[1],
            "action_dim": actions.shape[1],
            "hidden_dim": args.hidden_dim,
            "tactile_feature_names": list(TACTILE_FEATURE_NAMES),
            "tactile_mean": torch.from_numpy(tactile_mean),
            "tactile_std": torch.from_numpy(tactile_std),
            "image_shape": list(images.shape[1:]),
            "input_type": "hand_camera_rgbd_plus_tactile",
            "uses_oracle_geometry": False,
            "uses_grasp_assist": False,
            "args": vars(args),
            "demo_metadata": demo_meta,
        },
        checkpoint_path,
    )
    metrics = {
        "num_transitions": int(len(actions)),
        "image_shape": list(images.shape[1:]),
        "tactile_dim": int(tactile.shape[1]),
        "action_dim": int(actions.shape[1]),
        "input_type": "hand_camera_rgbd_plus_tactile",
        "uses_oracle_geometry": False,
        "uses_grasp_assist": False,
        "checkpoint_path": str(checkpoint_path),
        "demo_metadata": demo_meta,
        "history": history,
        "final_train_loss": history[-1]["train_loss"] if history else 0.0,
        "final_val_loss": history[-1]["val_loss"] if history else 0.0,
    }
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Level 2 CNN BC from hand-camera RGB-D + tactile features.")
    parser.add_argument("--demo-dir", default="data/demos/pickcube_level2_vision_tactile")
    parser.add_argument("--output-dir", default="runs/vision_tactile_level2_bc")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
