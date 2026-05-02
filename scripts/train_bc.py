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

from src.data import DemoDataset


class BCPolicy(nn.Module):
    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_transitions(demo_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    dataset = DemoDataset(demo_dir)
    obs_features = []
    action_targets = []
    expected_obs_dim = None
    expected_action_dim = None

    for episode in dataset:
        obs = episode.observations
        actions = episode.actions
        if expected_obs_dim is None:
            expected_obs_dim = obs.shape[1]
            expected_action_dim = actions.shape[1]
        if obs.shape[1] != expected_obs_dim or actions.shape[1] != expected_action_dim:
            raise ValueError(
                "All demo episodes must have matching observation and action dimensions. "
                f"Expected ({expected_obs_dim}, {expected_action_dim}), got "
                f"({obs.shape[1]}, {actions.shape[1]}) in {episode.path}."
            )

        extra = np.stack(
            [
                episode.uncertainty,
                episode.boundary_confidence,
            ],
            axis=1,
        )
        obs_features.append(np.concatenate([obs, extra], axis=1))
        action_targets.append(actions)

    return (
        np.concatenate(obs_features, axis=0).astype(np.float32),
        np.concatenate(action_targets, axis=0).astype(np.float32),
    )


def split_train_val(
    inputs: np.ndarray,
    targets: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(inputs))
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_fraction)) if len(indices) > 1 else 0
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    if len(train_indices) == 0:
        train_indices = val_indices
    return inputs[train_indices], targets[train_indices], inputs[val_indices], targets[val_indices]


def make_loader(inputs: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(targets))
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    if len(loader.dataset) == 0:
        return 0.0
    model.eval()
    losses = []
    loss_fn = nn.MSELoss()
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            losses.append(float(loss_fn(model(inputs), targets).item()))
    return float(np.mean(losses)) if losses else 0.0


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    inputs, targets = load_transitions(Path(args.demo_dir))
    train_x, train_y, val_x, val_y = split_train_val(inputs, targets, args.val_fraction, args.seed)

    train_loader = make_loader(train_x, train_y, args.batch_size, shuffle=True)
    val_loader = make_loader(val_x, val_y, args.batch_size, shuffle=False)
    model = BCPolicy(inputs.shape[1], targets.shape[1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch_inputs, batch_targets in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            pred = model(batch_inputs)
            loss = loss_fn(pred, batch_targets)
            optimizer.zero_grad()
            loss.backward()
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
    checkpoint_path = output_dir / "bc_policy.pt"
    metrics_path = output_dir / "bc_metrics.json"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": inputs.shape[1],
            "action_dim": targets.shape[1],
            "hidden_dim": args.hidden_dim,
            "feature_names": ["flattened_observation", "uncertainty", "boundary_confidence"],
            "args": vars(args),
        },
        checkpoint_path,
    )
    metrics = {
        "num_transitions": int(len(inputs)),
        "input_dim": int(inputs.shape[1]),
        "action_dim": int(targets.shape[1]),
        "checkpoint_path": str(checkpoint_path),
        "history": history,
        "final_train_loss": history[-1]["train_loss"] if history else 0.0,
        "final_val_loss": history[-1]["val_loss"] if history else 0.0,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved metrics: {metrics_path}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a minimal behavior cloning baseline.")
    parser.add_argument("--demo-dir", default="data/demos/pickcube_mvp")
    parser.add_argument("--output-dir", default="runs/bc_mvp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
