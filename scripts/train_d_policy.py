from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.d_policy import ActiveTactilePolicy


def load_demos(demo_dir: Path) -> dict[str, np.ndarray]:
    paths = sorted(demo_dir.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No episode_*.npz files found in {demo_dir}")
    arrays: dict[str, list[np.ndarray]] = {
        "vision_features": [],
        "tactile_features": [],
        "probe_flags": [],
        "target_actions": [],
        "target_probe_flags": [],
        "target_uncertainty": [],
        "target_phase_labels": [],
    }
    for path in paths:
        data = np.load(path, allow_pickle=True)
        transition_count = int(data["target_actions"].shape[0])
        arrays["vision_features"].append(data["vision_features"].astype(np.float32))
        arrays["tactile_features"].append(data["tactile_features"].astype(np.float32))
        arrays["probe_flags"].append(np.zeros(transition_count, dtype=np.float32))
        arrays["target_actions"].append(data["target_actions"].astype(np.float32))
        arrays["target_probe_flags"].append(data["target_probe_flags"].astype(np.float32))
        arrays["target_uncertainty"].append(data["target_uncertainty"].astype(np.float32))
        arrays["target_phase_labels"].append(data["target_phase_labels"].astype(np.int64))
    return {key: np.concatenate(value, axis=0) for key, value in arrays.items()}


def split_indices(count: int, val_fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(count)
    rng.shuffle(indices)
    val_count = max(1, int(round(count * val_fraction))) if count > 1 else 0
    return indices[val_count:], indices[:val_count]


def fit_normalizer(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = values.std(axis=0, dtype=np.float64).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def make_loader(data: dict[str, np.ndarray], indices: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(
        torch.from_numpy(data["vision_features"][indices]).float(),
        torch.from_numpy(data["tactile_features"][indices]).float(),
        torch.from_numpy(data["probe_flags"][indices]).float(),
        torch.from_numpy(data["target_actions"][indices]).float(),
        torch.from_numpy(data["target_probe_flags"][indices]).float(),
        torch.from_numpy(data["target_uncertainty"][indices]).float(),
        torch.from_numpy(data["target_phase_labels"][indices]).long(),
    )
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle)


def batch_from_tensors(policy: ActiveTactilePolicy, tensors, device: torch.device):
    vision, tactile, probe, actions, target_probe, uncertainty, phase = [tensor.to(device) for tensor in tensors]
    return policy.make_batch(
        vision_features=vision,
        tactile_features=tactile,
        probe_flags=probe,
        target_actions=actions,
        target_probe_flags=target_probe,
        target_uncertainty=uncertainty,
        target_phase_labels=phase,
    )


def evaluate(policy: ActiveTactilePolicy, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    policy.eval()
    totals = {"total_loss": 0.0, "action_loss": 0.0, "probe_loss": 0.0, "uncertainty_loss": 0.0, "phase_loss": 0.0}
    correct = 0
    count = 0
    with torch.no_grad():
        for tensors in loader:
            batch = batch_from_tensors(policy, tensors, device)
            losses = policy.compute_loss(
                batch,
                action_weight=args.action_weight,
                probe_weight=args.probe_weight,
                uncertainty_weight=args.uncertainty_weight,
                phase_weight=args.phase_weight,
            )
            batch_count = int(batch.vision_features.shape[0])
            totals["total_loss"] += float(losses.total_loss.detach().cpu()) * batch_count
            totals["action_loss"] += float(losses.action_loss.detach().cpu()) * batch_count
            totals["probe_loss"] += float(losses.probe_loss.detach().cpu()) * batch_count
            totals["uncertainty_loss"] += float(losses.uncertainty_loss.detach().cpu()) * batch_count
            totals["phase_loss"] += float(losses.phase_loss.detach().cpu()) * batch_count
            outputs = policy.forward_batch(batch)
            if outputs.phase_logits is not None:
                pred = outputs.phase_logits.argmax(dim=-1)
                correct += int((pred == batch.target_phase_labels.long()).sum().detach().cpu())
            count += batch_count
    metrics = {key: value / max(count, 1) for key, value in totals.items()}
    metrics["phase_accuracy"] = correct / max(count, 1)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train D ActiveTactilePolicy from assist-teacher demonstrations.")
    parser.add_argument("--demo-dir", default="data/demos/pickcube_d_assist_teacher")
    parser.add_argument("--output-dir", default="runs/d_policy_assist_teacher")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--action-weight", type=float, default=1.0)
    parser.add_argument("--probe-weight", type=float, default=0.1)
    parser.add_argument("--uncertainty-weight", type=float, default=0.1)
    parser.add_argument("--phase-weight", type=float, default=0.5)
    parser.add_argument("--log-every", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data = load_demos(Path(args.demo_dir))
    train_idx, val_idx = split_indices(len(data["target_actions"]), args.val_fraction, args.seed)

    vision_mean, vision_std = fit_normalizer(data["vision_features"][train_idx])
    tactile_mean, tactile_std = fit_normalizer(data["tactile_features"][train_idx])
    data["vision_features"] = ((data["vision_features"] - vision_mean) / vision_std).astype(np.float32)
    data["tactile_features"] = ((data["tactile_features"] - tactile_mean) / tactile_std).astype(np.float32)

    train_loader = make_loader(data, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(data, val_idx, args.batch_size, shuffle=False)
    phase_names = ("bc", "align_xy", "descend", "close", "transfer", "settle")
    policy = ActiveTactilePolicy(
        action_dim=int(data["target_actions"].shape[1]),
        vision_dim=int(data["vision_features"].shape[1]),
        tactile_dim=int(data["tactile_features"].shape[1]),
        hidden_dim=args.hidden_dim,
        num_phase_classes=len(phase_names),
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)

    history = []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        for tensors in train_loader:
            batch = batch_from_tensors(policy, tensors, device)
            policy.train_step(
                batch,
                optimizer,
                action_weight=args.action_weight,
                probe_weight=args.probe_weight,
                uncertainty_weight=args.uncertainty_weight,
                phase_weight=args.phase_weight,
            )
        train_metrics = evaluate(policy, train_loader, device, args)
        val_metrics = evaluate(policy, val_loader, device, args)
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps({key: round(float(value), 6) if isinstance(value, float) else value for key, value in row.items()}, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "d_policy.pt"
    torch.save(
        {
            "model_state_dict": policy.state_dict(),
            "action_dim": int(data["target_actions"].shape[1]),
            "vision_dim": int(data["vision_features"].shape[1]),
            "tactile_dim": int(data["tactile_features"].shape[1]),
            "hidden_dim": args.hidden_dim,
            "phase_names": phase_names,
            "vision_mean": torch.from_numpy(vision_mean),
            "vision_std": torch.from_numpy(vision_std),
            "tactile_mean": torch.from_numpy(tactile_mean),
            "tactile_std": torch.from_numpy(tactile_std),
            "args": vars(args),
        },
        checkpoint_path,
    )
    metrics = {
        "checkpoint_path": str(checkpoint_path),
        "num_transitions": int(len(data["target_actions"])),
        "train_transitions": int(len(train_idx)),
        "val_transitions": int(len(val_idx)),
        "device": str(device),
        "history": history,
        "final": history[-1] if history else {},
    }
    (output_dir / "d_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(checkpoint_path), "device": str(device), "final": metrics["final"]}, indent=2))


if __name__ == "__main__":
    main()