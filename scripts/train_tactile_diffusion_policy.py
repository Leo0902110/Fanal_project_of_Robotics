from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.tactile_diffusion_policy import DiffusionScheduler, TactileDiffusionPolicy
from scripts.evaluate_d_policy import load_policy as load_d_policy


def load_windows(demo_dir: Path, action_horizon: int) -> dict[str, np.ndarray]:
    paths = sorted(demo_dir.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No episode_*.npz files found in {demo_dir}")
    vision_features = []
    tactile_features = []
    action_chunks = []
    phase_labels = []
    for path in paths:
        data = np.load(path, allow_pickle=True)
        vision = data["vision_features"].astype(np.float32)
        tactile = data["tactile_features"].astype(np.float32)
        actions = data["target_actions"].astype(np.float32)
        phases = data["target_phase_labels"].astype(np.int64)
        for step in range(actions.shape[0]):
            end = min(step + action_horizon, actions.shape[0])
            chunk = actions[step:end]
            if chunk.shape[0] < action_horizon:
                pad = np.repeat(chunk[-1:], action_horizon - chunk.shape[0], axis=0)
                chunk = np.concatenate([chunk, pad], axis=0)
            vision_features.append(vision[step])
            tactile_features.append(tactile[step])
            action_chunks.append(chunk)
            phase_labels.append(phases[step])
    return {
        "vision_features": np.asarray(vision_features, dtype=np.float32),
        "tactile_features": np.asarray(tactile_features, dtype=np.float32),
        "action_chunks": np.asarray(action_chunks, dtype=np.float32),
        "phase_labels": np.asarray(phase_labels, dtype=np.int64),
    }


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
    dataset = TensorDataset(
        torch.from_numpy(data["vision_features"][indices]).float(),
        torch.from_numpy(data["tactile_features"][indices]).float(),
        torch.from_numpy(data["action_chunks"][indices]).float(),
        torch.from_numpy(data["phase_labels"][indices]).long(),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def predict_base_d_actions(
    vision_features: np.ndarray,
    tactile_features: np.ndarray,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    policy, _phase_names, normalizers = load_d_policy(checkpoint_path, device)
    policy.eval()
    vision = ((vision_features - normalizers["vision_mean"]) / normalizers["vision_std"]).astype(np.float32)
    tactile = ((tactile_features - normalizers["tactile_mean"]) / normalizers["tactile_std"]).astype(np.float32)
    outputs = []
    for start in range(0, vision.shape[0], batch_size):
        end = min(start + batch_size, vision.shape[0])
        batch_count = end - start
        batch = policy.make_batch(
            vision_features=torch.from_numpy(vision[start:end]).to(device),
            tactile_features=torch.from_numpy(tactile[start:end]).to(device),
            probe_flags=torch.zeros(batch_count, device=device),
        )
        pred = policy.forward_batch(batch).actions.detach().cpu().numpy().astype(np.float32)
        outputs.append(pred)
    return np.concatenate(outputs, axis=0)


def run_epoch(
    model: TactileDiffusionPolicy,
    scheduler: DiffusionScheduler,
    loader: DataLoader,
    device: torch.device,
    phase_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_noise_loss = 0.0
    total_phase_loss = 0.0
    total_phase_correct = 0
    total_count = 0
    for vision, tactile, actions, phase_labels in loader:
        vision = vision.to(device)
        tactile = tactile.to(device)
        actions = actions.to(device)
        phase_labels = phase_labels.to(device)
        timesteps = torch.randint(0, scheduler.num_train_steps, (actions.shape[0],), device=device)
        noise = torch.randn_like(actions)
        noisy_actions = scheduler.add_noise(actions, noise, timesteps)
        pred_noise = model(noisy_actions, timesteps, vision, tactile, phase_labels=phase_labels)
        phase_logits = model.predict_phase_logits(vision, tactile)
        noise_loss = F.mse_loss(pred_noise, noise)
        phase_loss = F.cross_entropy(phase_logits, phase_labels)
        loss = noise_loss + phase_loss_weight * phase_loss
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        batch_count = int(actions.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_count
        total_noise_loss += float(noise_loss.detach().cpu()) * batch_count
        total_phase_loss += float(phase_loss.detach().cpu()) * batch_count
        total_phase_correct += int((phase_logits.argmax(dim=-1) == phase_labels).sum().detach().cpu())
        total_count += batch_count
    return {
        "loss": total_loss / max(total_count, 1),
        "noise_mse": total_noise_loss / max(total_count, 1),
        "phase_loss": total_phase_loss / max(total_count, 1),
        "phase_accuracy": total_phase_correct / max(total_count, 1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tactile-conditioned diffusion policy from assist-teacher demos.")
    parser.add_argument("--demo-dir", default="data/demos/pickcube_d_assist_teacher_500")
    parser.add_argument("--output-dir", default="runs/tactile_diffusion_policy_500")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--phase-loss-weight", type=float, default=0.1)
    parser.add_argument("--base-d-checkpoint", default="")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    data = load_windows(Path(args.demo_dir), args.action_horizon)
    train_idx, val_idx = split_indices(len(data["action_chunks"]), args.val_fraction, args.seed)

    base_d_checkpoint = str(args.base_d_checkpoint).strip()
    if base_d_checkpoint:
        base_actions = predict_base_d_actions(
            data["vision_features"],
            data["tactile_features"],
            Path(base_d_checkpoint),
            device,
            args.batch_size,
        )
        base_chunks = np.repeat(base_actions[:, None, :], args.action_horizon, axis=1)
        data["action_chunks"] = (data["action_chunks"] - base_chunks).astype(np.float32)

    vision_mean, vision_std = fit_normalizer(data["vision_features"][train_idx])
    tactile_mean, tactile_std = fit_normalizer(data["tactile_features"][train_idx])
    flat_actions = data["action_chunks"][train_idx].reshape(-1, data["action_chunks"].shape[-1])
    action_mean, action_std = fit_normalizer(flat_actions)
    action_constant_mask = (flat_actions.std(axis=0, dtype=np.float64) < 1e-6).astype(np.float32)
    data["vision_features"] = ((data["vision_features"] - vision_mean) / vision_std).astype(np.float32)
    data["tactile_features"] = ((data["tactile_features"] - tactile_mean) / tactile_std).astype(np.float32)
    data["action_chunks"] = ((data["action_chunks"] - action_mean.reshape(1, 1, -1)) / action_std.reshape(1, 1, -1)).astype(np.float32)

    train_loader = make_loader(data, train_idx, args.batch_size, shuffle=True)
    val_loader = make_loader(data, val_idx, args.batch_size, shuffle=False)
    model = TactileDiffusionPolicy(
        vision_dim=int(data["vision_features"].shape[1]),
        tactile_dim=int(data["tactile_features"].shape[1]),
        action_dim=int(data["action_chunks"].shape[2]),
        action_horizon=args.action_horizon,
        hidden_dim=args.hidden_dim,
        num_phase_classes=6,
    ).to(device)
    scheduler = DiffusionScheduler(num_train_steps=args.diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, scheduler, train_loader, device, args.phase_loss_weight, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, scheduler, val_loader, device, args.phase_loss_weight, optimizer=None)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        history.append(row)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(json.dumps({key: round(float(value), 6) if isinstance(value, float) else value for key, value in row.items()}, indent=2))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "tactile_diffusion_policy.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vision_dim": model.vision_dim,
            "tactile_dim": model.tactile_dim,
            "action_dim": model.action_dim,
            "action_horizon": model.action_horizon,
            "hidden_dim": args.hidden_dim,
            "num_phase_classes": model.num_phase_classes,
            "phase_names": ("bc", "align_xy", "descend", "close", "transfer", "settle"),
            "diffusion_steps": args.diffusion_steps,
            "vision_mean": torch.from_numpy(vision_mean),
            "vision_std": torch.from_numpy(vision_std),
            "tactile_mean": torch.from_numpy(tactile_mean),
            "tactile_std": torch.from_numpy(tactile_std),
            "action_mean": torch.from_numpy(action_mean),
            "action_std": torch.from_numpy(action_std),
            "action_constant_mask": torch.from_numpy(action_constant_mask),
            "residual_mode": bool(base_d_checkpoint),
            "base_d_checkpoint": base_d_checkpoint,
            "args": vars(args),
        },
        checkpoint_path,
    )
    metrics = {
        "checkpoint_path": str(checkpoint_path),
        "num_windows": int(len(data["action_chunks"])),
        "train_windows": int(len(train_idx)),
        "val_windows": int(len(val_idx)),
        "device": str(device),
        "residual_mode": bool(base_d_checkpoint),
        "base_d_checkpoint": base_d_checkpoint,
        "history": history,
        "final": history[-1] if history else {},
    }
    (output_dir / "diffusion_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps({"saved": str(checkpoint_path), "device": str(device), "final": metrics["final"]}, indent=2))


if __name__ == "__main__":
    main()