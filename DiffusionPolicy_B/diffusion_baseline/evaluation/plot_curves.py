"""
Plot training curves from TensorBoard event files or per-episode CSV.
Usage:
    python diffusion_baseline/evaluation/plot_curves.py --logdir diffusion_baseline/logs/env_train_YYYYMMDD_HHMMSS/
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_from_csv(csv_path: Path, output_dir: Path) -> None:
    if not csv_path.exists():
        print(f"[WARN] CSV not found: {csv_path}")
        return

    data = np.genfromtxt(str(csv_path), delimiter=",", names=True, dtype=None, encoding="utf-8")
    episodes = data["episode_id"].astype(int)
    returns = data["total_reward"].astype(float)
    successes = data["success_flag"].astype(int)

    window = max(1, len(returns) // 10)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Training Curves - {csv_path.parent.name}", fontsize=14)

    # Loss would come from TB; here we show what CSV has
    # 1. Episode return
    ax = axes[0, 0]
    ax.plot(episodes, returns, "b-", alpha=0.4, linewidth=0.8, label="Per-episode")
    if len(returns) >= window:
        smoothed = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(episodes[window - 1 :], smoothed, "b-", linewidth=2, label=f"Smoothed (w={window})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Return")
    ax.set_title("Episode Return")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Success rate (cumulative)
    ax = axes[0, 1]
    cum_success = np.cumsum(successes) / np.arange(1, len(successes) + 1)
    ax.plot(episodes, cum_success, "g-", linewidth=1.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title("Cumulative Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # 3. Return histogram
    ax = axes[1, 0]
    ax.hist(returns, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(returns), color="red", linestyle="--", linewidth=1.5, label=f"Mean={np.mean(returns):.3f}")
    ax.set_xlabel("Return")
    ax.set_ylabel("Count")
    ax.set_title("Return Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Raw vs Clamped action range
    ax = axes[1, 1]
    raw_keys = ["raw_action_min", "raw_action_max", "raw_action_mean"]
    clamped_keys = ["clamped_action_min", "clamped_action_max", "clamped_action_mean"]
    if all(k in data.dtype.names for k in raw_keys):
        raw_min = data["raw_action_min"].astype(float)
        raw_max = data["raw_action_max"].astype(float)
        clamped_min = data["clamped_action_min"].astype(float)
        clamped_max = data["clamped_action_max"].astype(float)
        ax.fill_between(episodes, raw_min, raw_max, alpha=0.2, color="red", label="Raw action range")
        ax.fill_between(episodes, clamped_min, clamped_max, alpha=0.3, color="blue", label="Clamped action range")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Action Value")
        ax.set_title("Action Range (Raw vs Clamped)")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No action data in CSV", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Action Range")

    plt.tight_layout()
    out_path = output_dir / "training_curves.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"plot_saved={out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training curves from CSV.")
    parser.add_argument("--logdir", type=str, required=True, help="Path to training log directory.")
    args = parser.parse_args()

    log_dir = Path(args.logdir)
    csv_path = log_dir / "train_episodes.csv"
    plot_from_csv(csv_path, log_dir)


if __name__ == "__main__":
    main()
