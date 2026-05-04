# 用途: 加载环境训练 checkpoint，运行评估并保存 per-episode 诊断日志。
# Purpose: Evaluate an environment-trained checkpoint and save per-episode diagnostics.

from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_baseline.models.diffusion_net import DiffusionPolicyNet
from diffusion_baseline.models.encoder import CNNEncoder
from diffusion_baseline.models.schedule import DiffusionSchedule
from diffusion_baseline.utils.collector import EnvCollector, StepResult


REPR_DIM = 128
STATE_DIM = 32
ACTION_HORIZON = 8
NUM_DIFFUSION_STEPS = 100


class EnvPolicy(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.encoder = CNNEncoder(in_channels=3, repr_dim=REPR_DIM)
        self.diffusion_net = DiffusionPolicyNet(REPR_DIM, STATE_DIM, action_dim, ACTION_HORIZON)

    def forward(self, images: torch.Tensor, states: torch.Tensor, noisy_actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(images)
        return self.diffusion_net(encoded, states.float(), noisy_actions.float(), timesteps)


class EvalEpisodeLogger:
    def __init__(self, log_dir: Path) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / "eval_episodes.jsonl"
        self.records: list[dict[str, Any]] = []

    def write_episode(self, episode_id: int, steps: list[StepResult], total_reward: float, success: bool) -> dict[str, Any]:
        raw_actions = [v for s in steps if s.action_before_clamp is not None for v in s.action_before_clamp.float().reshape(-1).tolist()]
        clamped_actions = [v for s in steps if s.clamped_action is not None for v in s.clamped_action.float().reshape(-1).tolist()]
        rewards = [s.reward for s in steps]
        record = {
            "episode_id": episode_id,
            "mode": "eval",
            "total_reward": total_reward,
            "success_flag": success,
            "num_steps": len(steps),
            "actions_summary": {"raw": self._summary(raw_actions), "clamped": self._summary(clamped_actions)},
            "reward_summary": self._summary(rewards),
            "obs_sample": {
                "first": self._obs_summary(steps[0].prev_image, steps[0].prev_state) if steps else None,
                "last": self._obs_summary(steps[-1].image, steps[-1].state) if steps else None,
            },
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        self.records.append(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def write_summary(self) -> Path:
        returns = [float(r["total_reward"]) for r in self.records]
        successes = [bool(r["success_flag"]) for r in self.records]
        best = max(self.records, key=lambda r: r["total_reward"], default=None)
        worst = min(self.records, key=lambda r: r["total_reward"], default=None)
        for record in self.records:
            record["rank_tag"] = "best" if record is best else "worst" if record is worst else "middle"
        summary = {
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "std_return": float(np.std(returns)) if returns else 0.0,
            "best_episode": best,
            "worst_episode": worst,
            "episodes_path": str(self.path),
        }
        summary_path = self.log_dir / "eval_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary_path

    def _summary(self, values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        arr = np.asarray(values, dtype=np.float32)
        return {"count": int(arr.size), "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}

    def _obs_summary(self, image: torch.Tensor | None, state: torch.Tensor | None) -> dict[str, Any] | None:
        if image is None or state is None:
            return None
        image_f = image.detach().cpu().float()
        state_f = state.detach().cpu().float()
        return {
            "rgb_shape": list(image.shape),
            "rgb_mean": float(image_f.mean()),
            "rgb_min": int(image.min()),
            "rgb_max": int(image.max()),
            "state_head": [float(x) for x in state_f[:8].tolist()],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a minimal env-trained Diffusion Policy checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--env_id", type=str, default=None)
    parser.add_argument("--env_backend", type=str, default=None, choices=["auto", "maniskill", "fallback"])
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--max_episode_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[警告] CUDA 不可用，已切换到 CPU。当前安装为 CUDA 版 PyTorch，兼容 CPU 运算。")
        return torch.device("cpu")
    return torch.device(requested_device)


@torch.no_grad()
def sample_policy_action(
    model: EnvPolicy,
    schedule: DiffusionSchedule,
    image: torch.Tensor,
    state: torch.Tensor,
    action_dim: int,
    action_low: torch.Tensor,
    action_high: torch.Tensor,
    sample_steps: int,
    device: torch.device,
) -> torch.Tensor:
    images = image.unsqueeze(0).to(device)
    states = state.unsqueeze(0).to(device).float()
    action_seq = torch.randn(1, ACTION_HORIZON, action_dim, device=device)
    timesteps = torch.linspace(schedule.num_timesteps - 1, 0, steps=max(sample_steps, 1), device=device).long()
    for timestep in timesteps:
        t = torch.full((1,), int(timestep.item()), device=device, dtype=torch.long)
        pred_noise = model(images, states, action_seq, t)
        alpha_bar = schedule.alphas_cumprod[t].reshape(1, 1, 1).to(device)
        pred_x0 = (action_seq - torch.sqrt(1.0 - alpha_bar) * pred_noise) / torch.sqrt(alpha_bar)
        action_seq = pred_x0.clamp(-5.0, 5.0)
    raw_action = action_seq[:, -1, :].squeeze(0)
    assert raw_action.shape == (action_dim,), f"sampled action shape mismatch: {tuple(raw_action.shape)}"
    return raw_action.detach().cpu()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    print(f"torch_version={torch.__version__}")
    print(f"torch_cuda_version={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device={device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    action_dim = int(config["action_dim"])
    env_id = args.env_id or config.get("env_id", "PickCube-v1")
    env_backend = args.env_backend or config.get("env_backend", "auto")
    sample_steps = args.sample_steps or int(config.get("sample_steps", 10))

    model = EnvPolicy(action_dim=action_dim).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    schedule = DiffusionSchedule(num_timesteps=NUM_DIFFUSION_STEPS, device=device)
    action_low = checkpoint["action_low"].float()
    action_high = checkpoint["action_high"].float()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger = EvalEpisodeLogger(Path(f"diffusion_baseline/logs/env_eval_{run_id}"))
    collector = EnvCollector(env_id=env_id, state_dim=STATE_DIM, seed=args.seed, max_episode_steps=args.max_episode_steps, env_backend=env_backend)
    print(f"collector_env_name={collector.env_name}")
    print(f"checkpoint_step={checkpoint['step']}")

    returns: list[float] = []
    successes: list[float] = []
    action_mins: list[float] = []
    action_maxs: list[float] = []
    last_action_shape: tuple[int, ...] | None = None

    for episode in range(1, args.episodes + 1):
        collector.reset()
        steps: list[StepResult] = []
        episode_return = 0.0
        success = False
        for _ in range(args.max_episode_steps):
            image, state = collector.current_tensors()
            action = sample_policy_action(model, schedule, image, state, action_dim, action_low, action_high, sample_steps, device)
            last_action_shape = tuple(action.shape)
            action_mins.append(float(action.min()))
            action_maxs.append(float(action.max()))
            result = collector.step(action, buffer=None)
            steps.append(result)
            episode_return += result.reward
            success = success or bool(result.info.get("success", False))
            if result.done:
                break
        record = logger.write_episode(episode, steps, episode_return, success)
        returns.append(episode_return)
        successes.append(1.0 if success else 0.0)
        print(f"episode={episode:03d} return={episode_return:.6f} success={success} steps={record['num_steps']}")

    summary_path = logger.write_summary()
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    success_rate = float(np.mean(successes))
    print(f"sampled_action_shape={last_action_shape}")
    print(f"sampled_action_min={min(action_mins):.6f}")
    print(f"sampled_action_max={max(action_maxs):.6f}")
    print(f"success_rate={success_rate:.6f}")
    print(f"mean_return={mean_return:.6f}")
    print(f"std_return={std_return:.6f}")
    print(f"episode_log_path={logger.path}")
    print(f"summary_path={summary_path}")
    if success_rate == 0.0 and collector.env_name == "FallbackReachEnv":
        print("diagnosis_note=fallback env success_rate=0，检查 eval_summary.json 的 best/worst episode 与 action 饱和情况。")
    print("ENV_EVAL_SUCCESS")
    collector.close()


if __name__ == "__main__":
    main()
