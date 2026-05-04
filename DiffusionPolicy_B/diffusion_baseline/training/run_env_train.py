# 用途: 使用环境采样数据运行最小 Diffusion Policy 训练、记录 episode 诊断日志并保存 checkpoint。
# Purpose: Train a minimal Diffusion Policy from env samples, log episode diagnostics, and save checkpoints.
#
# Windows PowerShell:
#   .venv\Scripts\python.exe diffusion_baseline\training\run_env_train.py --batch_size 16 --num_steps 2000 --num_envs 1 --device cuda --warmup_steps 500 --buffer_size 5000 --amp True

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffusion_baseline.models.diffusion_net import DiffusionPolicyNet
from diffusion_baseline.models.encoder import CNNEncoder
from diffusion_baseline.models.schedule import DiffusionSchedule
from diffusion_baseline.utils.buffer import ReplayBuffer
from diffusion_baseline.utils.collector import EnvCollector, StepResult


REPR_DIM = 128
STATE_DIM = 32
ACTION_HORIZON = 8
NUM_DIFFUSION_STEPS = 100


class EnvPolicy(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.encoder = CNNEncoder(in_channels=3, repr_dim=REPR_DIM)
        self.diffusion_net = DiffusionPolicyNet(
            repr_dim=REPR_DIM,
            state_dim=STATE_DIM,
            action_dim=action_dim,
            action_horizon=ACTION_HORIZON,
        )

    def forward(self, images: torch.Tensor, states: torch.Tensor, noisy_actions: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(images)
        return self.diffusion_net(encoded, states.float(), noisy_actions.float(), timesteps)


class EpisodeLogger:
    def __init__(self, log_dir: Path, prefix: str) -> None:
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = log_dir / f"{prefix}_episodes.jsonl"
        self.csv_path = log_dir / f"{prefix}_episodes.csv"
        self.episode_id = 0
        self.records: list[dict[str, Any]] = []
        self._csv_header_written = False
        self._reset_current()

    def _reset_current(self) -> None:
        self.total_reward = 0.0
        self.num_steps = 0
        self.success = False
        self.raw_actions: list[float] = []
        self.clamped_actions: list[float] = []
        self.rewards: list[float] = []
        self.first_obs: dict[str, Any] | None = None
        self.last_obs: dict[str, Any] | None = None
        self.mode = "unknown"

    def add_step(self, result: StepResult, mode: str) -> dict[str, Any] | None:
        self.mode = mode
        if self.first_obs is None and result.prev_image is not None and result.prev_state is not None:
            self.first_obs = self._obs_summary(result.prev_image, result.prev_state)
        self.last_obs = self._obs_summary(result.image, result.state)
        self.total_reward += result.reward
        self.num_steps += 1
        self.success = self.success or bool(result.info.get("success", False))
        self.rewards.append(result.reward)
        if result.action_before_clamp is not None:
            self.raw_actions.extend(result.action_before_clamp.detach().cpu().float().reshape(-1).tolist())
        if result.clamped_action is not None:
            self.clamped_actions.extend(result.clamped_action.detach().cpu().float().reshape(-1).tolist())
        if result.done:
            return self.flush()
        return None

    def flush(self) -> dict[str, Any]:
        if self.num_steps == 0:
            return {}
        self.episode_id += 1
        record = {
            "episode_id": self.episode_id,
            "mode": self.mode,
            "total_reward": self.total_reward,
            "success_flag": self.success,
            "num_steps": self.num_steps,
            "actions_summary": {
                "raw": self._summary(self.raw_actions),
                "clamped": self._summary(self.clamped_actions),
            },
            "reward_summary": self._summary(self.rewards),
            "obs_sample": {"first": self.first_obs, "last": self.last_obs},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._write_csv_row(record)
        self.records.append(record)
        self._reset_current()
        return record

    def write_summary(self) -> Path:
        summary_path = self.log_dir / "train_summary.json"
        returns = [float(r["total_reward"]) for r in self.records]
        successes = [bool(r["success_flag"]) for r in self.records]
        raw_values = [v for r in self.records for v in self._summary_values(r, "raw")]
        clamped_values = [v for r in self.records for v in self._summary_values(r, "clamped")]
        best = max(self.records, key=lambda r: r["total_reward"], default=None)
        worst = min(self.records, key=lambda r: r["total_reward"], default=None)
        summary = {
            "num_episodes": len(self.records),
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "mean_return": float(np.mean(returns)) if returns else 0.0,
            "std_return": float(np.std(returns)) if returns else 0.0,
            "best_episode": best,
            "worst_episode": worst,
            "raw_action_summary": self._summary(raw_values),
            "clamped_action_summary": self._summary(clamped_values),
            "episodes_path": str(self.path),
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary_path

    def _summary_values(self, record: dict[str, Any], key: str) -> list[float]:
        item = record["actions_summary"][key]
        return [float(item["min"]), float(item["max"]), float(item["mean"])] if item["count"] else []

    def _summary(self, values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {"count": 0, "min": None, "max": None, "mean": None}
        arr = np.asarray(values, dtype=np.float32)
        return {"count": int(arr.size), "min": float(arr.min()), "max": float(arr.max()), "mean": float(arr.mean())}

    def _obs_summary(self, image: torch.Tensor, state: torch.Tensor) -> dict[str, Any]:
        image_f = image.detach().cpu().float()
        state_f = state.detach().cpu().float()
        return {
            "rgb_shape": list(image.shape),
            "rgb_mean": float(image_f.mean()),
            "rgb_min": int(image.min()),
            "rgb_max": int(image.max()),
            "state_head": [float(x) for x in state_f[:8].tolist()],
        }

    def _write_csv_row(self, record: dict[str, Any]) -> None:
        if not self._csv_header_written:
            header = "episode_id,mode,total_reward,success_flag,num_steps,raw_action_min,raw_action_max,raw_action_mean,clamped_action_min,clamped_action_max,clamped_action_mean,reward_min,reward_max,reward_mean,timestamp"
            self.csv_path.write_text(header + "\n", encoding="utf-8")
            self._csv_header_written = True
        raw_s = record["actions_summary"]["raw"]
        clamped_s = record["actions_summary"]["clamped"]
        reward_s = record["reward_summary"]
        row = (
            f'{record["episode_id"]},'
            f'{record["mode"]},'
            f'{record["total_reward"]:.6f},'
            f'{record["success_flag"]},'
            f'{record["num_steps"]},'
            f'{raw_s["min"] if raw_s["min"] is not None else ""},'
            f'{raw_s["max"] if raw_s["max"] is not None else ""},'
            f'{raw_s["mean"] if raw_s["mean"] is not None else ""},'
            f'{clamped_s["min"] if clamped_s["min"] is not None else ""},'
            f'{clamped_s["max"] if clamped_s["max"] is not None else ""},'
            f'{clamped_s["mean"] if clamped_s["mean"] is not None else ""},'
            f'{reward_s["min"] if reward_s["min"] is not None else ""},'
            f'{reward_s["max"] if reward_s["max"] is not None else ""},'
            f'{reward_s["mean"] if reward_s["mean"] is not None else ""},'
            f'{record["timestamp"]}'
        )
        with self.csv_path.open("a", encoding="utf-8") as f:
            f.write(row + "\n")


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.lower() in {"1", "true", "yes", "y", "on"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal env-collected Diffusion Policy training.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--env_id", type=str, default="PickCube-v1")
    parser.add_argument("--env_backend", type=str, default="auto", choices=["auto", "maniskill", "fallback"])
    parser.add_argument("--warmup_steps", "--warmup_size", type=int, default=500)
    parser.add_argument("--buffer_size", "--buffer_capacity", type=int, default=5000)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--sample_steps", type=int, default=10)
    parser.add_argument("--num_diffusion_steps", type=int, default=NUM_DIFFUSION_STEPS)
    parser.add_argument("--amp", type=parse_bool, default=True)
    parser.add_argument("--fallback_oracle", type=parse_bool, default=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Alias for --resume.")
    parser.add_argument("--use_tensorboard", type=parse_bool, default=True, help="Enable TensorBoard logging.")
    parser.add_argument("--use_wandb", type=parse_bool, default=False, help="Enable W&B logging.")
    parser.add_argument("--wandb_project", type=str, default="diffusion-policy-baseline")
    parser.add_argument("--wandb_name", type=str, default=None, help="W&B run name (auto-generated if not set).")
    return parser.parse_args()


def resolve_device(requested_device: str) -> torch.device:
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        print("[警告] CUDA 不可用，已切换到 CPU。当前安装为 CUDA 版 PyTorch，兼容 CPU 运算。")
        return torch.device("cpu")
    return torch.device(requested_device)


def print_runtime(device: torch.device) -> None:
    print(f"torch_version={torch.__version__}")
    print(f"torch_cuda_version={torch.version.cuda}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"cuda_device_name={torch.cuda.get_device_name(device)}")


@torch.no_grad()
def sample_policy_action(
    model: EnvPolicy,
    schedule: DiffusionSchedule,
    image: torch.Tensor,
    state: torch.Tensor,
    action_dim: int,
    sample_steps: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
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
    model.train()
    action = action_seq[:, -1, :].squeeze(0).detach().cpu()
    assert action.shape == (action_dim,), f"policy action shape mismatch: {tuple(action.shape)} vs {(action_dim,)}"
    return action


def save_checkpoint(path: Path, model: EnvPolicy, optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, step: int, collector: EnvCollector, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "step": step,
            "config": {
                "repr_dim": REPR_DIM,
                "state_dim": STATE_DIM,
                "action_horizon": ACTION_HORIZON,
                "action_dim": collector.action_dim,
                "num_diffusion_steps": args.num_diffusion_steps,
                "env_id": args.env_id,
                "env_backend": args.env_backend,
                "env_name": collector.env_name,
                "sample_steps": args.sample_steps,
            },
            "action_low": torch.as_tensor(collector.action_low, dtype=torch.float32),
            "action_high": torch.as_tensor(collector.action_high, dtype=torch.float32),
        },
        path,
    )
    print(f"checkpoint_saved={path} size_bytes={path.stat().st_size}")


def diagnose(logger: EpisodeLogger, collector: EnvCollector) -> dict[str, float]:
    result: dict[str, float] = {}
    if not logger.records:
        print("diagnosis=no_completed_episodes")
        return result
    successes = [bool(r["success_flag"]) for r in logger.records]
    returns = np.asarray([float(r["total_reward"]) for r in logger.records], dtype=np.float32)
    success_rate = float(np.mean(successes))
    print(f"diagnosis_success_rate={success_rate:.6f}")
    print(f"diagnosis_return_min={returns.min():.6f} mean={returns.mean():.6f} max={returns.max():.6f}")
    result["success_rate"] = success_rate
    result["return_min"] = float(returns.min())
    result["return_mean"] = float(returns.mean())
    result["return_max"] = float(returns.max())
    if success_rate == 0.0 and collector.env_name == "FallbackReachEnv":
        print("diagnosis_note=fallback env success_flag 全为 False，说明当前 policy 尚未稳定到达 distance<0.15。")
    raw = [r["actions_summary"]["raw"] for r in logger.records if r["actions_summary"]["raw"]["count"]]
    if raw:
        raw_min = min(float(x["min"]) for x in raw)
        raw_max = max(float(x["max"]) for x in raw)
        print(f"diagnosis_raw_action_range=({raw_min:.6f}, {raw_max:.6f})")
        if raw_min <= float(collector.action_low.min()) or raw_max >= float(collector.action_high.max()):
            print("diagnosis_note=action_before_clamp 触及动作边界；可尝试降低采样噪声、增加模型容量或加入动作平滑/奖励 shaping。")
    if returns.max() < 0:
        print("diagnosis_note=reward 全为负是 reaching 距离代价的正常表现；可加入成功奖励或距离差分 reward shaping。")
    return result


def main() -> None:
    args = parse_args()
    if args.num_envs != 1:
        print("[警告] 当前最小集成仅实现单环境同步采样，已按 num_envs=1 运行。")
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    print_runtime(device)
    print(f"amp_enabled={amp_enabled}")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"diffusion_baseline/logs/env_train_{run_id}")
    logger = EpisodeLogger(log_dir, "train")

    writer: SummaryWriter | None = None
    if args.use_tensorboard:
        tb_dir = log_dir / "tensorboard"
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"tensorboard_logdir={tb_dir}")

    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or f"env_train_{run_id}",
                config=vars(args),
            )
            print(f"wandb_initialized project={args.wandb_project}")
        except ImportError:
            print("[警告] wandb 未安装，跳过 W&B 日志。pip install wandb")

    collector = EnvCollector(env_id=args.env_id, state_dim=STATE_DIM, seed=args.seed, env_backend=args.env_backend)
    print(f"collector_env_name={collector.env_name}")
    print(f"action_shape={collector.action_shape} action_dim={collector.action_dim}")
    print(f"action_low_min={collector.action_low.min():.4f} action_high_max={collector.action_high.max():.4f}")

    buffer = ReplayBuffer(capacity=args.buffer_size)
    print(f"warmup_collect_steps={args.warmup_steps}")
    for _ in range(args.warmup_steps):
        action = collector.heuristic_action() if args.fallback_oracle and collector.env_name == "FallbackReachEnv" else collector.random_action()
        record = logger.add_step(collector.step(action, buffer=buffer), mode="warmup")
        if record:
            print(f"episode_done id={record['episode_id']} mode=warmup return={record['total_reward']:.6f} success={record['success_flag']}")
    print(f"buffer_size_after_warmup={len(buffer)}")

    model = EnvPolicy(action_dim=collector.action_dim).to(device)
    schedule = DiffusionSchedule(num_timesteps=args.num_diffusion_steps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_step = 0
    resume_path = args.resume or args.checkpoint
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt.get("step", 0)
        print(f"[RESUME] Loaded checkpoint from {resume_path} at step={start_step}")

    last_loss = float("nan")
    for step in range(start_step + 1, start_step + args.num_steps + 1):
        batch = {key: value.to(device) for key, value in buffer.sample(args.batch_size).items()}
        images = batch["image"]
        states = batch["state"].float()
        actions = batch["action"].float()
        assert images.ndim == 4 and images.shape[-1] == 3, f"image batch must be [B,H,W,3], got {tuple(images.shape)}"
        assert states.shape == (args.batch_size, STATE_DIM), f"state batch shape mismatch: {tuple(states.shape)}"
        assert actions.shape == (args.batch_size, collector.action_dim), f"action batch shape mismatch: {tuple(actions.shape)}"

        action_seq = actions.unsqueeze(1).repeat(1, ACTION_HORIZON, 1)
        noise = torch.randn_like(action_seq)
        timesteps = torch.randint(0, schedule.num_timesteps, (args.batch_size,), device=device, dtype=torch.long)
        noisy_actions = schedule.q_sample(action_seq, timesteps, noise)
        autocast_ctx = torch.amp.autocast(device_type=device.type, enabled=amp_enabled) if device.type == "cuda" else nullcontext()
        with autocast_ctx:
            pred_noise = model(images, states, noisy_actions, timesteps)
            loss = torch.nn.functional.mse_loss(pred_noise.float(), noise.float())

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        last_loss = float(loss.detach().cpu())
        if writer is not None:
            writer.add_scalar("train/loss", last_loss, step)
        if wandb_run is not None:
            wandb_run.log({"train/loss": last_loss}, step=step)

        image, state = collector.current_tensors()
        policy_action = sample_policy_action(model, schedule, image, state, collector.action_dim, args.sample_steps, device)
        record = logger.add_step(collector.step(policy_action, buffer=buffer), mode="train")
        if record:
            print(f"episode_done id={record['episode_id']} mode=train return={record['total_reward']:.6f} success={record['success_flag']}")
            if writer is not None:
                returns = [float(r["total_reward"]) for r in logger.records]
                successes = [bool(r["success_flag"]) for r in logger.records]
                writer.add_scalar("episode/return", record["total_reward"], record["episode_id"])
                writer.add_scalar("episode/success_rate", float(np.mean(successes)), record["episode_id"])
                writer.add_scalar("episode/mean_return", float(np.mean(returns)), record["episode_id"])
                writer.add_scalar("episode/std_return", float(np.std(returns)), record["episode_id"])
            if wandb_run is not None:
                wandb_run.log({
                    "episode/return": record["total_reward"],
                    "episode/success": int(record["success_flag"]),
                }, step=step)

        if step % 50 == 0 or step == 1:
            print(f"step={step:04d} loss={last_loss:.6f} buffer_size={len(buffer)} log_dir={log_dir}")
        if step % args.save_every == 0:
            save_checkpoint(Path(f"diffusion_baseline/checkpoints/ckpt_step{step}.pt"), model, optimizer, scaler, step, collector, args)

    final_path = Path(f"diffusion_baseline/checkpoints/ckpt_step{start_step + args.num_steps}.pt")
    if (start_step + args.num_steps) % args.save_every != 0:
        save_checkpoint(final_path, model, optimizer, scaler, start_step + args.num_steps, collector, args)
    logger.flush()
    summary_path = logger.write_summary()
    diagnose(logger, collector)
    print(f"episode_log_path={logger.path}")
    print(f"summary_path={summary_path}")
    print(f"ENV_TRAIN_SUCCESS final_step={start_step + args.num_steps} final_loss={last_loss:.6f} final_checkpoint={final_path}")
    if writer is not None:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    collector.close()


if __name__ == "__main__":
    main()
