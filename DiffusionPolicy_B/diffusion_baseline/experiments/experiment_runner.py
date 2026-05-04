"""
超参搜索实验运行器（直接导入模式，避免子进程问题）
Usage:
    python diffusion_baseline/experiments/experiment_runner.py --configs experiments/configs/grid.json --device cuda
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

RESULTS_CSV_FIELDS = [
    "run_id", "timestamp", "exit_code",
    "warmup_steps", "batch_size", "num_diffusion_steps", "amp",
    "num_steps", "device", "env_id", "seed",
    "success_rate", "return_min", "return_mean", "return_max", "final_loss",
    "duration_sec",
]


def _run_training_experiment(config: dict[str, Any], run_id: int) -> dict[str, Any]:
    import os
    os.environ.setdefault("MANISKILL_GPU_SIM", "0")

    import numpy as np
    import torch
    from contextlib import nullcontext

    from diffusion_baseline.training.run_env_train import (
        EpisodeLogger, EnvCollector, DiffusionSchedule, EnvPolicy,
        sample_policy_action, diagnose,
        REPR_DIM, STATE_DIM, ACTION_HORIZON,
    )
    from diffusion_baseline.utils.buffer import ReplayBuffer

    device = torch.device(config.get("device", "cuda"))
    warmup_steps = int(config.get("warmup_steps", 100))
    num_steps = int(config.get("num_steps", 500))
    batch_size = int(config.get("batch_size", 16))
    num_diffusion_steps = int(config.get("num_diffusion_steps", 20))
    amp_enabled = bool(config.get("amp", True))
    env_id = config.get("env_id", "PickCube-v1")
    seed = int(config.get("seed", 0))

    print(f"  [{run_id+1}] torch={torch.__version__} cuda={torch.cuda.is_available()} device={device}", flush=True)

    run_id_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(f"diffusion_baseline/logs/env_train_{run_id_str}_exp{run_id}")
    logger = EpisodeLogger(log_dir, "train")

    collector = EnvCollector(env_id=env_id, state_dim=STATE_DIM, seed=seed, env_backend="fallback")
    buffer = ReplayBuffer(capacity=int(config.get("buffer_size", 10000)))
    print(f"  [{run_id+1}] collector_env={collector.env_name} action_dim={collector.action_dim}", flush=True)
    for _ in range(warmup_steps):
        action = collector.heuristic_action() if collector.env_name == "FallbackReachEnv" else collector.random_action()
        record = logger.add_step(collector.step(action, buffer=buffer), mode="warmup")
        if record:
            pass

    model = EnvPolicy(action_dim=collector.action_dim).to(device)
    schedule = DiffusionSchedule(num_timesteps=num_diffusion_steps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_step = 0
    last_loss = float("nan")
    t0 = time.perf_counter()

    for step in range(start_step + 1, start_step + num_steps + 1):
        batch = {key: value.to(device) for key, value in buffer.sample(batch_size).items()}
        images = batch["image"]
        states = batch["state"].float()
        actions = batch["action"].float()

        action_seq = actions.unsqueeze(1).repeat(1, ACTION_HORIZON, 1)
        noise = torch.randn_like(action_seq)
        timesteps = torch.randint(0, schedule.num_timesteps, (batch_size,), device=device, dtype=torch.long)
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

        image, state = collector.current_tensors()
        policy_action = sample_policy_action(model, schedule, image, state, collector.action_dim, 10, device)
        record = logger.add_step(collector.step(policy_action, buffer=buffer), mode="train")
        if record:
            pass

        if step % 100 == 0:
            print(f"  [{run_id+1}] step={step:04d} loss={last_loss:.6f}", flush=True)

    elapsed = time.perf_counter() - t0
    logger.flush()
    _ = logger.write_summary()
    diagnosis = diagnose(logger, collector)
    collector.close()

    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "exit_code": 0,
        "warmup_steps": warmup_steps,
        "batch_size": batch_size,
        "num_diffusion_steps": num_diffusion_steps,
        "amp": amp_enabled,
        "num_steps": num_steps,
        "device": str(device),
        "env_id": env_id,
        "seed": seed,
        "success_rate": diagnosis["success_rate"],
        "return_min": diagnosis["return_min"],
        "return_mean": diagnosis["return_mean"],
        "return_max": diagnosis["return_max"],
        "final_loss": last_loss,
        "duration_sec": round(elapsed, 1),
    }


def write_results_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    total = 0
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            total = sum(1 for _ in f) - 1
    print(f"\nresults_csv={path} rows_appended={len(results)} total_rows={total}", flush=True)


def find_best(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [r for r in results if r["exit_code"] == 0 and r["success_rate"] is not None]
    if not valid:
        return None
    valid.sort(key=lambda r: (-(r["success_rate"] or 0), -(r["return_mean"] or float("-inf"))))
    return valid[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hyperparameter grid search (direct import mode).")
    parser.add_argument("--configs", type=str, required=True, help="Path to grid JSON config file.")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    config_path = Path(args.configs)
    if not config_path.exists():
        print(f"[FATAL] Config file not found: {config_path}", flush=True)
        sys.exit(1)

    with config_path.open("r", encoding="utf-8") as f:
        grid = json.load(f)
    experiments: list[dict[str, Any]] = grid.get("experiments", [])
    if not experiments:
        print("[FATAL] No experiments defined.", flush=True)
        sys.exit(1)

    print(f"total_experiments={len(experiments)} device={args.device}", flush=True)
    print("=" * 60, flush=True)
    t_start = time.perf_counter()

    results: list[dict[str, Any]] = []
    for idx, cfg in enumerate(experiments):
        cfg["device"] = cfg.get("device", args.device)
        config_display = ", ".join(
            f"{k}={cfg.get(k, '?')}"
            for k in ["warmup_steps", "batch_size", "num_diffusion_steps", "amp"]
        )
        print(f"  [{idx+1}] 开始: {config_display}", flush=True)
        row = _run_training_experiment(cfg, idx)
        results.append(row)
        print(f"  [{idx+1}] 结束: sr={row['success_rate']} r_mean={row['return_mean']} "
              f"loss={row['final_loss']} dur={row['duration_sec']}s", flush=True)

    total_sec = time.perf_counter() - t_start
    print(f"\n{'='*60}", flush=True)
    print(f"  ALL EXPERIMENTS DONE  total_time={total_sec:.1f}s  runs={len(results)}", flush=True)
    print(f"{'='*60}", flush=True)

    out_path = Path("diffusion_baseline/experiments/results.csv")
    write_results_csv(out_path, results)

    best = find_best(results)
    if best:
        print(f"\n最佳配置 (优先级: success_rate ↓, return_mean ↓):", flush=True)
        print(f"  warmup_steps={best['warmup_steps']}  batch_size={best['batch_size']}"
              f"  num_diffusion_steps={best['num_diffusion_steps']}  amp={best['amp']}", flush=True)
        print(f"  success_rate={best['success_rate']}  return_mean={best['return_mean']}"
              f"  final_loss={best['final_loss']}", flush=True)
    else:
        print("\n[WARN] 无有效实验结果。", flush=True)


if __name__ == "__main__":
    main()
