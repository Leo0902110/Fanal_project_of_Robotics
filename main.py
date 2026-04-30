from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.env.wrapper import ManiSkillAgent
from src.models import ActivePerceptionPolicy, SineProbePolicy
from src.perception import PseudoBlurConfig
from src.tactile import ContactFeatureExtractor


def _success_from_info(info: dict) -> float:
    if not info:
        return 0.0
    value = info.get("success", 0.0)
    try:
        return float(np.asarray(value, dtype=np.float32).mean())
    except Exception:
        return 0.0


def run_episode(
    name: str,
    env_id: str,
    obs_mode: str,
    max_steps: int,
    output_dir: Path,
    pseudo_blur: bool,
    active_probe: bool,
    seed: int,
    save_video: bool,
) -> dict:
    blur_config = PseudoBlurConfig(enabled=pseudo_blur, seed=seed)
    render_mode = "rgb_array" if save_video else None
    render_backend = None if save_video else "none"
    agent = ManiSkillAgent(
        env_id=env_id,
        obs_mode=obs_mode,
        render_mode=render_mode,
        render_backend=render_backend,
        pseudo_blur=blur_config,
    )
    obs = agent.reset(seed=seed)
    tactile = ContactFeatureExtractor()
    base_policy = SineProbePolicy(agent.env.action_space)
    policy = ActivePerceptionPolicy(base_policy, agent.env.action_space) if active_probe else base_policy

    rewards = []
    uncertainty_values = []
    trigger_count = 0
    tactile_contact_count = 0
    success = 0.0

    for step in range(max_steps):
        uncertainty = agent.get_visual_uncertainty(obs)
        tactile_feature = tactile.extract(obs, agent.last_info)
        context = {
            "active_probe": active_probe and bool(uncertainty["triggered"]),
            "uncertainty": uncertainty,
            "tactile": tactile_feature,
        }
        action = policy.predict(obs, step=step, context=context)
        obs, reward, done, info = agent.step(action)

        rewards.append(reward)
        uncertainty_values.append(float(uncertainty["uncertainty"]))
        trigger_count += int(bool(uncertainty["triggered"]))
        tactile_contact_count += int(tactile_feature["contact_detected"] > 0.0)
        success = max(success, _success_from_info(info))
        if done:
            break

    video_path = ""
    if save_video:
        candidate = output_dir / f"{name}.mp4"
        saved = agent.save_video(str(candidate))
        video_path = saved or ""
    agent.close()

    return {
        "experiment": name,
        "env_id": env_id,
        "obs_mode": agent.obs_mode,
        "pseudo_blur": pseudo_blur,
        "active_probe": active_probe,
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "success": success,
        "mean_uncertainty": float(np.mean(uncertainty_values)) if uncertainty_values else 0.0,
        "trigger_count": trigger_count,
        "tactile_contact_count": tactile_contact_count,
        "video_path": video_path,
        "blur_config": asdict(blur_config),
    }


def write_results(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "mvp_results.json"
    csv_path = output_dir / "mvp_results.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "experiment",
        "env_id",
        "obs_mode",
        "pseudo_blur",
        "active_probe",
        "steps",
        "total_reward",
        "success",
        "mean_uncertainty",
        "trigger_count",
        "tactile_contact_count",
        "video_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"结果已保存：{json_path}")
    print(f"结果已保存：{csv_path}")


def build_experiments(mode: str) -> list[dict]:
    if mode == "smoke":
        return [
            {"name": "smoke_clean", "pseudo_blur": False, "active_probe": False},
        ]
    return [
        {"name": "baseline_clean", "pseudo_blur": False, "active_probe": False},
        {"name": "baseline_pseudo_blur", "pseudo_blur": True, "active_probe": False},
        {"name": "active_tactile_mvp", "pseudo_blur": True, "active_probe": True},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Colab-ready MVP runner for vision-tactile active perception.")
    parser.add_argument("--mode", choices=["smoke", "mvp"], default="mvp")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=["state", "rgbd"], default="rgbd")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/mvp")
    parser.add_argument("--no-video", action="store_true", help="Disable video export for faster Colab runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, config in enumerate(build_experiments(args.mode)):
        print(f"\n开始实验：{config['name']}")
        row = run_episode(
            name=config["name"],
            env_id=args.env_id,
            obs_mode=args.obs_mode,
            max_steps=args.max_steps,
            output_dir=output_dir,
            pseudo_blur=config["pseudo_blur"],
            active_probe=config["active_probe"],
            seed=args.seed + idx,
            save_video=not args.no_video,
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    write_results(rows, output_dir)


if __name__ == "__main__":
    main()
