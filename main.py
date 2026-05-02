from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.models import ActivePerceptionPolicy, ScriptedPickCubePolicy, SineProbePolicy
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
    condition: str,
    env_id: str,
    obs_mode: str,
    max_steps: int,
    output_dir: Path,
    pseudo_blur: bool,
    active_probe: bool,
    seed: int,
    save_video: bool,
    policy_name: str,
) -> dict:
    from src.env.wrapper import ManiSkillAgent

    blur_config = PseudoBlurConfig(enabled=pseudo_blur, seed=seed)
    render_mode = "rgb_array" if save_video else None
    render_backend = None if save_video else "none"
    control_mode = "pd_ee_delta_pose" if policy_name == "scripted" else None
    agent = ManiSkillAgent(
        env_id=env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        render_backend=render_backend,
        pseudo_blur=blur_config,
    )

    obs = agent.reset(seed=seed)
    tactile = ContactFeatureExtractor()
    effective_policy_name = policy_name
    scripted_ready = policy_name == "scripted" and agent.obs_mode != "state"
    if policy_name == "scripted" and not scripted_ready:
        effective_policy_name = "sine_fallback"

    if effective_policy_name == "scripted":
        policy = ScriptedPickCubePolicy(agent.env.action_space, use_active_probe=active_probe)
    else:
        base_policy = SineProbePolicy(agent.env.action_space)
        policy = ActivePerceptionPolicy(base_policy, agent.env.action_space) if active_probe else base_policy

    rewards = []
    uncertainty_values = []
    visual_trigger_count = 0
    tactile_contact_count = 0
    success = 0.0

    for step in range(max_steps):
        uncertainty = agent.get_visual_uncertainty(obs)
        tactile_feature = tactile.extract(obs, agent.last_info)
        context = {
            "active_probe": active_probe and bool(uncertainty["triggered"]),
            "uncertainty": uncertainty,
            "mean_uncertainty": float(uncertainty["uncertainty"]),
            "tactile": tactile_feature,
            "oracle": agent.get_task_state(),
        }
        action = policy.predict(obs, step=step, context=context)
        obs, reward, done, info = agent.step(action)

        rewards.append(reward)
        uncertainty_values.append(float(uncertainty["uncertainty"]))
        visual_trigger_count += int(bool(uncertainty["triggered"]))
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

    trigger_count = int(getattr(policy, "probe_count", 0))
    return {
        "experiment": name,
        "condition": condition,
        "env_id": env_id,
        "obs_mode": agent.obs_mode,
        "pseudo_blur": pseudo_blur,
        "active_probe": active_probe,
        "requested_policy": policy_name,
        "effective_policy": effective_policy_name,
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "success": success,
        "success_rate": success,
        "mean_uncertainty": float(np.mean(uncertainty_values)) if uncertainty_values else 0.0,
        "trigger_count": trigger_count,
        "visual_trigger_count": visual_trigger_count,
        "tactile_contact_count": tactile_contact_count,
        "video_path": video_path,
        "fallback_used": bool(agent.obs_mode != obs_mode or effective_policy_name != policy_name),
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
        "condition",
        "env_id",
        "obs_mode",
        "pseudo_blur",
        "active_probe",
        "requested_policy",
        "effective_policy",
        "steps",
        "total_reward",
        "success",
        "success_rate",
        "mean_uncertainty",
        "trigger_count",
        "visual_trigger_count",
        "tactile_contact_count",
        "video_path",
        "fallback_used",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    print(f"Saved results: {json_path}")
    print(f"Saved results: {csv_path}")


def build_experiments(mode: str, scene: str, use_active_probe: bool) -> list[dict]:
    if mode == "smoke":
        return [
            {
                "name": "smoke_clean",
                "condition": "Smoke",
                "pseudo_blur": False,
                "active_probe": False,
            },
        ]
    if scene == "clean":
        return [
            {
                "name": "clean",
                "condition": "Clean",
                "pseudo_blur": False,
                "active_probe": use_active_probe,
            },
        ]
    if scene == "pseudo_blur":
        return [
            {
                "name": "active_probe" if use_active_probe else "pseudo_blur",
                "condition": "Active-Probe" if use_active_probe else "Pseudo-Blur",
                "pseudo_blur": True,
                "active_probe": use_active_probe,
            },
        ]
    return [
        {
            "name": "baseline_clean",
            "condition": "Clean",
            "pseudo_blur": False,
            "active_probe": False,
        },
        {
            "name": "baseline_pseudo_blur",
            "condition": "Pseudo-Blur",
            "pseudo_blur": True,
            "active_probe": False,
        },
        {
            "name": "active_tactile_mvp",
            "condition": "Active-Probe",
            "pseudo_blur": True,
            "active_probe": True,
        },
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP runner for active perception under visual pseudo-blur.")
    parser.add_argument("--mode", choices=["smoke", "mvp"], default="mvp")
    parser.add_argument("--scene", choices=["all", "clean", "pseudo_blur"], default="all")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=["state", "rgbd"], default="rgbd")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/mvp")
    parser.add_argument("--policy", choices=["sine", "scripted"], default="scripted")
    parser.add_argument("--use-active-probe", action="store_true")
    parser.add_argument("--no-video", action="store_true", help="Disable video export for faster runs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for config in build_experiments(args.mode, args.scene, args.use_active_probe):
        print(f"\nRunning experiment: {config['name']}")
        row = run_episode(
            name=config["name"],
            condition=config["condition"],
            env_id=args.env_id,
            obs_mode=args.obs_mode,
            max_steps=args.max_steps,
            output_dir=output_dir,
            pseudo_blur=config["pseudo_blur"],
            active_probe=config["active_probe"],
            seed=args.seed,
            save_video=not args.no_video,
            policy_name=args.policy,
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    write_results(rows, output_dir)


if __name__ == "__main__":
    main()
