from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.active_perception import ActivePerceptionConfig, ActivePerceptionCoordinator
from src.models import ActivePerceptionPolicy, JointScriptedPickCubePolicy, SineProbePolicy
from src.perception import build_pseudo_blur_config
from src.tactile import ContactFeatureExtractor, TactileBoundaryRefiner


def _success_from_info(info: dict) -> float:
    if not info:
        return 0.0
    try:
        return float(np.asarray(info.get("success", 0.0), dtype=np.float32).mean())
    except Exception:
        return 0.0


def evaluate_episode(args: argparse.Namespace, episode_index: int, output_dir: Path) -> dict:
    from src.env.wrapper import ManiSkillAgent

    seed = args.seed + episode_index
    pseudo_blur = args.scene == "pseudo_blur"
    blur_config = build_pseudo_blur_config(
        enabled=pseudo_blur,
        seed=seed,
        profile=args.pseudo_blur_profile,
        severity=args.pseudo_blur_severity,
    )
    control_mode = "pd_joint_delta_pos" if args.policy == "joint_scripted" else None
    agent = ManiSkillAgent(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=control_mode,
        render_mode=None,
        render_backend="none",
        pseudo_blur=blur_config,
    )
    obs = agent.reset(seed=seed)
    if args.policy == "joint_scripted":
        policy = JointScriptedPickCubePolicy(agent.env.action_space, use_active_probe=args.use_active_probe, probe_steps=2)
    else:
        base_policy = SineProbePolicy(agent.env.action_space)
        policy = ActivePerceptionPolicy(base_policy, agent.env.action_space) if args.use_active_probe else base_policy
    tactile = ContactFeatureExtractor()
    boundary_refiner = TactileBoundaryRefiner()
    active_perception = ActivePerceptionCoordinator(
        ActivePerceptionConfig(
            enabled=args.use_active_probe,
            uncertainty_threshold=0.5,
            probe_budget=2,
            probe_phases=("fallback", "approach"),
        )
    )

    rewards = []
    success = 0.0
    uncertainty_values = []
    visual_trigger_count = 0
    tactile_contact_count = 0
    trace_rows = []

    for step in range(args.max_steps):
        uncertainty = agent.get_visual_uncertainty(obs)
        tactile_feature = tactile.extract(obs, agent.last_info, env=agent.env)
        decision = active_perception.decide(
            step=step,
            phase=str(getattr(policy, "phase", "fallback")),
            uncertainty=uncertainty,
            tactile_feature=tactile_feature,
        )
        refinement = boundary_refiner.update(
            step=step,
            decision=decision.to_dict(),
            tactile_feature=tactile_feature,
            visual_uncertainty=float(uncertainty["uncertainty"]),
        )
        action = policy.predict(
            obs,
            step=step,
            context={
                "active_probe": bool(decision.should_probe),
                "uncertainty": uncertainty,
                "active_perception": decision.to_dict(),
                "boundary_refinement": refinement.to_dict(),
                "oracle": agent.get_task_state(),
            },
        )
        obs, reward, done, info = agent.step(action)

        rewards.append(float(reward))
        success = max(success, _success_from_info(info))
        uncertainty_values.append(float(uncertainty["uncertainty"]))
        visual_trigger_count += int(bool(uncertainty["triggered"]))
        tactile_contact_count += int(tactile_feature["contact_detected"] > 0.0)
        trace_rows.append(
            {
                **decision.to_dict(),
                "boundary_confidence": refinement.boundary_confidence,
                "post_probe_uncertainty": refinement.post_probe_uncertainty,
                "refinement_reason": refinement.reason,
                "reward": float(reward),
                "visual_triggered": bool(uncertainty["triggered"]),
            }
        )
        if done:
            break

    agent.close()
    trace_path = output_dir / f"episode_{episode_index:04d}_{args.policy}_eval_trace.csv"
    write_trace(trace_rows, trace_path)
    return {
        "episode": episode_index,
        "seed": seed,
        "env_id": args.env_id,
        "obs_mode": agent.obs_mode,
        "scene": args.scene,
        "pseudo_blur": pseudo_blur,
        "pseudo_blur_profile": blur_config.profile,
        "pseudo_blur_severity": blur_config.severity,
        "active_probe": bool(args.use_active_probe),
        "policy": args.policy,
        "checkpoint": "",
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "success": success,
        "success_rate": success,
        "mean_uncertainty": float(np.mean(uncertainty_values)) if uncertainty_values else 0.0,
        "visual_trigger_count": visual_trigger_count,
        **active_perception.summary(),
        **boundary_refiner.summary(),
        "tactile_contact_count": tactile_contact_count,
        "trace_path": str(trace_path),
        "fallback_used": bool(agent.obs_mode != args.obs_mode),
        "blur_config": asdict(blur_config),
    }


def write_trace(rows: list[dict], trace_path: Path) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = [
        "step",
        "phase",
        "ambiguity_score",
        "visual_uncertainty",
        "tactile_confidence",
        "state",
        "should_probe",
        "probe_index",
        "reason",
        "boundary_confidence",
        "post_probe_uncertainty",
        "refinement_reason",
        "reward",
        "visual_triggered",
    ]
    with trace_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = str(rows[0].get("policy", "policy")) if rows else "policy"
    json_path = output_dir / f"{policy}_eval_results.json"
    csv_path = output_dir / f"{policy}_eval_results.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "episode",
        "seed",
        "env_id",
        "obs_mode",
        "scene",
        "pseudo_blur",
        "pseudo_blur_profile",
        "pseudo_blur_severity",
        "active_probe",
        "policy",
        "checkpoint",
        "steps",
        "total_reward",
        "success",
        "success_rate",
        "mean_uncertainty",
        "visual_trigger_count",
        "decision_ambiguity_count",
        "probe_request_count",
        "contact_resolved_count",
        "refinement_count",
        "final_boundary_confidence",
        "tactile_contact_count",
        "trace_path",
        "fallback_used",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Saved {policy} eval results: {json_path}")
    print(f"Saved {policy} eval results: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate sine or joint-scripted policies for BC comparison.")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=["state", "rgbd"], default="rgbd")
    parser.add_argument("--scene", choices=["clean", "pseudo_blur"], default="pseudo_blur")
    parser.add_argument("--pseudo-blur-profile", choices=["mild", "transparent", "dark", "reflective", "low_texture"], default="mild")
    parser.add_argument("--pseudo-blur-severity", type=float, default=1.0)
    parser.add_argument("--policy", choices=["sine", "joint_scripted"], default="sine")
    parser.add_argument("--use-active-probe", action="store_true")
    parser.add_argument("--output-dir", default="results/fallback_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = []
    for episode_index in range(args.num_episodes):
        row = evaluate_episode(args, episode_index, output_dir)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
    write_summary(rows, output_dir)


if __name__ == "__main__":
    main()
