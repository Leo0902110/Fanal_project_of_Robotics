from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_bc import BCPolicyRuntime, _success_from_info, build_task_geometry_features
from src.active_perception import ActivePerceptionConfig, ActivePerceptionCoordinator
from src.env.wrapper import ManiSkillAgent
from src.perception import build_pseudo_blur_config
from src.tactile import ContactFeatureExtractor, TactileBoundaryRefiner


def _vector(value: Any, length: int = 3) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < length:
        return None
    return arr[:length]


def _distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(a - b))


def _xy_distance(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float(np.linalg.norm(a[:2] - b[:2]))


def _z_delta(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return float("nan")
    return float(a[2] - b[2])


def summarize_demo_actions(demo_dir: Path) -> dict[str, Any]:
    actions = []
    successes = 0
    episode_count = 0
    for path in sorted(demo_dir.glob("episode_*.npz")):
        data = np.load(path, allow_pickle=True)
        episode_count += 1
        metadata = data["metadata"].item() if "metadata" in data else {}
        successes += int(float(metadata.get("success", 0.0)) > 0.5)
        actions.append(data["actions"].astype(np.float32))
    if not actions:
        return {"episodes": 0, "successes": 0, "transitions": 0}

    action_matrix = np.concatenate(actions, axis=0)
    gripper = action_matrix[:, -1]
    arm = action_matrix[:, :-1]
    return {
        "episodes": episode_count,
        "successes": successes,
        "transitions": int(action_matrix.shape[0]),
        "action_dim": int(action_matrix.shape[1]),
        "arm_abs_mean": float(np.mean(np.abs(arm))) if arm.size else 0.0,
        "arm_l2_mean": float(np.mean(np.linalg.norm(arm, axis=1))) if arm.size else 0.0,
        "gripper_mean": float(np.mean(gripper)),
        "gripper_min": float(np.min(gripper)),
        "gripper_max": float(np.max(gripper)),
        "gripper_close_fraction": float(np.mean(gripper < 0.0)),
        "gripper_open_fraction": float(np.mean(gripper > 0.0)),
        "gripper_quantiles": {
            str(q): float(np.quantile(gripper, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
        },
    }


def diagnose_episode(args: argparse.Namespace, episode_index: int, output_dir: Path) -> dict[str, Any]:
    seed = args.seed + episode_index
    pseudo_blur = args.scene == "pseudo_blur"
    agent = ManiSkillAgent(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=None,
        render_backend="none",
        pseudo_blur=build_pseudo_blur_config(
            enabled=pseudo_blur,
            seed=seed,
            profile=args.pseudo_blur_profile,
            severity=args.pseudo_blur_severity,
        ),
    )
    obs = agent.reset(seed=seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    policy = BCPolicyRuntime(Path(args.checkpoint), agent.env.action_space, device)
    tactile = ContactFeatureExtractor()
    boundary_refiner = TactileBoundaryRefiner()
    active_perception = ActivePerceptionCoordinator(
        ActivePerceptionConfig(
            enabled=args.use_active_probe,
            uncertainty_threshold=0.5,
            probe_budget=2,
            probe_phases=("bc_policy", "fallback"),
        )
    )

    rows = []
    rewards = []
    success = 0.0
    min_tcp_obj_dist = float("inf")
    min_tcp_obj_xy_dist = float("inf")
    min_obj_goal_dist = float("inf")
    close_steps = 0
    contact_steps = 0

    for step in range(args.max_steps):
        uncertainty = agent.get_visual_uncertainty(obs)
        tactile_feature = tactile.extract(obs, agent.last_info, env=agent.env)
        decision = active_perception.decide(
            step=step,
            phase="bc_policy",
            uncertainty=uncertainty,
            tactile_feature=tactile_feature,
        )
        refinement = boundary_refiner.update(
            step=step,
            decision=decision.to_dict(),
            tactile_feature=tactile_feature,
            visual_uncertainty=float(uncertainty["uncertainty"]),
        )
        state = agent.get_task_state()
        tcp_pos = _vector(state.get("tcp_pos"))
        obj_pos = _vector(state.get("obj_pos"))
        goal_pos = _vector(state.get("goal_pos"))
        tcp_obj_dist = _distance(tcp_pos, obj_pos)
        tcp_obj_xy_dist = _xy_distance(tcp_pos, obj_pos)
        tcp_minus_obj_z = _z_delta(tcp_pos, obj_pos)
        obj_goal_dist = _distance(obj_pos, goal_pos)

        action = policy.predict(
            obs,
            uncertainty=float(uncertainty["uncertainty"]),
            boundary_confidence=float(refinement.boundary_confidence),
            tactile_feature=tactile_feature,
            post_probe_uncertainty=float(refinement.post_probe_uncertainty),
            task_geometry=build_task_geometry_features(state),
        )
        arm = np.asarray(action[:-1], dtype=np.float32)
        gripper = float(action[-1])
        close_steps += int(gripper < 0.0)
        contact_steps += int(float(tactile_feature.get("contact_detected", 0.0)) > 0.0)
        min_tcp_obj_dist = min(min_tcp_obj_dist, tcp_obj_dist)
        min_tcp_obj_xy_dist = min(min_tcp_obj_xy_dist, tcp_obj_xy_dist)
        min_obj_goal_dist = min(min_obj_goal_dist, obj_goal_dist)

        obs, reward, done, info = agent.step(action)
        rewards.append(float(reward))
        success = max(success, _success_from_info(info))
        rows.append(
            {
                "episode": episode_index,
                "seed": seed,
                "step": step,
                "reward": float(reward),
                "success_so_far": success,
                "tcp_obj_dist": tcp_obj_dist,
                "tcp_obj_xy_dist": tcp_obj_xy_dist,
                "tcp_minus_obj_z": tcp_minus_obj_z,
                "obj_goal_dist": obj_goal_dist,
                "gripper": gripper,
                "gripper_close_cmd": int(gripper < 0.0),
                "arm_l2": float(np.linalg.norm(arm)) if arm.size else 0.0,
                "arm_abs_mean": float(np.mean(np.abs(arm))) if arm.size else 0.0,
                "contact_detected": float(tactile_feature.get("contact_detected", 0.0)),
                "contact_strength": float(tactile_feature.get("contact_strength", 0.0)),
                "active_probe": int(bool(decision.should_probe)),
                "visual_uncertainty": float(uncertainty["uncertainty"]),
                **{f"action_{idx}": float(value) for idx, value in enumerate(action)},
            }
        )
        if done:
            break

    agent.close()
    trace_path = output_dir / f"episode_{episode_index:04d}_bc_action_trace.csv"
    write_rows(rows, trace_path)
    return {
        "episode": episode_index,
        "seed": seed,
        "steps": len(rows),
        "success": success,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "min_tcp_obj_dist": min_tcp_obj_dist,
        "min_tcp_obj_xy_dist": min_tcp_obj_xy_dist,
        "min_obj_goal_dist": min_obj_goal_dist,
        "close_fraction": close_steps / len(rows) if rows else 0.0,
        "contact_fraction": contact_steps / len(rows) if rows else 0.0,
        "trace_path": str(trace_path),
    }


def write_rows(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose BC rollout actions and gripper behavior.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--demo-dir", default="")
    parser.add_argument("--output-dir", default="results/action_diagnostics/bc")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=600)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=["state", "rgbd"], default="rgbd")
    parser.add_argument("--scene", choices=["clean", "pseudo_blur"], default="pseudo_blur")
    parser.add_argument("--pseudo-blur-profile", choices=["mild", "transparent", "dark", "reflective", "low_texture"], default="mild")
    parser.add_argument("--pseudo-blur-severity", type=float, default=1.0)
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--use-active-probe", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = [diagnose_episode(args, idx, output_dir) for idx in range(args.num_episodes)]
    summary = {
        "checkpoint": args.checkpoint,
        "demo_dir": args.demo_dir,
        "episodes": episode_rows,
        "rollout_mean": {
            key: float(np.mean([row[key] for row in episode_rows]))
            for key in (
                "success",
                "total_reward",
                "min_tcp_obj_dist",
                "min_tcp_obj_xy_dist",
                "min_obj_goal_dist",
                "close_fraction",
                "contact_fraction",
            )
        },
    }
    if args.demo_dir:
        summary["demo_action_stats"] = summarize_demo_actions(Path(args.demo_dir))
    (output_dir / "diagnosis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_rows(episode_rows, output_dir / "episode_summary.csv")
    print(json.dumps(summary["rollout_mean"], indent=2))
    if "demo_action_stats" in summary:
        print(json.dumps(summary["demo_action_stats"], indent=2))


if __name__ == "__main__":
    main()
