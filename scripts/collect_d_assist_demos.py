from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_bc import OracleGraspAssist, _success_from_info
from src.env.wrapper import ManiSkillAgent
from scripts.material_stress import (
    MATERIAL_STRESS_PROFILE_CHOICES,
    OBJECT_PROFILE_CHOICES,
    PSEUDO_BLUR_PROFILE_CHOICES,
    build_scene_blur_config,
)
from utils.d_features import TactileContactReading, extract_contact_reading


PHASE_NAMES = ("bc", "align_xy", "descend", "close", "transfer", "settle")
PHASE_TO_INDEX = {name: idx for idx, name in enumerate(PHASE_NAMES)}


def _vector(value: Any, length: int = 3) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < length:
        return None
    return arr[:length]


def _scalar(value: Any, default: float = 0.0) -> float:
    try:
        return float(np.asarray(value, dtype=np.float32).mean())
    except Exception:
        return default


def _safe_contact_reading(env) -> TactileContactReading:
    try:
        return extract_contact_reading(env)
    except Exception:
        zeros = np.zeros(3, dtype=np.float32)
        return TactileContactReading(
            feature_vector=np.zeros(11, dtype=np.float32),
            named={
                "left_force": zeros,
                "right_force": zeros,
                "net_force": zeros,
                "left_force_norm": 0.0,
                "right_force_norm": 0.0,
                "net_force_norm": 0.0,
                "contact_active": 0.0,
                "finger_links_found": 0.0,
            },
        )


def build_d_tactile_features(task_state: dict, info: dict, contact: TactileContactReading) -> np.ndarray:
    tcp_pos = _vector(task_state.get("tcp_pos"))
    obj_pos = _vector(task_state.get("obj_pos"))
    goal_pos = _vector(task_state.get("goal_pos"))
    tcp_to_obj = obj_pos - tcp_pos if tcp_pos is not None and obj_pos is not None else np.zeros(3, dtype=np.float32)
    obj_to_goal = goal_pos - obj_pos if goal_pos is not None and obj_pos is not None else np.zeros(3, dtype=np.float32)
    qpos = _vector(task_state.get("qpos"), 7)
    qvel_norm = 0.0

    tactile_surrogate = np.asarray(
        [
            _scalar(info.get("is_grasped", 0.0)),
            float(np.linalg.norm(tcp_to_obj)),
            float(np.linalg.norm(obj_to_goal)),
            qvel_norm,
            _success_from_info(info),
            _scalar(info.get("is_robot_static", 0.0)),
            float(contact.named.get("left_force_norm", 0.0)),
            float(contact.named.get("right_force_norm", 0.0)),
            float(contact.named.get("net_force_norm", 0.0)),
            float(contact.named.get("contact_active", 0.0)),
        ],
        dtype=np.float32,
    )
    del qpos
    return np.concatenate(
        [
            tactile_surrogate,
            contact.feature_vector.astype(np.float32, copy=False),
            tcp_to_obj.astype(np.float32, copy=False),
            obj_to_goal.astype(np.float32, copy=False),
        ]
    ).astype(np.float32, copy=False)


def build_vision_features(uncertainty: dict, vision_dim: int) -> np.ndarray:
    values = np.zeros(vision_dim, dtype=np.float32)
    base = np.asarray(
        [
            float(uncertainty.get("uncertainty", 0.0)),
            float(uncertainty.get("missing_ratio", 0.0)),
            float(uncertainty.get("center_missing_ratio", 0.0)),
            float(uncertainty.get("normalized_depth_std", 0.0)),
            float(bool(uncertainty.get("triggered", False))),
        ],
        dtype=np.float32,
    )
    values[: min(values.size, base.size)] = base[: values.size]
    return values


def collect_episode(args: argparse.Namespace, episode_index: int, output_dir: Path) -> dict:
    seed = args.seed + episode_index
    object_profile = args.object_profile if args.scene == "material_object" else "default"
    blur_config = build_scene_blur_config(
        scene=args.scene,
        seed=seed,
        object_profile=object_profile,
        pseudo_blur_profile=args.pseudo_blur_profile,
        pseudo_blur_severity=args.pseudo_blur_severity,
        material_visual_stress=args.material_visual_stress,
        material_stress_profile=args.material_stress_profile,
    )
    agent = ManiSkillAgent(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        control_mode="pd_joint_pos",
        render_mode=None,
        render_backend="none",
        object_profile=object_profile,
        pseudo_blur=blur_config,
    )
    obs = agent.reset(seed=seed)
    assist = OracleGraspAssist(
        agent.env.action_space,
        xy_threshold=args.assist_xy_threshold,
        z_threshold=args.assist_z_threshold,
        target_z=args.assist_target_z,
        arm_gain=args.assist_arm_gain,
        max_arm_command=args.assist_max_arm_command,
        transfer_gain=args.assist_transfer_gain,
        settle_threshold=args.assist_settle_threshold,
        joint_position_scale=args.assist_joint_position_scale,
    )

    vision_features = []
    tactile_features = []
    target_delta_actions = []
    executed_actions = []
    target_probe_flags = []
    target_uncertainty = []
    target_phase_labels = []
    phase_names = []
    rewards = []
    success = 0.0

    for step in range(args.max_steps):
        uncertainty = agent.get_visual_uncertainty(obs)
        task_state = agent.get_task_state()
        contact = _safe_contact_reading(agent.env)
        zero_bc_action = np.zeros(agent.env.action_space.shape, dtype=np.float32)
        action = assist.apply(zero_bc_action, task_state, agent.last_info, step)
        delta_action = assist.last_delta_action if assist.last_delta_action is not None else action
        phase = assist.phase

        vision_features.append(build_vision_features(uncertainty, args.vision_dim))
        tactile_features.append(build_d_tactile_features(task_state, agent.last_info, contact))
        target_delta_actions.append(np.asarray(delta_action, dtype=np.float32).reshape(-1))
        executed_actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        target_probe_flags.append(0.0)
        target_uncertainty.append(float(uncertainty.get("uncertainty", 0.0)))
        target_phase_labels.append(PHASE_TO_INDEX.get(phase, 0))
        phase_names.append(phase)

        obs, reward, done, info = agent.step(action)
        rewards.append(float(reward))
        success = max(success, _success_from_info(info))
        if done:
            break

    agent.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"episode_{episode_index:04d}.npz"
    metadata = {
        "episode": episode_index,
        "seed": seed,
        "success": success,
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "scene": args.scene,
        "object_profile": object_profile,
        "pseudo_blur_profile": blur_config.profile,
        "pseudo_blur_severity": blur_config.severity,
        "material_visual_stress": bool(args.material_visual_stress),
        "control_mode": "pd_joint_pos",
        "phase_names": PHASE_NAMES,
        "teacher": "oracle_grasp_assist_pd_joint_pos",
    }
    np.savez_compressed(
        path,
        vision_features=np.asarray(vision_features, dtype=np.float32),
        tactile_features=np.asarray(tactile_features, dtype=np.float32),
        target_actions=np.asarray(target_delta_actions, dtype=np.float32),
        executed_actions=np.asarray(executed_actions, dtype=np.float32),
        target_probe_flags=np.asarray(target_probe_flags, dtype=np.float32),
        target_uncertainty=np.asarray(target_uncertainty, dtype=np.float32),
        target_phase_labels=np.asarray(target_phase_labels, dtype=np.int64),
        phase_names=np.asarray(phase_names, dtype=object),
        rewards=np.asarray(rewards, dtype=np.float32),
        metadata=np.asarray(metadata, dtype=object),
    )
    return {"path": str(path), **metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect D-policy distillation demos from the 90% assist teacher.")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=900)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=["state", "rgbd"], default="rgbd")
    parser.add_argument("--scene", choices=["clean", "pseudo_blur", "material_object"], default="pseudo_blur")
    parser.add_argument("--object-profile", choices=OBJECT_PROFILE_CHOICES, default="default")
    parser.add_argument("--pseudo-blur-profile", choices=PSEUDO_BLUR_PROFILE_CHOICES, default="mild")
    parser.add_argument("--pseudo-blur-severity", type=float, default=1.0)
    parser.add_argument(
        "--material-visual-stress",
        action="store_true",
        help="Apply material-matched pseudo-blur to observations even when scene=material_object.",
    )
    parser.add_argument("--material-stress-profile", choices=MATERIAL_STRESS_PROFILE_CHOICES, default="auto")
    parser.add_argument("--vision-dim", type=int, default=32)
    parser.add_argument("--output-dir", default="data/demos/pickcube_d_assist_teacher")
    parser.add_argument("--assist-xy-threshold", type=float, default=0.02)
    parser.add_argument("--assist-z-threshold", type=float, default=0.012)
    parser.add_argument("--assist-target-z", type=float, default=0.01)
    parser.add_argument("--assist-arm-gain", type=float, default=10.0)
    parser.add_argument("--assist-max-arm-command", type=float, default=1.0)
    parser.add_argument("--assist-transfer-gain", type=float, default=2.0)
    parser.add_argument("--assist-settle-threshold", type=float, default=0.028)
    parser.add_argument("--assist-joint-position-scale", type=float, default=0.36)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = []
    for episode_index in range(args.start_index, args.start_index + args.num_episodes):
        row = collect_episode(args, episode_index, output_dir)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    success_count = sum(float(row["success"]) > 0.0 for row in rows)
    print(json.dumps({"episodes": len(rows), "successes": success_count, "success_rate": success_count / max(len(rows), 1)}, indent=2))


if __name__ == "__main__":
    main()