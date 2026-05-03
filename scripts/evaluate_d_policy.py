from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.d_policy import ActiveTactilePolicy
from scripts.collect_d_assist_demos import build_d_tactile_features, build_vision_features, _safe_contact_reading, _vector
from scripts.evaluate_bc import _scalar, _success_from_info
from src.env.wrapper import ManiSkillAgent
from src.perception import build_pseudo_blur_config


def load_policy(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    phase_names = tuple(checkpoint.get("phase_names", ("bc", "align_xy", "descend", "close", "transfer", "settle")))
    policy = ActiveTactilePolicy(
        action_dim=int(checkpoint["action_dim"]),
        vision_dim=int(checkpoint["vision_dim"]),
        tactile_dim=int(checkpoint["tactile_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_phase_classes=len(phase_names),
    ).to(device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.eval()
    normalizers = {
        "vision_mean": _to_numpy(checkpoint["vision_mean"]),
        "vision_std": _to_numpy(checkpoint["vision_std"]),
        "tactile_mean": _to_numpy(checkpoint["tactile_mean"]),
        "tactile_std": _to_numpy(checkpoint["tactile_std"]),
    }
    return policy, phase_names, normalizers


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def _qpos_delta_to_joint_pos(action_delta: np.ndarray, task_state: dict, action_space, scale: float) -> np.ndarray:
    qpos = _vector(task_state.get("qpos"), 7)
    if qpos is None:
        return np.clip(action_delta, action_space.low, action_space.high)
    action = np.zeros(action_space.shape, dtype=np.float32)
    action[:7] = qpos[:7] + scale * action_delta[:7]
    action[-1] = action_delta[-1]
    return np.clip(action, action_space.low, action_space.high)


def predict_action(policy, phase_names, normalizers, obs, task_state, info, agent, device, vision_dim: int):
    uncertainty = agent.get_visual_uncertainty(obs)
    contact = _safe_contact_reading(agent.env)
    vision = build_vision_features(uncertainty, vision_dim)
    tactile = build_d_tactile_features(task_state, info, contact)
    vision = ((vision - normalizers["vision_mean"]) / normalizers["vision_std"]).astype(np.float32)
    tactile = ((tactile - normalizers["tactile_mean"]) / normalizers["tactile_std"]).astype(np.float32)
    batch = policy.make_batch(
        vision_features=torch.from_numpy(vision).to(device),
        tactile_features=torch.from_numpy(tactile).to(device),
        probe_flags=torch.zeros(1, device=device),
    )
    with torch.no_grad():
        outputs = policy.forward_batch(batch)
    action_delta = outputs.actions.detach().cpu().numpy()[0].astype(np.float32)
    if outputs.phase_logits is None:
        phase = "fused_policy"
    else:
        phase_idx = int(outputs.phase_logits.argmax(dim=-1).detach().cpu().numpy().reshape(-1)[0])
        phase = phase_names[phase_idx] if phase_idx < len(phase_names) else str(phase_idx)
    return action_delta, phase, float(uncertainty.get("uncertainty", 0.0))


def evaluate_episode(args: argparse.Namespace, episode_index: int, output_dir: Path) -> dict:
    seed = args.seed + episode_index
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    policy, phase_names, normalizers = load_policy(Path(args.checkpoint), device)
    blur_config = build_pseudo_blur_config(
        enabled=args.scene == "pseudo_blur",
        seed=seed,
        profile=args.pseudo_blur_profile,
        severity=args.pseudo_blur_severity,
    )
    object_profile = args.object_profile if args.scene == "material_object" else "default"
    agent = ManiSkillAgent(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=None,
        render_backend="none",
        object_profile=object_profile,
        pseudo_blur=blur_config,
    )
    obs = agent.reset(seed=seed)
    rewards = []
    success = 0.0
    max_is_grasped = 0.0
    phase_counts: dict[str, int] = {}
    trace_rows = []

    for step in range(args.max_steps):
        task_state = agent.get_task_state()
        action_delta, phase, uncertainty = predict_action(
            policy,
            phase_names,
            normalizers,
            obs,
            task_state,
            agent.last_info,
            agent,
            device,
            int(policy.vision_dim),
        )
        if args.control_mode == "pd_joint_pos":
            action = _qpos_delta_to_joint_pos(action_delta, task_state, agent.env.action_space, args.joint_position_scale)
        else:
            action = np.clip(action_delta, agent.env.action_space.low, agent.env.action_space.high)
        obs, reward, done, info = agent.step(action)
        is_grasped = _scalar(info.get("is_grasped", 0.0))
        max_is_grasped = max(max_is_grasped, is_grasped)
        success = max(success, _success_from_info(info))
        rewards.append(float(reward))
        phase_counts[phase] = phase_counts.get(phase, 0) + 1
        trace_rows.append(
            {
                "step": step,
                "phase": phase,
                "uncertainty": uncertainty,
                "reward": float(reward),
                "success": success,
                "is_grasped": is_grasped,
                **{f"action_delta_{idx}": float(value) for idx, value in enumerate(action_delta)},
            }
        )
        if done:
            break

    agent.close()
    trace_path = output_dir / f"episode_{episode_index:04d}_d_eval_trace.csv"
    write_trace(trace_rows, trace_path)
    return {
        "episode": episode_index,
        "seed": seed,
        "checkpoint": args.checkpoint,
        "env_id": args.env_id,
        "obs_mode": args.obs_mode,
        "control_mode": args.control_mode,
        "scene": args.scene,
        "object_profile": object_profile,
        "pseudo_blur_profile": blur_config.profile,
        "pseudo_blur_severity": blur_config.severity,
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "success": success,
        "success_rate": success,
        "max_is_grasped": max_is_grasped,
        "phase_counts": phase_counts,
        "trace_path": str(trace_path),
    }


def write_trace(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained D ActiveTactilePolicy checkpoint.")
    parser.add_argument("--checkpoint", default="runs/d_policy_assist_teacher_50_gpu/d_policy.pt")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1200)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=["state", "rgbd"], default="rgbd")
    parser.add_argument("--control-mode", choices=["pd_joint_pos", "pd_joint_delta_pos"], default="pd_joint_pos")
    parser.add_argument("--joint-position-scale", type=float, default=0.36)
    parser.add_argument("--scene", choices=["clean", "pseudo_blur", "material_object"], default="pseudo_blur")
    parser.add_argument("--object-profile", choices=["default", "transparent", "dark", "reflective", "low_texture"], default="default")
    parser.add_argument("--pseudo-blur-profile", choices=["mild", "transparent", "dark", "reflective", "low_texture"], default="mild")
    parser.add_argument("--pseudo-blur-severity", type=float, default=1.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-dir", default="results/d_policy_assist_teacher_50_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_episode(args, episode_index, output_dir) for episode_index in range(args.num_episodes)]
    json_path = output_dir / "d_eval_results.json"
    csv_path = output_dir / "d_eval_results.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "episode",
            "seed",
            "checkpoint",
            "env_id",
            "obs_mode",
            "control_mode",
            "scene",
            "object_profile",
            "pseudo_blur_profile",
            "pseudo_blur_severity",
            "steps",
            "total_reward",
            "success",
            "success_rate",
            "max_is_grasped",
            "trace_path",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    success_count = sum(float(row["success"]) > 0.0 for row in rows)
    grasp_count = sum(float(row["max_is_grasped"]) > 0.0 for row in rows)
    print(json.dumps({"episodes": len(rows), "success": success_count, "success_rate": success_count / max(len(rows), 1), "grasp": grasp_count}, indent=2))
    print(f"Saved D eval results: {json_path}")
    print(f"Saved D eval results: {csv_path}")


if __name__ == "__main__":
    main()