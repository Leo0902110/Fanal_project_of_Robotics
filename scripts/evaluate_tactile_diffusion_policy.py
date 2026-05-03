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

from models.tactile_diffusion_policy import DiffusionScheduler, TactileDiffusionPolicy
from scripts.collect_d_assist_demos import build_d_tactile_features, build_vision_features, _safe_contact_reading, _vector
from scripts.evaluate_bc import _scalar, _success_from_info
from scripts.evaluate_d_policy import load_policy as load_d_policy
from scripts.material_stress import (
    MATERIAL_STRESS_PROFILE_CHOICES,
    OBJECT_PROFILE_CHOICES,
    PSEUDO_BLUR_PROFILE_CHOICES,
    build_scene_blur_config,
)
from src.env.wrapper import ManiSkillAgent


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


def load_policy(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = TactileDiffusionPolicy(
        vision_dim=int(checkpoint["vision_dim"]),
        tactile_dim=int(checkpoint["tactile_dim"]),
        action_dim=int(checkpoint["action_dim"]),
        action_horizon=int(checkpoint["action_horizon"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_phase_classes=int(checkpoint.get("num_phase_classes", 6)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    scheduler = DiffusionScheduler(num_train_steps=int(checkpoint["diffusion_steps"])).to(device)
    normalizers = {
        "vision_mean": _to_numpy(checkpoint["vision_mean"]),
        "vision_std": _to_numpy(checkpoint["vision_std"]),
        "tactile_mean": _to_numpy(checkpoint["tactile_mean"]),
        "tactile_std": _to_numpy(checkpoint["tactile_std"]),
        "action_mean": _to_numpy(checkpoint["action_mean"]),
        "action_std": _to_numpy(checkpoint["action_std"]),
        "action_constant_mask": _to_numpy(checkpoint.get("action_constant_mask", np.zeros(int(checkpoint["action_dim"]), dtype=np.float32))),
    }
    phase_names = tuple(checkpoint.get("phase_names", ("bc", "align_xy", "descend", "close", "transfer", "settle")))
    residual_mode = bool(checkpoint.get("residual_mode", False))
    base_d_checkpoint = str(checkpoint.get("base_d_checkpoint", "") or "")
    return model, scheduler, normalizers, phase_names, residual_mode, base_d_checkpoint


@torch.no_grad()
def predict_base_d_action(
    base_policy,
    base_normalizers,
    vision: np.ndarray,
    tactile: np.ndarray,
    device: torch.device,
    condition_clip_sigma: float = 0.0,
) -> np.ndarray:
    vision_norm = _normalize_condition(vision, base_normalizers["vision_mean"], base_normalizers["vision_std"], condition_clip_sigma)
    tactile_norm = _normalize_condition(tactile, base_normalizers["tactile_mean"], base_normalizers["tactile_std"], condition_clip_sigma)
    batch = base_policy.make_batch(
        vision_features=torch.from_numpy(vision_norm).to(device).unsqueeze(0),
        tactile_features=torch.from_numpy(tactile_norm).to(device).unsqueeze(0),
        probe_flags=torch.zeros(1, device=device),
    )
    return base_policy.forward_batch(batch).actions.detach().cpu().numpy()[0].astype(np.float32)


def _normalize_condition(values: np.ndarray, mean: np.ndarray, std: np.ndarray, clip_sigma: float = 0.0) -> np.ndarray:
    normalized = ((values - mean) / std).astype(np.float32)
    if clip_sigma and clip_sigma > 0.0:
        normalized = np.clip(normalized, -float(clip_sigma), float(clip_sigma)).astype(np.float32)
    return normalized


def _qpos_delta_to_joint_pos(action_delta: np.ndarray, task_state: dict, action_space, scale: float) -> np.ndarray:
    qpos = _vector(task_state.get("qpos"), 7)
    if qpos is None:
        return np.clip(action_delta, action_space.low, action_space.high)
    action = np.zeros(action_space.shape, dtype=np.float32)
    action[:7] = qpos[:7] + scale * action_delta[:7]
    action[-1] = action_delta[-1]
    return np.clip(action, action_space.low, action_space.high)


@torch.no_grad()
def sample_action_chunk(
    model,
    scheduler,
    normalizers,
    phase_names,
    obs,
    task_state,
    info,
    agent,
    device,
    sample_steps: int,
    zero_action_dims: tuple[int, ...],
    init_noise_scale: float,
    base_policy=None,
    base_normalizers=None,
    residual_scale: float = 1.0,
    condition_clip_sigma: float = 0.0,
) -> tuple[np.ndarray, float, str]:
    uncertainty = agent.get_visual_uncertainty(obs)
    contact = _safe_contact_reading(agent.env)
    vision = build_vision_features(uncertainty, int(model.vision_dim))
    tactile = build_d_tactile_features(task_state, info, contact)
    raw_vision = vision.astype(np.float32, copy=True)
    raw_tactile = tactile.astype(np.float32, copy=True)
    vision = _normalize_condition(vision, normalizers["vision_mean"], normalizers["vision_std"], condition_clip_sigma)
    tactile = _normalize_condition(tactile, normalizers["tactile_mean"], normalizers["tactile_std"], condition_clip_sigma)
    vision_tensor = torch.from_numpy(vision).to(device).unsqueeze(0)
    tactile_tensor = torch.from_numpy(tactile).to(device).unsqueeze(0)
    phase_idx = int(model.predict_phase_logits(vision_tensor, tactile_tensor).argmax(dim=-1).detach().cpu().numpy()[0])
    phase_tensor = torch.tensor([phase_idx], device=device, dtype=torch.long)
    action_chunk = scheduler.sample(model, vision_tensor, tactile_tensor, sample_steps=sample_steps, phase_labels=phase_tensor, init_noise_scale=init_noise_scale)
    actions = action_chunk.detach().cpu().numpy()[0]
    actions = actions * normalizers["action_std"].reshape(1, -1) + normalizers["action_mean"].reshape(1, -1)
    if base_policy is not None and base_normalizers is not None:
        base_action = predict_base_d_action(
            base_policy,
            base_normalizers,
            raw_vision,
            raw_tactile,
            device,
            condition_clip_sigma=condition_clip_sigma,
        )
        actions = base_action.reshape(1, -1) + residual_scale * actions
    constant_mask = normalizers["action_constant_mask"].astype(bool)
    for dim in zero_action_dims:
        if 0 <= dim < constant_mask.shape[0]:
            constant_mask[dim] = True
    actions[:, constant_mask] = normalizers["action_mean"][constant_mask].reshape(1, -1)
    phase = phase_names[phase_idx] if phase_idx < len(phase_names) else str(phase_idx)
    return np.clip(actions.astype(np.float32), -1.0, 1.0), float(uncertainty.get("uncertainty", 0.0)), phase


def evaluate_episode(args: argparse.Namespace, episode_index: int, output_dir: Path) -> dict:
    seed = args.seed + episode_index
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model, scheduler, normalizers, phase_names, residual_mode, checkpoint_base_d = load_policy(Path(args.checkpoint), device)
    base_policy = None
    base_normalizers = None
    base_d_checkpoint = str(args.base_d_checkpoint or checkpoint_base_d).strip()
    if base_d_checkpoint:
        base_policy, _base_phase_names, base_normalizers = load_d_policy(Path(base_d_checkpoint), device)
        base_policy.eval()
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
        control_mode=args.control_mode,
        render_mode="rgb_array" if args.save_video else None,
        render_backend=None if args.save_video else "none",
        object_profile=object_profile,
        pseudo_blur=blur_config,
    )
    obs = agent.reset(seed=seed)
    rewards = []
    success = 0.0
    max_is_grasped = 0.0
    trace_rows = []
    planned_actions: list[np.ndarray] = []
    planned_phase = "diffusion"
    planned_uncertainty = 0.0

    for step in range(args.max_steps):
        task_state = agent.get_task_state()
        if not planned_actions or step % args.replan_interval == 0:
            action_chunk, planned_uncertainty, planned_phase = sample_action_chunk(
                model,
                scheduler,
                normalizers,
                phase_names,
                obs,
                task_state,
                agent.last_info,
                agent,
                device,
                args.sample_steps,
                args.zero_action_dims,
                args.init_noise_scale,
                base_policy=base_policy,
                base_normalizers=base_normalizers,
                residual_scale=args.residual_scale,
                condition_clip_sigma=args.condition_clip_sigma,
            )
            planned_actions = [action_chunk[idx] for idx in range(min(args.replan_interval, action_chunk.shape[0]))]
        action_delta = planned_actions.pop(0)
        if args.control_mode == "pd_joint_pos":
            action = _qpos_delta_to_joint_pos(action_delta, task_state, agent.env.action_space, args.joint_position_scale)
        else:
            action = np.clip(action_delta, agent.env.action_space.low, agent.env.action_space.high)
        obs, reward, done, info = agent.step(action)
        is_grasped = _scalar(info.get("is_grasped", 0.0))
        max_is_grasped = max(max_is_grasped, is_grasped)
        success = max(success, _success_from_info(info))
        rewards.append(float(reward))
        trace_rows.append(
            {
                "step": step,
                "phase": planned_phase,
                "uncertainty": planned_uncertainty,
                "reward": float(reward),
                "success": success,
                "is_grasped": is_grasped,
                **{f"action_delta_{idx}": float(value) for idx, value in enumerate(action_delta)},
            }
        )
        if done:
            break

    trace_path = output_dir / f"episode_{episode_index:04d}_tdp_eval_trace.csv"
    write_trace(trace_rows, trace_path)
    video_path = ""
    if args.save_video:
        saved = agent.save_video(str(output_dir / f"episode_{episode_index:04d}_tdp_eval.mp4"), fps=args.video_fps)
        video_path = saved or ""
    agent.close()
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
        "material_visual_stress": bool(args.material_visual_stress),
        "sample_steps": args.sample_steps,
        "replan_interval": args.replan_interval,
        "residual_mode": residual_mode or bool(base_d_checkpoint),
        "base_d_checkpoint": base_d_checkpoint,
        "residual_scale": args.residual_scale,
        "condition_clip_sigma": args.condition_clip_sigma,
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "success": success,
        "success_rate": success,
        "max_is_grasped": max_is_grasped,
        "trace_path": str(trace_path),
        "video_path": video_path,
    }


def write_trace(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a tactile-conditioned diffusion policy checkpoint.")
    parser.add_argument("--checkpoint", default="runs/tactile_diffusion_policy_500/tactile_diffusion_policy.pt")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=4000)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=["state", "rgbd"], default="rgbd")
    parser.add_argument("--control-mode", choices=["pd_joint_pos", "pd_joint_delta_pos"], default="pd_joint_pos")
    parser.add_argument("--joint-position-scale", type=float, default=0.36)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--replan-interval", type=int, default=4)
    parser.add_argument("--zero-action-dims", type=int, nargs="*", default=[6])
    parser.add_argument("--init-noise-scale", type=float, default=1.0)
    parser.add_argument("--base-d-checkpoint", default="")
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--condition-clip-sigma", type=float, default=0.0)
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
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=25)
    parser.add_argument("--output-dir", default="results/tactile_diffusion_policy_500_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [evaluate_episode(args, episode_index, output_dir) for episode_index in range(args.num_episodes)]
    json_path = output_dir / "tdp_eval_results.json"
    csv_path = output_dir / "tdp_eval_results.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
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
        "material_visual_stress",
        "sample_steps",
        "replan_interval",
        "residual_mode",
        "base_d_checkpoint",
        "residual_scale",
        "condition_clip_sigma",
        "steps",
        "total_reward",
        "success",
        "success_rate",
        "max_is_grasped",
        "trace_path",
        "video_path",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    success_count = sum(float(row["success"]) > 0.0 for row in rows)
    grasp_count = sum(float(row["max_is_grasped"]) > 0.0 for row in rows)
    print(json.dumps({"episodes": len(rows), "success": success_count, "success_rate": success_count / max(len(rows), 1), "grasp": grasp_count}, indent=2))
    print(f"Saved tactile diffusion eval results: {json_path}")
    print(f"Saved tactile diffusion eval results: {csv_path}")


if __name__ == "__main__":
    main()