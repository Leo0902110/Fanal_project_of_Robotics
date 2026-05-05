from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_vision_tactile_bc import VisionTactileBC
from src.active_perception import ActivePerceptionConfig, ActivePerceptionCoordinator, TactileProbePlanner
from src.data import camera_rgbd_tensor, tactile_vector
from src.perception import build_pseudo_blur_config
from src.tactile import ContactFeatureExtractor, TactileBoundaryRefiner


def _success_from_info(info: dict) -> float:
    try:
        return float(np.asarray(info.get("success", 0.0), dtype=np.float32).mean())
    except Exception:
        return 0.0


def _scalar(value, default: float = 0.0) -> float:
    try:
        return float(np.asarray(value, dtype=np.float32).mean())
    except Exception:
        return default


def _vector(value, length: int = 3) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < length:
        return None
    return arr[:length]


class VisionTactileRuntime:
    def __init__(self, checkpoint: Path, action_space, device: torch.device):
        ckpt = torch.load(checkpoint, map_location=device)
        self.model = VisionTactileBC(
            tactile_dim=int(ckpt["tactile_dim"]),
            action_dim=int(ckpt["action_dim"]),
            hidden_dim=int(ckpt["hidden_dim"]),
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.tactile_mean = self._to_numpy(ckpt["tactile_mean"])
        self.tactile_std = self._to_numpy(ckpt["tactile_std"])
        self.image_shape = tuple(int(x) for x in ckpt["image_shape"])
        self.action_space = action_space
        self.device = device

    def predict(self, image: np.ndarray, tactile: np.ndarray) -> np.ndarray:
        if tuple(image.shape) != self.image_shape:
            raise ValueError(f"Image shape mismatch: expected {self.image_shape}, got {image.shape}")
        tactile = ((tactile - self.tactile_mean) / self.tactile_std).astype(np.float32)
        with torch.no_grad():
            action = self.model(
                torch.from_numpy(image[None].astype(np.float32)).to(self.device),
                torch.from_numpy(tactile[None]).to(self.device),
            ).cpu().numpy()[0]
        return np.clip(action.astype(np.float32), self.action_space.low, self.action_space.high)

    def _to_numpy(self, value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)


def evaluate_episode(args: argparse.Namespace, episode_index: int, output_dir: Path) -> dict:
    from src.env.wrapper import ManiSkillAgent

    seed = args.seed + episode_index
    blur_config = build_pseudo_blur_config(
        enabled=args.scene == "pseudo_blur",
        seed=seed,
        profile=args.pseudo_blur_profile,
        severity=args.pseudo_blur_severity,
    )
    agent = ManiSkillAgent(
        env_id=args.env_id,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
        render_backend="none",
        robot_uids=args.robot_uids,
        sensor_configs={
            args.camera: {
                "width": args.sensor_width,
                "height": args.sensor_height,
            }
        },
        pseudo_blur=blur_config,
    )
    obs = agent.reset(seed=seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    policy = VisionTactileRuntime(Path(args.checkpoint), agent.env.action_space, device)
    tactile = ContactFeatureExtractor()
    boundary_refiner = TactileBoundaryRefiner()
    probe_planner = TactileProbePlanner()
    active_perception = ActivePerceptionCoordinator(
        ActivePerceptionConfig(enabled=args.use_active_probe, uncertainty_threshold=0.5, probe_budget=2)
    )

    rewards, trace_rows, frames = [], [], []
    success = 0.0
    visual_trigger_count = 0
    tactile_contact_count = 0
    grasp_count = 0
    max_is_grasped = 0.0
    initial_obj_z = None
    max_obj_lift = 0.0
    prev_action = None

    for step in range(args.max_steps):
        image = camera_rgbd_tensor(obs, args.camera)
        if args.save_video:
            rgb = np.moveaxis(image[:3], 0, -1)
            frames.append(np.uint8(np.clip(rgb * 255.0, 0, 255)))

        uncertainty = agent.get_visual_uncertainty(obs)
        tactile_feature = tactile.extract(obs, agent.last_info, env=agent.env)
        decision = active_perception.decide(
            step=step,
            phase="vision_tactile_bc",
            uncertainty=uncertainty,
            tactile_feature=tactile_feature,
        )
        probe_plan = probe_planner.plan(decision=decision.to_dict(), uncertainty=uncertainty, oracle_state={})
        refinement = boundary_refiner.update(
            step=step,
            decision=decision.to_dict(),
            tactile_feature=tactile_feature,
            visual_uncertainty=float(uncertainty["uncertainty"]),
            probe_plan=probe_plan.to_dict(),
            oracle_state={},
        )
        tactile_input = tactile_vector(
            visual_uncertainty=float(uncertainty["uncertainty"]),
            boundary_confidence=float(refinement.boundary_confidence),
            tactile_feature=tactile_feature,
            post_probe_uncertainty=float(refinement.post_probe_uncertainty),
            probe_state=float(1.0 if decision.should_probe else 0.0),
        )
        action = policy.predict(image, tactile_input)
        if prev_action is not None and args.action_smoothing > 0.0:
            alpha = float(np.clip(args.action_smoothing, 0.0, 0.95))
            action = (1.0 - alpha) * action + alpha * prev_action
        prev_action = action.astype(np.float32, copy=True)

        task_state = agent.get_task_state()
        obj_pos = _vector(task_state.get("obj_pos"))
        if initial_obj_z is None and obj_pos is not None:
            initial_obj_z = float(obj_pos[2])
        if initial_obj_z is not None and obj_pos is not None:
            max_obj_lift = max(max_obj_lift, float(obj_pos[2]) - initial_obj_z)

        obs, reward, done, info = agent.step(action)
        rewards.append(float(reward))
        success = max(success, _success_from_info(info))
        visual_trigger_count += int(bool(uncertainty["triggered"]))
        tactile_contact_count += int(float(tactile_feature.get("contact_detected", 0.0)) > 0.0)
        is_grasped = _scalar(info.get("is_grasped", 0.0))
        max_is_grasped = max(max_is_grasped, is_grasped)
        grasp_count += int(is_grasped > 0.5)
        trace_rows.append(
            {
                **decision.to_dict(),
                "boundary_confidence": refinement.boundary_confidence,
                "post_probe_uncertainty": refinement.post_probe_uncertainty,
                "refinement_reason": refinement.reason,
                "contact_detected": tactile_feature.get("contact_detected", 0.0),
                "contact_strength": tactile_feature.get("contact_strength", 0.0),
                "reward": float(reward),
                "is_grasped": is_grasped,
            }
        )
        if done:
            break

    agent.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"episode_{episode_index:04d}_vision_tactile_trace.csv"
    write_trace(trace_rows, trace_path)
    video_path = ""
    if args.save_video and frames:
        video_path = str(output_dir / f"episode_{episode_index:04d}_vision_tactile.mp4")
        imageio.mimsave(video_path, frames, fps=args.video_fps)
    return {
        "episode": episode_index,
        "seed": seed,
        "env_id": args.env_id,
        "obs_mode": agent.obs_mode,
        "control_mode": "pd_ee_delta_pose",
        "scene": args.scene,
        "pseudo_blur_profile": blur_config.profile,
        "pseudo_blur_severity": blur_config.severity,
        "robot_uids": args.robot_uids,
        "training_camera": args.camera,
        "sensor_width": args.sensor_width,
        "sensor_height": args.sensor_height,
        "policy": "vision_tactile_cnn_bc",
        "checkpoint": args.checkpoint,
        "input_type": "hand_camera_rgbd_plus_tactile",
        "uses_oracle_geometry": False,
        "uses_grasp_assist": False,
        "env_backend": agent.backend_name,
        "init_error": agent.init_error,
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "success": success,
        "success_rate": success,
        "visual_trigger_count": visual_trigger_count,
        **active_perception.summary(),
        **boundary_refiner.summary(),
        "tactile_contact_count": tactile_contact_count,
        "grasp_count": grasp_count,
        "max_is_grasped": max_is_grasped,
        "max_obj_lift": max_obj_lift,
        "video_path": video_path,
        "trace_path": str(trace_path),
        "fallback_used": bool(agent.using_mock_env or agent.obs_mode != "rgbd"),
        "blur_config": asdict(blur_config),
    }


def write_trace(rows: list[dict], trace_path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with trace_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "vision_tactile_eval_results.json"
    csv_path = output_dir / "vision_tactile_eval_results.csv"
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Saved Level 2 eval results: {json_path}")
    print(f"Saved Level 2 eval results: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Level 2 vision-tactile CNN BC without oracle geometry.")
    parser.add_argument("--checkpoint", default="runs/vision_tactile_level2_bc/vision_tactile_bc.pt")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1009)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--robot-uids", default="panda_wristcam")
    parser.add_argument("--camera", default="hand_camera")
    parser.add_argument("--sensor-width", type=int, default=128)
    parser.add_argument("--sensor-height", type=int, default=128)
    parser.add_argument("--scene", choices=["clean", "pseudo_blur"], default="pseudo_blur")
    parser.add_argument("--pseudo-blur-profile", choices=["mild", "transparent", "dark", "reflective", "low_texture"], default="dark")
    parser.add_argument("--pseudo-blur-severity", type=float, default=1.0)
    parser.add_argument("--use-active-probe", action="store_true")
    parser.add_argument("--action-smoothing", type=float, default=0.25)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=6)
    parser.add_argument("--output-dir", default="results/vision_tactile_level2_eval")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    output_dir = Path(args.output_dir)
    for episode_index in range(args.num_episodes):
        row = evaluate_episode(args, episode_index, output_dir)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
    write_summary(rows, output_dir)


if __name__ == "__main__":
    main()
