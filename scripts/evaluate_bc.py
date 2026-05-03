from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_bc import BCPolicy
from src.active_perception import ActivePerceptionConfig, ActivePerceptionCoordinator
from src.data import DEFAULT_BC_FEATURE_NAMES, LEGACY_BC_FEATURE_NAMES, build_policy_feature_vector
from src.perception import build_pseudo_blur_config
from src.tactile import ContactFeatureExtractor, TactileBoundaryRefiner


def _success_from_info(info: dict) -> float:
    if not info:
        return 0.0
    try:
        return float(np.asarray(info.get("success", 0.0), dtype=np.float32).mean())
    except Exception:
        return 0.0


def _vector(value, length: int = 3) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < length:
        return None
    return arr[:length]


def build_task_geometry_features(task_state: dict) -> dict[str, float]:
    tcp_pos = _vector(task_state.get("tcp_pos"))
    obj_pos = _vector(task_state.get("obj_pos"))
    goal_pos = _vector(task_state.get("goal_pos"))
    tcp_to_obj = obj_pos - tcp_pos if tcp_pos is not None and obj_pos is not None else np.zeros(3, dtype=np.float32)
    obj_to_goal = goal_pos - obj_pos if obj_pos is not None and goal_pos is not None else np.zeros(3, dtype=np.float32)
    return {
        "tcp_to_obj_x": float(tcp_to_obj[0]),
        "tcp_to_obj_y": float(tcp_to_obj[1]),
        "tcp_to_obj_z": float(tcp_to_obj[2]),
        "obj_to_goal_x": float(obj_to_goal[0]),
        "obj_to_goal_y": float(obj_to_goal[1]),
        "obj_to_goal_z": float(obj_to_goal[2]),
    }


def _scalar(value, default: float = 0.0) -> float:
    try:
        return float(np.asarray(value, dtype=np.float32).mean())
    except Exception:
        return default


class OracleGraspAssist:
    def __init__(
        self,
        action_space,
        *,
        xy_threshold: float = 0.02,
        z_threshold: float = 0.012,
        target_z: float = 0.01,
        lift_command: float = -0.7,
        arm_gain: float = 10.0,
        max_arm_command: float = 1.0,
        max_assist_steps: int = 55,
        transfer_gain: float = 2.0,
        settle_threshold: float = 0.028,
        joint_position_scale: float = 0.0,
    ):
        self.action_space = action_space
        self.xy_threshold = xy_threshold
        self.z_threshold = z_threshold
        self.target_z = target_z
        self.lift_command = lift_command
        self.arm_gain = arm_gain
        self.max_arm_command = max_arm_command
        self.max_assist_steps = max_assist_steps
        self.transfer_gain = transfer_gain
        self.settle_threshold = settle_threshold
        self.joint_position_scale = joint_position_scale
        self.phase = "bc"
        self.last_delta_action: np.ndarray | None = None

    def apply(self, bc_action: np.ndarray, task_state: dict, info: dict, step: int) -> np.ndarray:
        tcp_pos = _vector(task_state.get("tcp_pos"))
        obj_pos = _vector(task_state.get("obj_pos"))
        goal_pos = _vector(task_state.get("goal_pos"))
        if tcp_pos is None or obj_pos is None:
            self.phase = "bc"
            return bc_action

        is_grasped = _scalar(info.get("is_grasped", 0.0))
        tcp_to_obj = obj_pos - tcp_pos
        xy_error = float(np.linalg.norm(tcp_to_obj[:2]))
        tcp_minus_obj_z = float(tcp_pos[2] - obj_pos[2])

        obj_to_goal = goal_pos - obj_pos if goal_pos is not None else None
        if obj_to_goal is not None and float(np.linalg.norm(obj_to_goal)) < self.settle_threshold:
            self.phase = "settle"
            return self._finalize_action(self._base_delta_action(gripper=-1.0), task_state)

        if is_grasped > 0.5:
            self.phase = "transfer"
            return self._transfer_action(obj_to_goal, task_state)

        if step > self.max_assist_steps:
            self.phase = "bc"
            return bc_action

        if xy_error > self.xy_threshold:
            self.phase = "align_xy"
            target = np.array([0.0, 0.0, float(tcp_to_obj[2])], dtype=np.float32)
            return self._servo_to_tcp_offset(tcp_to_obj, target, gripper=1.0, task_state=task_state)

        if tcp_minus_obj_z > self.z_threshold:
            self.phase = "descend"
            target = np.array([0.0, 0.0, self.target_z], dtype=np.float32)
            return self._servo_to_tcp_offset(tcp_to_obj, target, gripper=1.0, task_state=task_state)

        self.phase = "close"
        target = np.array([0.0, 0.0, self.target_z], dtype=np.float32)
        return self._servo_to_tcp_offset(tcp_to_obj, target, gripper=-1.0, task_state=task_state)

    def _base_delta_action(self, gripper: float) -> np.ndarray:
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        action[-1] = gripper
        return action

    def _finalize_action(self, delta_action: np.ndarray, task_state: dict) -> np.ndarray:
        if self.joint_position_scale <= 0.0:
            self.last_delta_action = delta_action.astype(np.float32, copy=True)
            return np.clip(delta_action, self.action_space.low, self.action_space.high)

        qpos = _vector(task_state.get("qpos"), 7)
        if qpos is None or delta_action.size < 8:
            self.last_delta_action = delta_action.astype(np.float32, copy=True)
            return np.clip(delta_action, self.action_space.low, self.action_space.high)

        self.last_delta_action = delta_action.astype(np.float32, copy=True)
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        action[:7] = qpos[:7] + self.joint_position_scale * delta_action[:7]
        action[-1] = delta_action[-1]
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _servo_to_tcp_offset(
        self,
        tcp_to_obj: np.ndarray,
        target_offset: np.ndarray,
        gripper: float,
        task_state: dict,
    ) -> np.ndarray:
        action = self._base_delta_action(gripper=gripper)
        error = target_offset - tcp_to_obj
        if action.size >= 7:
            action[0] = -self.arm_gain * error[1]
            action[2] = -0.65 * self.arm_gain * error[1]
            action[3] = -0.95 * self.arm_gain * error[0]
            action[5] = -0.45 * self.arm_gain * error[0]
            action[1] = 0.85 * self.arm_gain * error[2]
            action[4] = 0.15 * self.arm_gain * error[1]
            action[:7] = np.clip(action[:7], -self.max_arm_command, self.max_arm_command)
        return self._finalize_action(action, task_state)

    def _transfer_action(self, obj_to_goal: np.ndarray | None, task_state: dict) -> np.ndarray:
        action = self._base_delta_action(gripper=-1.0)
        if obj_to_goal is None or action.size < 7:
            return self._finalize_action(action, task_state)

        error = obj_to_goal.astype(np.float32, copy=False)
        action[0] = self.transfer_gain * 4.0 * error[1]
        action[2] = self.transfer_gain * 2.2 * error[1]
        action[3] = self.transfer_gain * 5.0 * error[0]
        action[5] = self.transfer_gain * 2.4 * error[0]
        action[1] = self.transfer_gain * -3.2 * error[2]
        action[:7] = np.clip(action[:7], -self.max_arm_command, self.max_arm_command)
        return self._finalize_action(action, task_state)


class BCPolicyRuntime:
    def __init__(self, checkpoint_path: Path, action_space, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.input_dim = int(checkpoint["input_dim"])
        self.action_dim = int(checkpoint["action_dim"])
        self.feature_names = tuple(checkpoint.get("feature_names") or LEGACY_BC_FEATURE_NAMES)
        self.input_mean = checkpoint.get("input_mean")
        self.input_std = checkpoint.get("input_std")
        if self.input_mean is not None and self.input_std is not None:
            self.input_mean = self._to_numpy(self.input_mean)
            self.input_std = self._to_numpy(self.input_std)
        self.model = BCPolicy(self.input_dim, self.action_dim, int(checkpoint["hidden_dim"])).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.action_space = action_space
        self.device = device

    def predict(
        self,
        obs,
        *,
        uncertainty: float,
        boundary_confidence: float,
        tactile_feature: dict[str, float],
        post_probe_uncertainty: float,
        task_geometry: dict[str, float] | None = None,
    ) -> np.ndarray:
        feature_values = {
            "uncertainty": uncertainty,
            "boundary_confidence": boundary_confidence,
            "contact_detected": float(tactile_feature.get("contact_detected", 0.0)),
            "contact_strength": float(tactile_feature.get("contact_strength", 0.0)),
            "left_force_norm": float(tactile_feature.get("left_force_norm", 0.0)),
            "right_force_norm": float(tactile_feature.get("right_force_norm", 0.0)),
            "net_force_norm": float(tactile_feature.get("net_force_norm", 0.0)),
            "pairwise_contact_used": float(tactile_feature.get("pairwise_contact_used", 0.0)),
            "post_probe_uncertainty": post_probe_uncertainty,
        }
        if task_geometry:
            feature_values.update(task_geometry)
        features = build_policy_feature_vector(obs, feature_values, feature_names=self.feature_names)
        if features.shape[0] != self.input_dim:
            raise ValueError(f"BC input dim mismatch: expected {self.input_dim}, got {features.shape[0]}")
        if self.input_mean is not None and self.input_std is not None:
            features = ((features - self.input_mean) / self.input_std).astype(np.float32)
        with torch.no_grad():
            action = self.model(torch.from_numpy(features).to(self.device).unsqueeze(0)).cpu().numpy()[0]
        if action.shape[0] != self.action_dim:
            raise ValueError(f"BC action dim mismatch: expected {self.action_dim}, got {action.shape[0]}")
        return np.clip(action.astype(np.float32), self.action_space.low, self.action_space.high)

    def _to_numpy(self, value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)


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
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    policy = BCPolicyRuntime(Path(args.checkpoint), agent.env.action_space, device)
    grasp_assist = (
        OracleGraspAssist(
            agent.env.action_space,
            xy_threshold=args.assist_xy_threshold,
            z_threshold=args.assist_z_threshold,
            target_z=args.assist_target_z,
            lift_command=args.assist_lift_command,
            arm_gain=args.assist_arm_gain,
            max_arm_command=args.assist_max_arm_command,
            transfer_gain=args.assist_transfer_gain,
            settle_threshold=args.assist_settle_threshold,
            joint_position_scale=args.assist_joint_position_scale,
        )
        if args.grasp_assist
        else None
    )
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

    rewards = []
    success = 0.0
    uncertainty_values = []
    visual_trigger_count = 0
    tactile_contact_count = 0
    grasp_count = 0
    max_is_grasped = 0.0
    initial_obj_z = None
    max_obj_lift = 0.0
    trace_rows = []

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
        task_state = agent.get_task_state()
        obj_pos = _vector(task_state.get("obj_pos"))
        if initial_obj_z is None and obj_pos is not None:
            initial_obj_z = float(obj_pos[2])
        if initial_obj_z is not None and obj_pos is not None:
            max_obj_lift = max(max_obj_lift, float(obj_pos[2]) - initial_obj_z)

        action = policy.predict(
            obs,
            uncertainty=float(uncertainty["uncertainty"]),
            boundary_confidence=float(refinement.boundary_confidence),
            tactile_feature=tactile_feature,
            post_probe_uncertainty=float(refinement.post_probe_uncertainty),
            task_geometry=build_task_geometry_features(task_state),
        )
        assist_phase = "bc"
        if grasp_assist is not None:
            action = grasp_assist.apply(action, task_state, agent.last_info, step)
            assist_phase = grasp_assist.phase
        obs, reward, done, info = agent.step(action)

        rewards.append(float(reward))
        success = max(success, _success_from_info(info))
        uncertainty_values.append(float(uncertainty["uncertainty"]))
        visual_trigger_count += int(bool(uncertainty["triggered"]))
        tactile_contact_count += int(tactile_feature["contact_detected"] > 0.0)
        is_grasped = _scalar(info.get("is_grasped", 0.0))
        max_is_grasped = max(max_is_grasped, is_grasped)
        grasp_count += int(is_grasped > 0.5)
        trace_rows.append(
            {
                **decision.to_dict(),
                "boundary_confidence": refinement.boundary_confidence,
                "post_probe_uncertainty": refinement.post_probe_uncertainty,
                "refinement_reason": refinement.reason,
                "reward": float(reward),
                "visual_triggered": bool(uncertainty["triggered"]),
                "assist_phase": assist_phase,
                "is_grasped": is_grasped,
            }
        )
        if done:
            break

    agent.close()
    trace_path = output_dir / f"episode_{episode_index:04d}_bc_eval_trace.csv"
    write_trace(trace_rows, trace_path)
    return {
        "episode": episode_index,
        "seed": seed,
        "env_id": args.env_id,
        "obs_mode": agent.obs_mode,
        "control_mode": args.control_mode or "default",
        "scene": args.scene,
        "object_profile": object_profile,
        "pseudo_blur": pseudo_blur,
        "pseudo_blur_profile": blur_config.profile,
        "pseudo_blur_severity": blur_config.severity,
        "active_probe": bool(args.use_active_probe),
        "policy": "bc",
        "checkpoint": args.checkpoint,
        "steps": len(rewards),
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "success": success,
        "success_rate": success,
        "mean_uncertainty": float(np.mean(uncertainty_values)) if uncertainty_values else 0.0,
        "visual_trigger_count": visual_trigger_count,
        **active_perception.summary(),
        **boundary_refiner.summary(),
        "tactile_contact_count": tactile_contact_count,
        "grasp_count": grasp_count,
        "max_is_grasped": max_is_grasped,
        "max_obj_lift": max_obj_lift,
        "grasp_assist": bool(args.grasp_assist),
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
        "assist_phase",
        "is_grasped",
    ]
    with trace_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_summary(rows: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "bc_eval_results.json"
    csv_path = output_dir / "bc_eval_results.csv"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "episode",
        "seed",
        "env_id",
        "obs_mode",
        "control_mode",
        "scene",
        "object_profile",
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
        "grasp_count",
        "max_is_grasped",
        "max_obj_lift",
        "grasp_assist",
        "trace_path",
        "fallback_used",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"Saved BC eval results: {json_path}")
    print(f"Saved BC eval results: {csv_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained BC policy in the MVP environment.")
    parser.add_argument("--checkpoint", default="runs/bc_mvp/bc_policy.pt")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=["state", "rgbd"], default="rgbd")
    parser.add_argument(
        "--control-mode",
        choices=[
            "pd_joint_delta_pos",
            "pd_joint_target_delta_pos",
            "pd_joint_pos",
            "pd_ee_delta_pos",
            "pd_ee_delta_pose",
            "pd_ee_pose",
        ],
        default=None,
    )
    parser.add_argument("--scene", choices=["clean", "pseudo_blur", "material_object"], default="pseudo_blur")
    parser.add_argument("--object-profile", choices=["default", "transparent", "dark", "reflective", "low_texture"], default="default")
    parser.add_argument("--pseudo-blur-profile", choices=["mild", "transparent", "dark", "reflective", "low_texture"], default="mild")
    parser.add_argument("--pseudo-blur-severity", type=float, default=1.0)
    parser.add_argument("--use-active-probe", action="store_true")
    parser.add_argument("--output-dir", default="results/bc_eval")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--grasp-assist", action="store_true")
    parser.add_argument("--assist-xy-threshold", type=float, default=0.02)
    parser.add_argument("--assist-z-threshold", type=float, default=0.012)
    parser.add_argument("--assist-target-z", type=float, default=0.01)
    parser.add_argument("--assist-lift-command", type=float, default=-0.7)
    parser.add_argument("--assist-arm-gain", type=float, default=10.0)
    parser.add_argument("--assist-max-arm-command", type=float, default=1.0)
    parser.add_argument("--assist-transfer-gain", type=float, default=2.0)
    parser.add_argument("--assist-settle-threshold", type=float, default=0.028)
    parser.add_argument("--assist-joint-position-scale", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grasp_assist and args.control_mode == "pd_joint_pos" and args.assist_joint_position_scale <= 0.0:
        raise ValueError("--control-mode pd_joint_pos with --grasp-assist requires --assist-joint-position-scale > 0")
    if args.assist_joint_position_scale > 0.0 and args.control_mode != "pd_joint_pos":
        raise ValueError("--assist-joint-position-scale is only valid with --control-mode pd_joint_pos")
    output_dir = Path(args.output_dir)
    rows = []
    for episode_index in range(args.num_episodes):
        row = evaluate_episode(args, episode_index, output_dir)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
    write_summary(rows, output_dir)


if __name__ == "__main__":
    main()
