from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.active_perception import ActivePerceptionConfig, ActivePerceptionCoordinator, TactileProbePlanner
from src.data import filter_observation_for_camera, flatten_observation
from src.models import ActivePerceptionPolicy, JointScriptedPickCubePolicy, ScriptedPickCubePolicy, SineProbePolicy
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


def _task_geometry(task_state: dict) -> tuple[np.ndarray, np.ndarray]:
    tcp_pos = _vector(task_state.get("tcp_pos"))
    obj_pos = _vector(task_state.get("obj_pos"))
    goal_pos = _vector(task_state.get("goal_pos"))
    tcp_to_obj = obj_pos - tcp_pos if tcp_pos is not None and obj_pos is not None else np.zeros(3, dtype=np.float32)
    obj_to_goal = goal_pos - obj_pos if obj_pos is not None and goal_pos is not None else np.zeros(3, dtype=np.float32)
    return tcp_to_obj.astype(np.float32), obj_to_goal.astype(np.float32)


def collect_episode(args: argparse.Namespace, episode_index: int, output_dir: Path) -> dict:
    from src.env.wrapper import ManiSkillAgent

    seed = args.seed + episode_index
    pseudo_blur = args.scene == "pseudo_blur"
    active_probe = bool(args.use_active_probe)
    blur_config = build_pseudo_blur_config(
        enabled=pseudo_blur,
        seed=seed,
        profile=args.pseudo_blur_profile,
        severity=args.pseudo_blur_severity,
    )
    object_profile = args.object_profile if args.scene == "material_object" else "default"
    if args.policy == "scripted":
        control_mode = "pd_ee_delta_pose"
    elif args.policy == "joint_scripted":
        control_mode = "pd_joint_delta_pos"
    else:
        control_mode = None
    agent = ManiSkillAgent(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=control_mode,
        render_mode=None,
        render_backend="none",
        robot_uids=args.robot_uids or None,
        sensor_configs=(
            {
                args.training_camera: {
                    "width": args.sensor_width,
                    "height": args.sensor_height,
                }
            }
            if args.training_camera and args.sensor_width and args.sensor_height
            else None
        ),
        object_profile=object_profile,
        pseudo_blur=blur_config,
    )

    obs = agent.reset(seed=seed)
    tactile = ContactFeatureExtractor()
    boundary_refiner = TactileBoundaryRefiner()
    probe_planner = TactileProbePlanner()
    active_perception = ActivePerceptionCoordinator(
        ActivePerceptionConfig(enabled=active_probe, uncertainty_threshold=0.5, probe_budget=2)
    )

    effective_policy = args.policy
    if args.policy == "scripted" and agent.obs_mode == "state" and not agent.using_mock_env:
        effective_policy = "sine_fallback"

    if effective_policy == "scripted":
        policy = ScriptedPickCubePolicy(agent.env.action_space, use_active_probe=active_probe, probe_steps=2)
    elif effective_policy == "joint_scripted":
        policy = JointScriptedPickCubePolicy(agent.env.action_space, use_active_probe=active_probe, probe_steps=2)
    else:
        base_policy = SineProbePolicy(agent.env.action_space)
        policy = ActivePerceptionPolicy(base_policy, agent.env.action_space) if active_probe else base_policy

    observations = []
    actions = []
    rewards = []
    dones = []
    uncertainty_values = []
    boundary_confidence = []
    contact_detected_values = []
    contact_strength_values = []
    left_force_norm_values = []
    right_force_norm_values = []
    net_force_norm_values = []
    pairwise_contact_used_values = []
    post_probe_uncertainty_values = []
    tcp_to_obj_values = []
    obj_to_goal_values = []
    dominant_reasons = []
    probe_states = []
    probe_points = []
    refined_grasp_targets = []
    decision_trace = []
    success = 0.0

    for step in range(args.max_steps):
        uncertainty = agent.get_visual_uncertainty(obs)
        tactile_feature = tactile.extract(obs, agent.last_info, env=agent.env)
        phase = str(getattr(policy, "phase", "fallback"))
        oracle_state = agent.get_task_state()
        decision = active_perception.decide(
            step=step,
            phase=phase,
            uncertainty=uncertainty,
            tactile_feature=tactile_feature,
        )
        probe_plan = probe_planner.plan(
            decision=decision.to_dict(),
            uncertainty=uncertainty,
            oracle_state=oracle_state,
        )
        refinement = boundary_refiner.update(
            step=step,
            decision=decision.to_dict(),
            tactile_feature=tactile_feature,
            visual_uncertainty=float(uncertainty["uncertainty"]),
            probe_plan=probe_plan.to_dict(),
            oracle_state=oracle_state,
        )
        context = {
            "active_probe": bool(decision.should_probe),
            "uncertainty": uncertainty,
            "mean_uncertainty": float(uncertainty["uncertainty"]),
            "tactile": tactile_feature,
            "active_perception": decision.to_dict(),
            "probe_plan": probe_plan.to_dict(),
            "boundary_refinement": refinement.to_dict(),
            "post_probe_uncertainty": refinement.post_probe_uncertainty,
            "refined_grasp_target": refinement.refined_grasp_target,
            "oracle": oracle_state,
        }
        tcp_to_obj, obj_to_goal = _task_geometry(context["oracle"])
        action = policy.predict(obs, step=step, context=context)

        learner_obs = filter_observation_for_camera(
            obs,
            camera=args.training_camera,
            keep_state=not args.camera_only,
        )
        observations.append(flatten_observation(learner_obs))
        actions.append(np.asarray(action, dtype=np.float32).reshape(-1))
        uncertainty_values.append(float(uncertainty["uncertainty"]))
        boundary_confidence.append(float(refinement.boundary_confidence))
        contact_detected_values.append(float(tactile_feature.get("contact_detected", 0.0)))
        contact_strength_values.append(float(tactile_feature.get("contact_strength", 0.0)))
        left_force_norm_values.append(float(tactile_feature.get("left_force_norm", 0.0)))
        right_force_norm_values.append(float(tactile_feature.get("right_force_norm", 0.0)))
        net_force_norm_values.append(float(tactile_feature.get("net_force_norm", 0.0)))
        pairwise_contact_used_values.append(float(tactile_feature.get("pairwise_contact_used", 0.0)))
        post_probe_uncertainty_values.append(float(refinement.post_probe_uncertainty))
        tcp_to_obj_values.append(tcp_to_obj)
        obj_to_goal_values.append(obj_to_goal)
        dominant_reasons.append(str(uncertainty.get("dominant_reason", "none")))
        probe_states.append(float(1.0 if decision.should_probe else 0.0))
        probe_points.append(np.asarray(probe_plan.probe_point, dtype=np.float32))
        refined_grasp_targets.append(
            np.asarray(
                refinement.refined_grasp_target if refinement.refined_grasp_target is not None else np.zeros(3),
                dtype=np.float32,
            )
        )

        obs, reward, done, info = agent.step(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        success = max(success, _success_from_info(info))
        decision_trace.append(
            {
                **decision.to_dict(),
                "boundary_confidence": refinement.boundary_confidence,
                "post_probe_uncertainty": refinement.post_probe_uncertainty,
                "refinement_reason": refinement.reason,
                "probe_reason": probe_plan.reason,
                "dominant_reason": uncertainty.get("dominant_reason", ""),
                "reward": float(reward),
            }
        )
        if done:
            break

    agent.close()

    demo_path = output_dir / f"episode_{episode_index:04d}.npz"
    metadata = {
        "episode_index": episode_index,
        "seed": seed,
        "env_id": args.env_id,
        "requested_obs_mode": args.obs_mode,
        "obs_mode": agent.obs_mode,
        "scene": args.scene,
        "object_profile": object_profile,
        "pseudo_blur": pseudo_blur,
        "pseudo_blur_profile": blur_config.profile,
        "pseudo_blur_severity": blur_config.severity,
        "active_probe": active_probe,
        "robot_uids": args.robot_uids,
        "training_camera": args.training_camera,
        "camera_only": bool(args.camera_only),
        "sensor_width": args.sensor_width,
        "sensor_height": args.sensor_height,
        "learner_observation_dim": int(observations[0].shape[0]) if observations else 0,
        "requested_policy": args.policy,
        "effective_policy": effective_policy,
        "env_backend": agent.backend_name,
        "init_error": agent.init_error,
        "success": success,
        "steps": len(rewards),
        "fallback_used": bool(
            agent.using_mock_env or agent.obs_mode != args.obs_mode or effective_policy != args.policy
        ),
        "blur_config": asdict(blur_config),
        **active_perception.summary(),
        **boundary_refiner.summary(),
    }
    np.savez_compressed(
        demo_path,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
        uncertainty=np.asarray(uncertainty_values, dtype=np.float32),
        boundary_confidence=np.asarray(boundary_confidence, dtype=np.float32),
        contact_detected=np.asarray(contact_detected_values, dtype=np.float32),
        contact_strength=np.asarray(contact_strength_values, dtype=np.float32),
        left_force_norm=np.asarray(left_force_norm_values, dtype=np.float32),
        right_force_norm=np.asarray(right_force_norm_values, dtype=np.float32),
        net_force_norm=np.asarray(net_force_norm_values, dtype=np.float32),
        pairwise_contact_used=np.asarray(pairwise_contact_used_values, dtype=np.float32),
        post_probe_uncertainty=np.asarray(post_probe_uncertainty_values, dtype=np.float32),
        tcp_to_obj_pos=np.asarray(tcp_to_obj_values, dtype=np.float32),
        obj_to_goal_pos=np.asarray(obj_to_goal_values, dtype=np.float32),
        dominant_reason=np.asarray(dominant_reasons),
        probe_state=np.asarray(probe_states, dtype=np.float32),
        probe_point=np.asarray(probe_points, dtype=np.float32),
        refined_grasp_target=np.asarray(refined_grasp_targets, dtype=np.float32),
        decision_trace=np.asarray(decision_trace, dtype=object),
        metadata=np.asarray(metadata, dtype=object),
    )
    return {"path": str(demo_path), **metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect MVP demonstrations for BC/DP training.")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=["state", "rgbd"], default="rgbd")
    parser.add_argument("--robot-uids", default="", help="Use panda_wristcam to collect eye-in-hand demonstrations.")
    parser.add_argument(
        "--training-camera",
        default="",
        help="Filter stored learner observations to this camera, e.g. hand_camera. Empty keeps all cameras.",
    )
    parser.add_argument("--camera-only", action="store_true", help="Drop non-sensor proprioceptive state from stored demos.")
    parser.add_argument("--sensor-width", type=int, default=384)
    parser.add_argument("--sensor-height", type=int, default=384)
    parser.add_argument("--scene", choices=["clean", "pseudo_blur", "material_object"], default="pseudo_blur")
    parser.add_argument("--object-profile", choices=["default", "transparent", "dark", "reflective", "low_texture"], default="default")
    parser.add_argument("--pseudo-blur-profile", choices=["mild", "transparent", "dark", "reflective", "low_texture"], default="mild")
    parser.add_argument("--pseudo-blur-severity", type=float, default=1.0)
    parser.add_argument("--policy", choices=["sine", "scripted", "joint_scripted"], default="scripted")
    parser.add_argument("--use-active-probe", action="store_true")
    parser.add_argument("--output-dir", default="data/demos/pickcube_mvp")
    parser.add_argument(
        "--clear-output-dir",
        action="store_true",
        help="Remove existing episode_*.npz and manifest.json in output-dir before collecting new demos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_output_dir:
        for path in sorted(output_dir.glob("episode_*.npz")):
            path.unlink()
        manifest_path = output_dir / "manifest.json"
        if manifest_path.exists():
            manifest_path.unlink()
        print(f"Cleared existing demos under: {output_dir}")

    rows = []
    for episode_index in range(args.start_index, args.start_index + args.num_episodes):
        row = collect_episode(args, episode_index, output_dir)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
