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
from src.data import camera_rgbd_tensor, tactile_vector
from src.models import ScriptedPickCubePolicy
from src.perception import build_pseudo_blur_config
from src.tactile import ContactFeatureExtractor, TactileBoundaryRefiner


def _success_from_info(info: dict) -> float:
    try:
        return float(np.asarray(info.get("success", 0.0), dtype=np.float32).mean())
    except Exception:
        return 0.0


def collect_episode(args: argparse.Namespace, episode_index: int, output_dir: Path) -> dict:
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
    policy = ScriptedPickCubePolicy(agent.env.action_space, use_active_probe=args.use_active_probe, probe_steps=2)
    tactile = ContactFeatureExtractor()
    boundary_refiner = TactileBoundaryRefiner()
    probe_planner = TactileProbePlanner()
    active_perception = ActivePerceptionCoordinator(
        ActivePerceptionConfig(enabled=args.use_active_probe, uncertainty_threshold=0.5, probe_budget=2)
    )

    images, tactile_rows, actions, rewards, dones, trace_rows = [], [], [], [], [], []
    success = 0.0
    for step in range(args.max_steps):
        uncertainty = agent.get_visual_uncertainty(obs)
        tactile_feature = tactile.extract(obs, agent.last_info, env=agent.env)
        oracle_state = agent.get_task_state()
        phase = str(getattr(policy, "phase", "fallback"))
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
        action = policy.predict(obs, step=step, context=context)

        images.append(camera_rgbd_tensor(obs, args.camera))
        tactile_rows.append(
            tactile_vector(
                visual_uncertainty=float(uncertainty["uncertainty"]),
                boundary_confidence=float(refinement.boundary_confidence),
                tactile_feature=tactile_feature,
                post_probe_uncertainty=float(refinement.post_probe_uncertainty),
                probe_state=float(1.0 if decision.should_probe else 0.0),
            )
        )
        actions.append(np.asarray(action, dtype=np.float32).reshape(-1))

        obs, reward, done, info = agent.step(action)
        rewards.append(float(reward))
        dones.append(bool(done))
        success = max(success, _success_from_info(info))
        trace_rows.append(
            {
                **decision.to_dict(),
                "boundary_confidence": refinement.boundary_confidence,
                "contact_detected": tactile_feature.get("contact_detected", 0.0),
                "contact_strength": tactile_feature.get("contact_strength", 0.0),
                "refinement_reason": refinement.reason,
                "reward": float(reward),
            }
        )
        if done:
            break
    agent.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    demo_path = output_dir / f"episode_{episode_index:04d}.npz"
    metadata = {
        "episode_index": episode_index,
        "seed": seed,
        "env_id": args.env_id,
        "obs_mode": agent.obs_mode,
        "robot_uids": args.robot_uids,
        "training_camera": args.camera,
        "sensor_width": args.sensor_width,
        "sensor_height": args.sensor_height,
        "input_type": "hand_camera_rgbd_plus_tactile",
        "uses_oracle_geometry_as_policy_input": False,
        "uses_grasp_assist": False,
        "expert_policy": "scripted",
        "env_backend": agent.backend_name,
        "fallback_used": bool(agent.using_mock_env or agent.obs_mode != "rgbd"),
        "init_error": agent.init_error,
        "success": success,
        "steps": len(rewards),
        "blur_config": asdict(blur_config),
        **active_perception.summary(),
        **boundary_refiner.summary(),
    }
    np.savez_compressed(
        demo_path,
        images=np.asarray(images, dtype=np.float16),
        tactile=np.asarray(tactile_rows, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        dones=np.asarray(dones, dtype=bool),
        decision_trace=np.asarray(trace_rows, dtype=object),
        metadata=np.asarray(metadata, dtype=object),
    )
    return {"path": str(demo_path), **metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect hand-camera RGB-D + tactile demos for Level 2 BC.")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--robot-uids", default="panda_wristcam")
    parser.add_argument("--camera", default="hand_camera")
    parser.add_argument("--sensor-width", type=int, default=128)
    parser.add_argument("--sensor-height", type=int, default=128)
    parser.add_argument("--scene", choices=["clean", "pseudo_blur"], default="pseudo_blur")
    parser.add_argument("--pseudo-blur-profile", choices=["mild", "transparent", "dark", "reflective", "low_texture"], default="dark")
    parser.add_argument("--pseudo-blur-severity", type=float, default=1.0)
    parser.add_argument("--use-active-probe", action="store_true")
    parser.add_argument("--output-dir", default="data/demos/pickcube_level2_vision_tactile")
    parser.add_argument("--clear-output-dir", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.clear_output_dir:
        for path in output_dir.glob("episode_*.npz"):
            path.unlink()
        manifest = output_dir / "manifest.json"
        if manifest.exists():
            manifest.unlink()
    rows = []
    for episode_index in range(args.start_index, args.start_index + args.num_episodes):
        row = collect_episode(args, episode_index, output_dir)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
