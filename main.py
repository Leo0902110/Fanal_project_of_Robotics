from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.active_perception import ActivePerceptionConfig, ActivePerceptionCoordinator
from src.models import ActivePerceptionPolicy, JointScriptedPickCubePolicy, ScriptedPickCubePolicy, SineProbePolicy
from src.perception import build_pseudo_blur_config
from src.tactile import ContactFeatureExtractor, TactileBoundaryRefiner


class DActiveTactilePolicyAdapter:
    """Adapter that exposes D's ActiveTactilePolicy through the main predict API."""

    def __init__(self, action_space, use_active_probe: bool, vision_dim: int = 32):
        self.action_space = action_space
        self.use_active_probe = use_active_probe
        self.vision_dim = vision_dim
        self.phase = "fallback"
        self.probe_count = 0
        self.policy = None

    def _ensure_policy(self, tactile_dim: int):
        if self.policy is not None:
            return
        from models.d_policy import ActiveTactilePolicy

        self.policy = ActiveTactilePolicy(
            action_dim=self.action_space.shape[0],
            vision_dim=self.vision_dim,
            tactile_dim=tactile_dim,
        )
        self.policy.eval()

    def predict(self, obs, step: int = 0, context: dict | None = None):
        context = context or {}
        env = context.get("env")
        info = context.get("info", {})
        visual_uncertainty = float(context.get("mean_uncertainty", 0.0))
        if env is None:
            return np.zeros(self.action_space.shape, dtype=np.float32)

        from utils.d_features import extract_contact_reading, extract_d_features
        from utils.d_interface import DInferenceRequest

        tactile_reading = extract_contact_reading(env)
        bundle = extract_d_features(obs, info, tactile_reading)
        self._ensure_policy(bundle.feature_vector.shape[0])
        request = DInferenceRequest(
            feature_bundle=bundle,
            visual_uncertainty=visual_uncertainty if self.use_active_probe else 0.0,
            probe_steps_used=self.probe_count,
            step_idx=step,
            metadata={"source": "main"},
        )
        output = self.policy.act_from_request(request)
        self.phase = output.policy_mode
        if output.probe_triggered:
            self.probe_count += 1
        return np.clip(output.action.astype(np.float32), self.action_space.low, self.action_space.high)


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
    pseudo_blur_profile: str,
    pseudo_blur_severity: float,
    object_profile: str,
) -> dict:
    from src.env.wrapper import ManiSkillAgent

    blur_config = build_pseudo_blur_config(
        enabled=pseudo_blur,
        seed=seed,
        profile=pseudo_blur_profile,
        severity=pseudo_blur_severity,
    )
    render_mode = "rgb_array" if save_video else None
    render_backend = None if save_video else "none"
    if policy_name == "scripted":
        control_mode = "pd_ee_delta_pose"
    elif policy_name in {"d", "joint_scripted"}:
        control_mode = "pd_joint_delta_pos"
    else:
        control_mode = None
    agent = ManiSkillAgent(
        env_id=env_id,
        obs_mode=obs_mode,
        control_mode=control_mode,
        render_mode=render_mode,
        render_backend=render_backend,
        object_profile=object_profile,
        pseudo_blur=blur_config,
    )

    obs = agent.reset(seed=seed)
    tactile = ContactFeatureExtractor()
    boundary_refiner = TactileBoundaryRefiner()
    active_perception = ActivePerceptionCoordinator(
        ActivePerceptionConfig(enabled=active_probe, uncertainty_threshold=0.5, probe_budget=2)
    )
    effective_policy_name = policy_name
    scripted_ready = policy_name == "scripted" and agent.obs_mode != "state"
    if policy_name == "scripted" and not scripted_ready:
        effective_policy_name = "sine_fallback"

    if effective_policy_name == "scripted":
        policy = ScriptedPickCubePolicy(agent.env.action_space, use_active_probe=active_probe, probe_steps=2)
    elif effective_policy_name == "joint_scripted":
        policy = JointScriptedPickCubePolicy(agent.env.action_space, use_active_probe=active_probe, probe_steps=2)
    elif effective_policy_name == "d":
        policy = DActiveTactilePolicyAdapter(agent.env.action_space, use_active_probe=active_probe)
    else:
        base_policy = SineProbePolicy(agent.env.action_space)
        policy = ActivePerceptionPolicy(base_policy, agent.env.action_space) if active_probe else base_policy

    rewards = []
    uncertainty_values = []
    visual_trigger_count = 0
    tactile_contact_count = 0
    success = 0.0
    decision_trace = []

    for step in range(max_steps):
        uncertainty = agent.get_visual_uncertainty(obs)
        tactile_feature = tactile.extract(obs, agent.last_info, env=agent.env)
        policy_phase = str(getattr(policy, "phase", "fallback"))
        decision = active_perception.decide(
            step=step,
            phase=policy_phase,
            uncertainty=uncertainty,
            tactile_feature=tactile_feature,
        )
        refinement = boundary_refiner.update(
            step=step,
            decision=decision.to_dict(),
            tactile_feature=tactile_feature,
            visual_uncertainty=float(uncertainty["uncertainty"]),
        )
        context = {
            "active_probe": bool(decision.should_probe),
            "uncertainty": uncertainty,
            "mean_uncertainty": float(uncertainty["uncertainty"]),
            "tactile": tactile_feature,
            "active_perception": decision.to_dict(),
            "boundary_refinement": refinement.to_dict(),
            "post_probe_uncertainty": refinement.post_probe_uncertainty,
            "oracle": agent.get_task_state(),
            "env": agent.env,
            "info": agent.last_info,
        }
        action = policy.predict(obs, step=step, context=context)
        obs, reward, done, info = agent.step(action)

        rewards.append(reward)
        uncertainty_values.append(float(uncertainty["uncertainty"]))
        visual_trigger_count += int(bool(uncertainty["triggered"]))
        tactile_contact_count += int(tactile_feature["contact_detected"] > 0.0)
        success = max(success, _success_from_info(info))
        decision_trace.append(
            {
                **decision.to_dict(),
                "refined": refinement.refined,
                "boundary_confidence": refinement.boundary_confidence,
                "confidence_delta": refinement.confidence_delta,
                "post_probe_uncertainty": refinement.post_probe_uncertainty,
                "refinement_reason": refinement.reason,
                "reward": reward,
                "visual_triggered": bool(uncertainty["triggered"]),
                "contact_detected": tactile_feature["contact_detected"],
                "contact_strength": tactile_feature["contact_strength"],
            }
        )
        if done:
            break

    video_path = ""
    if save_video:
        candidate = output_dir / f"{name}.mp4"
        saved = agent.save_video(str(candidate))
        video_path = saved or ""
    agent.close()
    trace_path = output_dir / f"{name}_decision_trace.csv"
    write_decision_trace(decision_trace, trace_path)

    trigger_count = int(getattr(policy, "probe_count", 0))
    decision_summary = active_perception.summary()
    refinement_summary = boundary_refiner.summary()
    return {
        "experiment": name,
        "condition": condition,
        "env_id": env_id,
        "obs_mode": agent.obs_mode,
        "object_profile": object_profile,
        "pseudo_blur": pseudo_blur,
        "pseudo_blur_profile": blur_config.profile,
        "pseudo_blur_severity": blur_config.severity,
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
        **decision_summary,
        **refinement_summary,
        "tactile_contact_count": tactile_contact_count,
        "video_path": video_path,
        "decision_trace_path": str(trace_path),
        "fallback_used": bool(agent.obs_mode != obs_mode or effective_policy_name != policy_name),
        "blur_config": asdict(blur_config),
    }


def write_decision_trace(rows: list[dict], trace_path: Path) -> None:
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
        "refined",
        "boundary_confidence",
        "confidence_delta",
        "post_probe_uncertainty",
        "refinement_reason",
        "reward",
        "visual_triggered",
        "contact_detected",
        "contact_strength",
    ]
    with trace_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


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
        "object_profile",
        "pseudo_blur",
        "pseudo_blur_profile",
        "pseudo_blur_severity",
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
        "decision_ambiguity_count",
        "probe_request_count",
        "contact_resolved_count",
        "refinement_count",
        "final_boundary_confidence",
        "tactile_contact_count",
        "video_path",
        "decision_trace_path",
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
    if scene == "material_object":
        return [
            {
                "name": "material_object",
                "condition": "Material Object",
                "pseudo_blur": False,
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
    parser.add_argument("--scene", choices=["all", "clean", "pseudo_blur", "material_object"], default="all")
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", choices=["state", "rgbd"], default="rgbd")
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/mvp")
    parser.add_argument("--policy", choices=["sine", "scripted", "joint_scripted", "d"], default="scripted")
    parser.add_argument("--pseudo-blur-profile", choices=["mild", "transparent", "dark", "reflective", "low_texture"], default="mild")
    parser.add_argument("--pseudo-blur-severity", type=float, default=1.0)
    parser.add_argument("--object-profile", choices=["default", "transparent", "dark", "reflective", "low_texture"], default="default")
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
            pseudo_blur_profile=args.pseudo_blur_profile,
            pseudo_blur_severity=args.pseudo_blur_severity,
            object_profile=args.object_profile if args.scene == "material_object" else "default",
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False, indent=2))

    write_results(rows, output_dir)


if __name__ == "__main__":
    main()
