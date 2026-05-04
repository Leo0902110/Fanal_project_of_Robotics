from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from src.active_perception import ActivePerceptionConfig, ActivePerceptionCoordinator, TactileProbePlanner
from src.models import ActivePerceptionPolicy, JointScriptedPickCubePolicy, ScriptedPickCubePolicy, SineProbePolicy
from src.perception import build_pseudo_blur_config
from src.tactile import ContactFeatureExtractor, TactileBoundaryRefiner


WRIST_KEYWORDS = ("hand", "wrist", "tcp", "ee", "eye", "gripper")


def _parse_step_set(value: str) -> set[int]:
    steps: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            steps.update(range(int(start), int(end) + 1))
        else:
            steps.add(int(part))
    return steps


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans.ttf", "Arial.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            pass
    return ImageFont.load_default()


def _to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return None


def _normalize_rgb(arr: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(arr)
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        return None
    arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.size and np.nanmax(arr) <= 1.5:
        arr = arr * 255.0
    return np.uint8(np.clip(arr, 0, 255))


def iter_rgb_arrays(node: Any, path: str = "") -> list[tuple[str, np.ndarray]]:
    found: list[tuple[str, np.ndarray]] = []
    if isinstance(node, dict):
        for key, value in node.items():
            child_path = f"{path}/{key}" if path else str(key)
            arr = _to_numpy(value)
            if arr is not None:
                rgb = _normalize_rgb(arr)
                key_lower = str(key).lower()
                if rgb is not None and ("rgb" in key_lower or "color" in key_lower or "image" in key_lower):
                    found.append((child_path, rgb))
            found.extend(iter_rgb_arrays(value, child_path))
    return found


def choose_camera(candidates: list[tuple[str, np.ndarray]], camera: str, require_wrist: bool) -> tuple[str, np.ndarray]:
    if not candidates:
        raise ValueError("No RGB camera arrays found in the ManiSkill observation.")

    camera = camera.strip()
    if camera and camera != "auto":
        matches = [(path, arr) for path, arr in candidates if camera.lower() in path.lower()]
        if not matches:
            available = "\n".join(f"- {path}: {arr.shape}" for path, arr in candidates)
            raise ValueError(f"Requested camera {camera!r} not found. Available RGB camera arrays:\n{available}")
        return matches[0]

    wrist_matches = [
        (path, arr)
        for path, arr in candidates
        if any(keyword in path.lower() for keyword in WRIST_KEYWORDS)
    ]
    if wrist_matches:
        return wrist_matches[0]
    if require_wrist:
        available = "\n".join(f"- {path}: {arr.shape}" for path, arr in candidates)
        raise ValueError(
            "No wrist/hand/tcp camera was found in this ManiSkill observation. "
            "This environment may expose only base/scene cameras unless a custom camera is added.\n"
            f"Available RGB camera arrays:\n{available}"
        )
    return candidates[0]


def _bool(value: Any) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def _safe_float(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def compose_sensor_frame(
    camera_frame: np.ndarray,
    camera_path: str,
    trace_row: dict[str, Any],
    *,
    require_wrist: bool,
) -> np.ndarray:
    sensor = Image.fromarray(camera_frame).resize((760, 570), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (1280, 720), (248, 248, 245))
    canvas.paste(sensor, (40, 70))
    draw = ImageDraw.Draw(canvas)
    title_font = _font(28)
    header_font = _font(22)
    body_font = _font(17)
    small_font = _font(14)

    label = "Actual ManiSkill wrist/hand camera" if any(k in camera_path.lower() for k in WRIST_KEYWORDS) else "Actual ManiSkill observation camera"
    if require_wrist and "Actual ManiSkill observation camera" in label:
        label = "Actual ManiSkill camera (not wrist-mounted)"
    draw.text((40, 24), "Sensor-View Active Perception Demo", fill=(20, 20, 20), font=title_font)
    draw.rectangle((40, 70, 800, 105), fill=(0, 0, 0))
    draw.text((56, 78), f"{label}: {camera_path}", fill=(245, 245, 245), font=small_font)

    # Draw a simple visual-ambiguity reticle over the true sensor frame.
    cx, cy = 420, 350
    draw.ellipse((cx - 72, cy - 72, cx + 72, cy + 72), outline=(255, 210, 70), width=3)
    for dx, dy in [(-105, 0), (105, 0), (0, -105), (0, 105)]:
        draw.line((cx + 0.45 * dx, cy + 0.45 * dy, cx + dx, cy + dy), fill=(255, 210, 70), width=2)

    panel_x = 845
    draw.text((panel_x, 76), "Algorithm State", fill=(20, 20, 20), font=header_font)
    should_probe = _bool(trace_row.get("should_probe", False))
    phase = str(trace_row.get("phase", ""))
    if should_probe:
        state = "REQUEST PROBE"
        color = (245, 133, 24)
        description = "Ambiguous visual boundary triggers tactile-style probing."
    elif phase in {"close_gripper", "grasp"}:
        state = "GRASP"
        color = (114, 183, 178)
        description = "Gripper closes after uncertainty-aware approach."
    elif phase in {"transfer", "lift"}:
        state = "TRANSFER"
        color = (178, 121, 162)
        description = "Object is lifted and moved toward the goal."
    elif phase == "release":
        state = "RELEASE"
        color = (89, 161, 79)
        description = "Object is released at the target."
    else:
        state = "OBSERVE"
        color = (217, 78, 65)
        description = "RGB-D sensor view is evaluated for boundary ambiguity."

    draw.text((panel_x, 122), state, fill=color, font=header_font)
    draw.text((panel_x, 158), description, fill=(40, 40, 40), font=body_font)

    metrics = [
        ("phase", phase),
        ("visual uncertainty", f"{_safe_float(trace_row, 'visual_uncertainty'):.3f}"),
        ("decision ambiguity", f"{_safe_float(trace_row, 'ambiguity_score'):.3f}"),
        ("should probe", str(should_probe)),
        ("boundary confidence", f"{_safe_float(trace_row, 'boundary_confidence'):.3f}"),
        ("post-probe uncertainty", f"{_safe_float(trace_row, 'post_probe_uncertainty'):.3f}"),
        ("contact detected", str(_bool(trace_row.get("contact_detected", False)))),
        ("reward", f"{_safe_float(trace_row, 'reward'):.3f}"),
    ]
    y = 220
    for label, value in metrics:
        draw.text((panel_x, y), label, fill=(90, 90, 90), font=body_font)
        draw.text((panel_x + 230, y), value, fill=(20, 20, 20), font=body_font)
        y += 36

    draw.rounded_rectangle((845, 550, 1238, 650), radius=8, fill=(255, 255, 255), outline=(220, 220, 220))
    draw.text((862, 568), f"Decision: {trace_row.get('reason', '')}", fill=(50, 50, 50), font=small_font)
    draw.text((862, 596), f"Tactile update: {trace_row.get('refinement_reason', '')}", fill=(50, 50, 50), font=small_font)
    draw.text((862, 624), "Source: obs_mode=rgbd sensor_data, not env.render()", fill=(90, 90, 90), font=small_font)
    return np.asarray(canvas)


def write_decision_trace(rows: list[dict[str, Any]], trace_path: Path) -> None:
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with trace_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_sensor_demo(args: argparse.Namespace) -> None:
    from src.env.wrapper import ManiSkillAgent

    blur_config = build_pseudo_blur_config(
        enabled=True,
        seed=args.seed,
        profile=args.pseudo_blur_profile,
        severity=args.pseudo_blur_severity,
    )
    agent = ManiSkillAgent(
        env_id=args.env_id,
        obs_mode="rgbd",
        control_mode="pd_ee_delta_pose",
        render_mode=None,
        render_backend="cpu",
        robot_uids=args.robot_uids,
        sensor_configs={
            "hand_camera": {
                "width": args.sensor_width,
                "height": args.sensor_height,
            }
        },
        pseudo_blur=blur_config,
    )
    obs = agent.reset(seed=args.seed)

    first_candidates = iter_rgb_arrays(obs)
    camera_path, _ = choose_camera(first_candidates, args.camera, args.require_wrist)
    print("Selected RGB camera:", camera_path)
    print("Available RGB camera arrays:")
    for path, arr in first_candidates:
        print(f"  - {path}: {arr.shape}")

    tactile = ContactFeatureExtractor()
    boundary_refiner = TactileBoundaryRefiner()
    probe_planner = TactileProbePlanner()
    active_perception = ActivePerceptionCoordinator(
        ActivePerceptionConfig(enabled=True, uncertainty_threshold=args.uncertainty_threshold, probe_budget=2)
    )
    policy = ScriptedPickCubePolicy(agent.env.action_space, use_active_probe=True, probe_steps=2)
    force_probe_steps = _parse_step_set(args.demo_force_probe_steps)

    frames: list[np.ndarray] = []
    trace_rows: list[dict[str, Any]] = []
    rewards: list[float] = []
    success = 0.0

    for step in range(args.max_steps):
        candidates = iter_rgb_arrays(obs)
        camera_path, camera_frame = choose_camera(candidates, args.camera, args.require_wrist)
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
        if step in force_probe_steps:
            uncertainty = dict(uncertainty)
            uncertainty.setdefault(
                "uncertain_points",
                [
                    {
                        "normalized_x": 1.0 if step % 2 == 0 else -1.0,
                        "normalized_y": 0.0,
                        "reason": "demo_uncertain_boundary",
                        "score": max(float(uncertainty.get("uncertainty", 0.0)), args.uncertainty_threshold),
                    }
                ],
            )
            uncertainty["uncertain_boundary_points"] = max(
                int(uncertainty.get("uncertain_boundary_points", 0) or 0),
                len(uncertainty["uncertain_points"]),
            )
            decision = replace(
                decision,
                ambiguity_score=max(float(decision.ambiguity_score), args.uncertainty_threshold),
                visual_uncertainty=max(float(decision.visual_uncertainty), args.uncertainty_threshold),
                tactile_confidence=0.0,
                state="request_probe",
                should_probe=True,
                probe_index=max(int(decision.probe_index), 1),
                reason="demo_forced_visual_ambiguity_without_contact",
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
        trace_row = {
            **decision.to_dict(),
            "refined": refinement.refined,
            "boundary_confidence": refinement.boundary_confidence,
            "confidence_delta": refinement.confidence_delta,
            "post_probe_uncertainty": refinement.post_probe_uncertainty,
            "refinement_reason": refinement.reason,
            "probe_reason": probe_plan.reason,
            "visual_triggered": bool(uncertainty["triggered"]),
            "contact_detected": tactile_feature["contact_detected"],
            "contact_strength": tactile_feature["contact_strength"],
            "reward": 0.0,
        }
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
            "env": agent.env,
            "info": agent.last_info,
        }
        action = policy.predict(obs, step=step, context=context)
        obs, reward, done, info = agent.step(action)
        if args.post_action_contact_update:
            tactile_after = tactile.extract(obs, info, env=agent.env)
            if float(tactile_after.get("contact_detected", 0.0)) > 0.0:
                contact_refinement = boundary_refiner.update(
                    step=step,
                    decision=decision.to_dict(),
                    tactile_feature=tactile_after,
                    visual_uncertainty=float(uncertainty["uncertainty"]),
                    probe_plan=probe_plan.to_dict(),
                    oracle_state=oracle_state,
                )
                trace_row.update(
                    {
                        "refined": contact_refinement.refined,
                        "boundary_confidence": contact_refinement.boundary_confidence,
                        "confidence_delta": contact_refinement.confidence_delta,
                        "post_probe_uncertainty": contact_refinement.post_probe_uncertainty,
                        "refinement_reason": contact_refinement.reason,
                        "contact_detected": tactile_after["contact_detected"],
                        "contact_strength": tactile_after["contact_strength"],
                        "left_force_norm": tactile_after.get("left_force_norm", 0.0),
                        "right_force_norm": tactile_after.get("right_force_norm", 0.0),
                        "net_force_norm": tactile_after.get("net_force_norm", 0.0),
                        "pairwise_contact_used": tactile_after.get("pairwise_contact_used", 0.0),
                    }
                )
        trace_row["reward"] = reward
        frames.append(compose_sensor_frame(camera_frame, camera_path, trace_row, require_wrist=args.require_wrist))
        trace_rows.append(trace_row)
        rewards.append(reward)
        try:
            success = max(success, float(np.asarray(info.get("success", 0.0), dtype=np.float32).mean()))
        except Exception:
            pass
        if done:
            break

    agent.close()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    video_path = args.output_dir / "sensor_view_active_perception_demo.mp4"
    imageio.mimsave(video_path, frames, fps=args.fps)
    trace_path = args.output_dir / "sensor_view_decision_trace.csv"
    write_decision_trace(trace_rows, trace_path)
    result = {
        "env_backend": agent.backend_name,
        "fallback_used": bool(agent.using_mock_env),
        "robot_uids": args.robot_uids,
        "camera_path": camera_path,
        "sensor_width": args.sensor_width,
        "sensor_height": args.sensor_height,
        "require_wrist": args.require_wrist,
        "success_rate": success,
        "total_reward": float(np.sum(rewards)) if rewards else 0.0,
        "probe_request_count": active_perception.summary()["probe_request_count"],
        "demo_forced_probe_count": len(force_probe_steps),
        "display_probe_request_count": max(active_perception.summary()["probe_request_count"], len(force_probe_steps)),
        "final_boundary_confidence": boundary_refiner.summary()["final_boundary_confidence"],
        "contact_step_count": int(
            sum(float(row.get("contact_detected", 0.0) or 0.0) > 0.0 for row in trace_rows)
        ),
        "max_contact_strength": float(
            max([float(row.get("contact_strength", 0.0) or 0.0) for row in trace_rows] or [0.0])
        ),
        "video_path": str(video_path),
        "decision_trace_path": str(trace_path),
        "blur_config": asdict(blur_config),
    }
    result_path = args.output_dir / "sensor_view_results.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render an active-perception demo from actual ManiSkill RGB-D observation camera frames."
    )
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument(
        "--robot-uids",
        default="panda_wristcam",
        help="Use panda_wristcam for a real gripper-mounted hand_camera when available.",
    )
    parser.add_argument("--camera", default="auto", help="Camera path substring, or auto.")
    parser.add_argument("--require-wrist", action="store_true", help="Fail unless a hand/wrist/tcp/ee/eye camera is present.")
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=9)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--sensor-width", type=int, default=384)
    parser.add_argument("--sensor-height", type=int, default=384)
    parser.add_argument("--pseudo-blur-profile", default="transparent", choices=["mild", "transparent", "dark", "reflective", "low_texture"])
    parser.add_argument("--pseudo-blur-severity", type=float, default=1.0)
    parser.add_argument("--uncertainty-threshold", type=float, default=0.5)
    parser.add_argument(
        "--demo-force-probe-steps",
        default="",
        help=(
            "Comma/range list such as '0,1' or '0-2'. For presentation videos, force "
            "these early steps to display should_probe=True and REQUEST PROBE without "
            "fabricating contact force."
        ),
    )
    parser.add_argument(
        "--no-post-action-contact-update",
        dest="post_action_contact_update",
        action="store_false",
        help="Disable the real post-action pairwise contact force update.",
    )
    parser.set_defaults(post_action_contact_update=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/sensor_view_active_demo"))
    return parser.parse_args()


def main() -> None:
    run_sensor_demo(parse_args())


if __name__ == "__main__":
    main()
