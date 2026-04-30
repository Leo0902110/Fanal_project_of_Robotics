from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from utils.d_features import DFeatureBundle


@dataclass
class ActiveProbeConfig:
    probe_distance_threshold: float = 0.08
    settle_velocity_threshold: float = 0.2
    max_probe_steps: int = 2
    probe_arm_delta: float = 0.15
    probe_gripper_delta: float = 0.0


def compute_probe_score(bundle: DFeatureBundle, config: ActiveProbeConfig) -> float:
    distance_gap = max(
        0.0, float(bundle.named["tcp_to_obj_dist"]) - config.probe_distance_threshold
    )
    moving_penalty = max(
        0.0, float(bundle.named["qvel_norm"]) - config.settle_velocity_threshold
    )
    grasp_bonus = 1.0 - float(bundle.named["is_grasped"])
    return distance_gap + 0.25 * moving_penalty + 0.1 * grasp_bonus


def should_probe(
    bundle: DFeatureBundle,
    probe_steps_used: int,
    config: ActiveProbeConfig,
) -> bool:
    if probe_steps_used >= config.max_probe_steps:
        return False
    if float(bundle.named["success"]) > 0.0:
        return False
    if float(bundle.named["is_grasped"]) > 0.5:
        return False
    return compute_probe_score(bundle, config) > 0.0


def build_joint_probe_action(
    action_dim: int,
    step_idx: int,
    config: ActiveProbeConfig,
) -> np.ndarray:
    action = np.zeros(action_dim, dtype=np.float32)
    if action_dim == 0:
        return action

    direction = 1.0 if step_idx % 2 == 0 else -1.0
    if action_dim >= 1:
        action[0] = direction * config.probe_arm_delta
    if action_dim >= 2:
        action[1] = -0.5 * direction * config.probe_arm_delta
    action[-1] = config.probe_gripper_delta
    return action


def select_action(
    base_action: np.ndarray,
    bundle: DFeatureBundle,
    probe_steps_used: int,
    step_idx: int,
    config: ActiveProbeConfig,
) -> tuple[np.ndarray, bool, float]:
    score = compute_probe_score(bundle, config)
    trigger_probe = should_probe(bundle, probe_steps_used, config)
    if not trigger_probe:
        return base_action, False, score
    return build_joint_probe_action(base_action.shape[0], step_idx, config), True, score