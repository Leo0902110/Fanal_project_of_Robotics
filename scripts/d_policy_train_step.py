import argparse
import pathlib
import sys
import warnings

import gymnasium as gym
import numpy as np
import torch

warnings.filterwarnings(
    "ignore",
    message="pinnochio package is not installed, robotics functionalities will not be available",
)

import mani_skill.envs  # noqa: F401

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.d_policy import ActiveTactilePolicy
from utils.d_active_probe import ActiveProbeConfig, build_joint_probe_action
from utils.d_features import extract_contact_reading, extract_d_features
from utils.d_interface import DTrainingRequest
from utils.d_uncertainty import VisualUncertaintyConfig, estimate_visual_uncertainty


def build_supervised_batch(
    env,
    batch_size: int,
    vision_dim: int,
    probe_config: ActiveProbeConfig,
    uncertainty_config: VisualUncertaintyConfig,
):
    obs, info = env.reset(seed=0)
    action_dim = env.action_space.shape[0]

    vision_features = []
    tactile_features = []
    probe_flags = []
    target_actions = []
    target_probe_flags = []
    target_uncertainty = []

    zero_action = np.zeros(action_dim, dtype=env.action_space.dtype)
    probe_steps_used = 0

    for step_idx in range(batch_size):
        tactile = extract_contact_reading(env)
        bundle = extract_d_features(obs, info, tactile)
        uncertainty = estimate_visual_uncertainty(bundle, uncertainty_config)
        trigger_probe = (
            uncertainty >= uncertainty_config.uncertainty_threshold
            and float(bundle.named["is_grasped"]) < 0.5
            and probe_steps_used < probe_config.max_probe_steps
        )
        target_action = (
            build_joint_probe_action(action_dim, step_idx, probe_config)
            if trigger_probe
            else zero_action.copy()
        )

        vision_features.append(np.zeros(vision_dim, dtype=np.float32))
        tactile_features.append(bundle.feature_vector.astype(np.float32, copy=False))
        probe_flags.append(float(trigger_probe))
        target_actions.append(target_action.astype(np.float32, copy=False))
        target_probe_flags.append(float(trigger_probe))
        target_uncertainty.append(float(uncertainty))

        if trigger_probe:
            probe_steps_used += 1
        obs, _, terminated, truncated, info = env.step(target_action)
        if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
            obs, info = env.reset(seed=step_idx + 1)
            probe_steps_used = 0

    return {
        "vision_features": torch.tensor(np.asarray(vision_features), dtype=torch.float32),
        "tactile_features": torch.tensor(np.asarray(tactile_features), dtype=torch.float32),
        "probe_flags": torch.tensor(np.asarray(probe_flags), dtype=torch.float32),
        "target_actions": torch.tensor(np.asarray(target_actions), dtype=torch.float32),
        "target_probe_flags": torch.tensor(np.asarray(target_probe_flags), dtype=torch.float32),
        "target_uncertainty": torch.tensor(np.asarray(target_uncertainty), dtype=torch.float32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run a minimal supervised training step for the D policy."
    )
    parser.add_argument("--task", default="PickCube-v1")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--vision-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args()

    env = gym.make(
        args.task,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
    )

    try:
        probe_config = ActiveProbeConfig()
        uncertainty_config = VisualUncertaintyConfig()
        batch_dict = build_supervised_batch(
            env,
            batch_size=args.batch_size,
            vision_dim=args.vision_dim,
            probe_config=probe_config,
            uncertainty_config=uncertainty_config,
        )
        policy = ActiveTactilePolicy(
            action_dim=env.action_space.shape[0],
            vision_dim=args.vision_dim,
            tactile_dim=batch_dict["tactile_features"].shape[-1],
            probe_config=probe_config,
            uncertainty_config=uncertainty_config,
        )
        optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
        batch = policy.make_batch_from_request(DTrainingRequest(**batch_dict))

        before = policy.compute_loss(batch)
        metrics = policy.train_step(batch, optimizer)
        after = policy.compute_loss(batch)

        print(
            {
                "batch_size": args.batch_size,
                "interface_spec": policy.interface_spec,
                "vision_dim": args.vision_dim,
                "tactile_dim": int(batch_dict["tactile_features"].shape[-1]),
                "loss_before": round(float(before.total_loss.detach().cpu().item()), 6),
                "train_step": {key: round(value, 6) for key, value in metrics.items()},
                "loss_after": round(float(after.total_loss.detach().cpu().item()), 6),
            }
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()