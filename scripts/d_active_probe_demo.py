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

from models.tactile_fusion import VisionTactileFusionMLP
from utils.d_active_probe import ActiveProbeConfig, select_action
from utils.d_features import extract_d_features, summarize_feature_bundle


def main():
    parser = argparse.ArgumentParser(
        description="Run a minimal active-probing demo for the D role."
    )
    parser.add_argument("--task", default="PickCube-v1")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vision-dim", type=int, default=32)
    args = parser.parse_args()

    env = gym.make(
        args.task,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
    )
    probe_config = ActiveProbeConfig()
    fusion_model = None
    probe_steps_used = 0

    try:
        obs, info = env.reset(seed=args.seed)
        base_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        for step_idx in range(args.steps):
            bundle = extract_d_features(obs, info)
            summary = summarize_feature_bundle(bundle)
            action, triggered, score = select_action(
                base_action=base_action,
                bundle=bundle,
                probe_steps_used=probe_steps_used,
                step_idx=step_idx,
                config=probe_config,
            )

            if fusion_model is None:
                fusion_model = VisionTactileFusionMLP(
                    vision_dim=args.vision_dim,
                    tactile_dim=bundle.feature_vector.shape[0],
                )

            vision_stub = torch.zeros((1, args.vision_dim), dtype=torch.float32)
            tactile_tensor = torch.from_numpy(bundle.feature_vector).float().unsqueeze(0)
            probe_flag = torch.tensor([1.0 if triggered else 0.0], dtype=torch.float32)
            fused = fusion_model(vision_stub, tactile_tensor, probe_flag)

            print(
                {
                    "step": step_idx,
                    "summary": summary,
                    "probe_triggered": triggered,
                    "probe_score": round(float(score), 4),
                    "action": np.round(action, 4).tolist(),
                    "fused_shape": tuple(fused.shape),
                }
            )

            if triggered:
                probe_steps_used += 1

            obs, reward, terminated, truncated, info = env.step(action)
            print(
                {
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "success": info.get("success"),
                }
            )

            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                break
    finally:
        env.close()


if __name__ == "__main__":
    main()