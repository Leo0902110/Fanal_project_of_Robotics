import argparse
import pathlib
import sys
import warnings

import gymnasium as gym
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="pinnochio package is not installed, robotics functionalities will not be available",
)

import mani_skill.envs  # noqa: F401

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.d_policy import ActiveTactilePolicy
from utils.d_features import (
    extract_contact_reading,
    extract_d_features,
    summarize_feature_bundle,
)
from utils.d_interface import DInferenceRequest


def main():
    parser = argparse.ArgumentParser(description="Roll out the minimal D policy.")
    parser.add_argument("--task", default="PickCube-v1")
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vision-dim", type=int, default=32)
    args = parser.parse_args()

    env = gym.make(
        args.task,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
    )

    try:
        obs, info = env.reset(seed=args.seed)
        tactile = extract_contact_reading(env)
        bundle = extract_d_features(obs, info, tactile)
        policy = ActiveTactilePolicy(
            action_dim=env.action_space.shape[0],
            vision_dim=args.vision_dim,
            tactile_dim=bundle.feature_vector.shape[0],
        )

        probe_steps_used = 0
        for step_idx in range(args.steps):
            tactile = extract_contact_reading(env)
            bundle = extract_d_features(obs, info, tactile)
            request = DInferenceRequest(
                feature_bundle=bundle,
                probe_steps_used=probe_steps_used,
                step_idx=step_idx,
            )
            output = policy.act_from_request(request)
            summary = summarize_feature_bundle(bundle)

            print(
                {
                    "step": step_idx,
                    "mode": output.policy_mode,
                    "uncertainty": round(output.uncertainty, 4),
                    "probe_triggered": output.probe_triggered,
                    "summary": summary,
                    "action": np.round(output.action, 4).tolist(),
                }
            )

            if output.probe_triggered:
                probe_steps_used += 1

            obs, reward, terminated, truncated, info = env.step(output.action)
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