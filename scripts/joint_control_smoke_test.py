import argparse
import warnings
from collections.abc import Mapping

import gymnasium as gym
import numpy as np

warnings.filterwarnings(
    "ignore",
    message="pinnochio package is not installed, robotics functionalities will not be available",
)

import mani_skill.envs  # noqa: F401


def summarize_value(value):
    if isinstance(value, Mapping):
        return {key: summarize_value(sub_value) for key, sub_value in value.items()}

    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None:
        return {
            "type": type(value).__name__,
            "shape": tuple(shape),
            "dtype": str(dtype),
        }

    return {"type": type(value).__name__, "value": value}


def main():
    parser = argparse.ArgumentParser(
        description="Run a minimal ManiSkill joint-control smoke test."
    )
    parser.add_argument("--task", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = gym.make(
        args.task,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
    )

    try:
        obs, info = env.reset(seed=args.seed)
        zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)

        print(f"task={args.task}")
        print(f"obs_mode={args.obs_mode}")
        print(f"control_mode={args.control_mode}")
        print(f"action_space={env.action_space}")
        print(f"initial_obs={summarize_value(obs)}")
        print(f"initial_info_keys={sorted(info.keys())}")

        for step_idx in range(args.steps):
            obs, reward, terminated, truncated, info = env.step(zero_action)
            print(
                f"step={step_idx} reward={reward} terminated={terminated} truncated={truncated}"
            )

            if bool(np.asarray(terminated).any()) or bool(np.asarray(truncated).any()):
                print("environment finished early")
                break

        print(f"final_obs={summarize_value(obs)}")
        print(f"final_info_keys={sorted(info.keys())}")
    finally:
        env.close()


if __name__ == "__main__":
    main()