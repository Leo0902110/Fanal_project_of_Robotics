from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


class SimpleBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], seed: int = 0):
        self.low = np.full(shape, low, dtype=np.float32)
        self.high = np.full(shape, high, dtype=np.float32)
        self.shape = shape
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        return self._rng.uniform(self.low, self.high).astype(np.float32)


@dataclass
class MockSceneConfig:
    obs_mode: str = "rgbd"
    pseudo_blur_enabled: bool = False
    max_steps: int = 120
    image_size: int = 32


class MockGraspEnv:
    """Tiny pick-and-place fallback env for pipeline validation.

    It mirrors the minimum ManiSkill interfaces the repo relies on:
    - reset/step/render/close
    - action_space
    - `cube` and `goal_site` pose-like accessors through `unwrapped`
    """

    def __init__(self, config: MockSceneConfig):
        self.config = config
        self.action_space = SimpleBox(-1.0, 1.0, (7,))
        self.step_count = 0
        self.last_contact = 0.0
        self.grasped = False
        self._seed = 0
        self._frame = None
        self.unwrapped = self
        self._build_pose_views()

    def _build_pose_views(self) -> None:
        class PoseView:
            def __init__(self, env: "MockGraspEnv", attr: str):
                self._env = env
                self._attr = attr

            @property
            def pose(self):
                class Pose:
                    def __init__(self, value: np.ndarray):
                        self.p = value

                return Pose(getattr(self._env, self._attr))

        self.cube = PoseView(self, "obj_pos")
        self.goal_site = PoseView(self, "goal_pos")

    def reset(self, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self._seed = int(seed or 0)
        self._rng = np.random.default_rng(self._seed)
        self.step_count = 0
        self.last_contact = 0.0
        self.grasped = False
        self.tcp_pos = np.array([0.0, 0.0, 0.20], dtype=np.float32)
        self.obj_pos = np.array([0.08, 0.02, 0.04], dtype=np.float32)
        self.goal_pos = np.array([-0.10, -0.08, 0.08], dtype=np.float32)
        obs = self._observation()
        return obs, {"success": 0.0}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        delta = np.clip(action[:3], -1.0, 1.0) * 0.025
        self.tcp_pos = self.tcp_pos + delta
        self.step_count += 1

        distance = float(np.linalg.norm(self.tcp_pos - self.obj_pos))
        contact = max(0.0, 1.0 - distance / 0.06)
        self.last_contact = contact if distance < 0.06 else 0.0

        gripper_close = float(action[-1]) < -0.2 if action.size else False
        if self.last_contact > 0.55 and gripper_close:
            self.grasped = True

        if self.grasped:
            self.obj_pos = self.tcp_pos + np.array([0.0, 0.0, -0.02], dtype=np.float32)

        goal_distance = float(np.linalg.norm(self.obj_pos - self.goal_pos))
        success = float(self.grasped and goal_distance < 0.05 and self.tcp_pos[2] > 0.08)
        reward = 1.0 - min(distance, 1.0) - 0.2 * goal_distance + 2.0 * success
        terminated = bool(success)
        truncated = self.step_count >= self.config.max_steps
        obs = self._observation()
        info = {
            "success": success,
            "contact": np.array([self.last_contact], dtype=np.float32),
            "tcp_wrench": np.array([self.last_contact, 0.0, 0.0], dtype=np.float32),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray:
        return self._frame if self._frame is not None else np.zeros((256, 256, 3), dtype=np.uint8)

    def close(self) -> None:
        return None

    def _observation(self) -> dict[str, Any]:
        rgb, depth = self._camera_arrays()
        tcp_pose = np.concatenate([self.tcp_pos, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)])
        obs = {
            "extra": {
                "tcp_pose": tcp_pose,
                "goal_pos": self.goal_pos.copy(),
                "contact_force": np.array([self.last_contact], dtype=np.float32),
            }
        }
        if self.config.obs_mode == "state":
            obs["agent"] = np.concatenate([self.tcp_pos, self.obj_pos, self.goal_pos], dtype=np.float32)
            return obs

        obs["sensor_data"] = {
            "base_camera": {
                "rgb": rgb,
                "depth": depth,
            }
        }
        self._frame = (rgb * 255.0).astype(np.uint8)
        return obs

    def _camera_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        size = self.config.image_size
        rgb = np.full((size, size, 3), 0.85, dtype=np.float32)
        depth = np.full((size, size), 1.0, dtype=np.float32)

        px = int(np.clip((self.obj_pos[0] + 0.2) / 0.4 * (size - 1), 4, size - 5))
        py = int(np.clip((self.obj_pos[1] + 0.2) / 0.4 * (size - 1), 4, size - 5))
        rgb[py - 3 : py + 3, px - 3 : px + 3, :] = np.array([0.08, 0.08, 0.08], dtype=np.float32)
        depth[py - 3 : py + 3, px - 3 : px + 3] = 0.55

        if self.config.pseudo_blur_enabled:
            depth[py - 2 : py + 2, px + 1 : px + 4] = 0.0
            rgb[py - 2 : py + 2, px + 1 : px + 4, :] = 0.02

        gx = int(np.clip((self.goal_pos[0] + 0.2) / 0.4 * (size - 1), 2, size - 3))
        gy = int(np.clip((self.goal_pos[1] + 0.2) / 0.4 * (size - 1), 2, size - 3))
        rgb[gy - 2 : gy + 2, gx - 2 : gx + 2, :] = np.array([0.2, 0.7, 0.2], dtype=np.float32)
        return rgb, depth
