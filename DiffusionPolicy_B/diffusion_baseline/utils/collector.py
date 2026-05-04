# 用途: 从 ManiSkill/Gym 风格环境同步采样 transition，并兼容 fallback 环境。
# Purpose: Collect transitions from ManiSkill/Gym-style environments with a fallback env.

from __future__ import annotations

from dataclasses import dataclass
import subprocess
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from diffusion_baseline.utils.buffer import ReplayBuffer


@dataclass
class StepResult:
    image: torch.Tensor
    state: torch.Tensor
    reward: float
    done: bool
    info: dict[str, Any]
    action_before_clamp: torch.Tensor | None = None
    clamped_action: torch.Tensor | None = None
    prev_image: torch.Tensor | None = None
    prev_state: torch.Tensor | None = None


class SimpleBox:
    def __init__(self, low: float, high: float, shape: tuple[int, ...], dtype: Any = np.float32) -> None:
        self.low = np.full(shape, low, dtype=dtype)
        self.high = np.full(shape, high, dtype=dtype)
        self.shape = shape
        self.dtype = dtype

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.low, self.high).astype(self.dtype)


class FallbackReachEnv:
    """Tiny reaching env; state includes current position and hidden target for diagnostics/training."""

    metadata: dict[str, Any] = {}

    def __init__(self, action_dim: int = 4, state_dim: int = 32, image_size: int = 64, max_steps: int = 100) -> None:
        self.action_space = SimpleBox(-1.0, 1.0, (action_dim,), np.float32)
        self.state_dim = state_dim
        self.image_size = image_size
        self.max_steps = max_steps
        self._rng = np.random.default_rng(0)
        self._step = 0
        self._state = np.zeros(state_dim, dtype=np.float32)
        self._target = np.zeros(action_dim, dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step = 0
        self._state = self._rng.normal(0.0, 0.25, size=self.state_dim).astype(np.float32)
        self._target = self._rng.uniform(-0.5, 0.5, size=self.action_space.shape[0]).astype(np.float32)
        return self._obs(), {}

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._step += 1
        action_dim = self.action_space.shape[0]
        self._state[:action_dim] = self._state[:action_dim] + 0.08 * action
        distance = float(np.linalg.norm(self._state[:action_dim] - self._target))
        reward = -distance
        success = distance < 0.15
        terminated = success
        truncated = self._step >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"success": success, "distance": distance}

    def close(self) -> None:
        return None

    def _obs(self) -> dict[str, Any]:
        action_dim = self.action_space.shape[0]
        state = self._state.copy()
        state[action_dim : 2 * action_dim] = self._target
        rgb = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        current = np.clip((state[:3] + 1.0) * 95.0, 0, 255).astype(np.uint8)
        target = np.clip((self._target[:3] + 1.0) * 95.0, 0, 255).astype(np.uint8)
        rgb[..., 0] = current[0]
        rgb[..., 1] = target[1]
        rgb[..., 2] = ((int(current[2]) + int(target[2])) // 2)
        return {"rgb": rgb, "state": state}


class EnvCollector:
    """Single-env synchronous collector for random, heuristic, or policy actions."""

    def __init__(
        self,
        env_id: str = "PickCube-v1",
        state_dim: int = 32,
        image_size: int = 64,
        seed: int = 0,
        max_episode_steps: int = 100,
        env_backend: str = "auto",
    ) -> None:
        self.state_dim = state_dim
        self.image_size = image_size
        self.seed = seed
        self.max_episode_steps = max_episode_steps
        self.env_backend = env_backend
        self.env, self.env_name = self._make_env(env_id)
        self.action_low = self._space_array("low")
        self.action_high = self._space_array("high")
        self.action_shape = tuple(int(x) for x in self.env.action_space.shape)
        if not self.action_shape:
            raise ValueError("Only continuous action spaces with a non-empty shape are supported.")
        self.action_dim = int(np.prod(self.action_shape))
        self._obs: Any = None
        self.reset()

    def _make_env(self, env_id: str) -> tuple[Any, str]:
        if self.env_backend == "fallback" or env_id.lower() in {"fallback", "fallbackreachenv"}:
            print("collector_env=FallbackReachEnv")
            return FallbackReachEnv(state_dim=self.state_dim, image_size=self.image_size, max_steps=self.max_episode_steps), "FallbackReachEnv"
        if self.env_backend == "auto" and not self._probe_env_in_subprocess(env_id):
            print("[警告] ManiSkill/Gym env 子进程探测失败，主进程使用 fallback env 以避免 native crash。")
            return FallbackReachEnv(state_dim=self.state_dim, image_size=self.image_size, max_steps=self.max_episode_steps), "FallbackReachEnv"
        try:
            import gymnasium as gym

            try:
                import mani_skill.envs  # noqa: F401
            except Exception:
                pass
            attempts = [
                {"obs_mode": "state"},
                {"obs_mode": "rgb", "render_mode": "rgb_array"},
                {},
            ]
            last_error: Exception | None = None
            for kwargs in attempts:
                try:
                    env = gym.make(env_id, **kwargs)
                    self._obs_mode = kwargs.get("obs_mode", "default")
                    print(f"collector_env={env_id} kwargs={kwargs}")
                    return env, env_id
                except Exception as exc:
                    last_error = exc
            print(f"[警告] ManiSkill/Gym env 创建失败，使用 fallback env: {last_error}")
        except Exception as exc:
            print(f"[警告] gymnasium/mani_skill 不可用，使用 fallback env: {exc}")
        self._obs_mode = "fallback"
        return FallbackReachEnv(state_dim=self.state_dim, image_size=self.image_size, max_steps=self.max_episode_steps), "FallbackReachEnv"

    def _probe_env_in_subprocess(self, env_id: str) -> bool:
        code = f"""
import os
os.environ.setdefault("MANISKILL_GPU_SIM", "0")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import gymnasium as gym
try:
    import mani_skill.envs  # noqa: F401
except Exception:
    pass
attempts = [
    dict(obs_mode='state', sim_backend='cpu'),
    dict(obs_mode='rgb', render_mode='rgb_array', sim_backend='cpu'),
    dict(sim_backend='cpu'),
]
last = None
for kwargs in attempts:
    try:
        env = gym.make({env_id!r}, **kwargs)
        env.reset(seed=0)
        env.close()
        raise SystemExit(0)
    except Exception as exc:
        last = exc
print(repr(last))
raise SystemExit(1)
"""
        try:
            result = subprocess.run(
                [sys.executable, "-c", code],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except Exception as exc:
            print(f"[警告] env 探测异常: {exc}")
            return False
        if result.returncode != 0:
            details = (result.stdout + result.stderr).strip()
            print(f"[警告] env 探测失败 returncode={result.returncode}" + (f": {details[-500:]}" if details else ""))
            return False
        return True

    def reset(self) -> StepResult:
        result = self.env.reset(seed=self.seed)
        self._obs = result[0] if isinstance(result, tuple) else result
        image, state = self.extract_observation(self._obs)
        return StepResult(image=image, state=state, reward=0.0, done=False, info={})

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()

    def current_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.extract_observation(self._obs)

    def random_action(self) -> np.ndarray:
        return np.asarray(self.env.action_space.sample(), dtype=np.float32).reshape(self.action_dim)

    def heuristic_action(self) -> np.ndarray:
        """FallbackReachEnv oracle action; random action for real external envs."""

        if self.env_name == "FallbackReachEnv" and hasattr(self.env, "_target") and hasattr(self.env, "_state"):
            delta = (self.env._target - self.env._state[: self.action_dim]) / 0.08
            return self.clip_action(delta)
        return self.random_action()

    def clip_action(self, action: np.ndarray | torch.Tensor) -> np.ndarray:
        action_np = np.asarray(action.detach().cpu() if torch.is_tensor(action) else action, dtype=np.float32)
        action_np = action_np.reshape(self.action_dim)
        return np.clip(action_np, self.action_low.reshape(self.action_dim), self.action_high.reshape(self.action_dim))

    def step(self, action: np.ndarray | torch.Tensor, buffer: ReplayBuffer | None = None) -> StepResult:
        image, state = self.extract_observation(self._obs)
        raw_action = np.asarray(action.detach().cpu() if torch.is_tensor(action) else action, dtype=np.float32).reshape(self.action_dim)
        clamped_action = self.clip_action(raw_action)
        env_action = clamped_action.reshape(self.action_shape)
        result = self.env.step(env_action)
        if len(result) == 5:
            next_obs, reward, terminated, truncated, info = result
            done = bool(terminated or truncated)
        else:
            next_obs, reward, done, info = result
            done = bool(done)

        next_image, next_state = self.extract_observation(next_obs)
        raw_tensor = torch.as_tensor(raw_action, dtype=torch.float32)
        clamped_tensor = torch.as_tensor(clamped_action, dtype=torch.float32)
        if buffer is not None:
            buffer.push_transition(
                {
                    "image": image,
                    "state": state,
                    "action": clamped_tensor,
                    "raw_action": raw_tensor,
                    "clamped_action": clamped_tensor,
                    "reward": torch.as_tensor(float(reward), dtype=torch.float32),
                    "next_image": next_image,
                    "next_state": next_state,
                    "done": torch.as_tensor(done, dtype=torch.bool),
                }
            )
        if done:
            reset_result = self.env.reset()
            self._obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        else:
            self._obs = next_obs
        return StepResult(
            image=next_image,
            state=next_state,
            reward=float(reward),
            done=done,
            info=dict(info),
            action_before_clamp=raw_tensor,
            clamped_action=clamped_tensor,
            prev_image=image,
            prev_state=state,
        )

    def collect_random(self, buffer: ReplayBuffer, steps: int) -> None:
        for _ in range(steps):
            self.step(self.random_action(), buffer=buffer)

    def extract_observation(self, obs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        image = self._extract_image(obs)
        state = self._extract_state(obs)
        assert image.shape == (self.image_size, self.image_size, 3), f"image shape mismatch: {tuple(image.shape)}"
        assert state.shape == (self.state_dim,), f"state shape mismatch: {tuple(state.shape)}"
        return image, state

    def _space_array(self, attr: str) -> np.ndarray:
        value = getattr(self.env.action_space, attr)
        if torch.is_tensor(value):
            value = value.detach().cpu().numpy()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    def _extract_image(self, obs: Any) -> torch.Tensor:
        candidate = self._find_image_array(obs)
        if candidate is None:
            obs_mode = getattr(self, "_obs_mode", "default")
            if obs_mode in ("state", "state_dict"):
                state_arr = self._raw_state_array(obs)
                candidate = self._synthesize_image_from_state(state_arr)
            else:
                candidate = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        arr = np.asarray(candidate)
        while arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim != 3 or arr.shape[-1] < 3:
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            else:
                arr = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)
        arr = arr[..., :3]
        if arr.dtype != np.uint8:
            arr = np.asarray(arr, dtype=np.float32)
            if arr.max(initial=0.0) <= 1.5:
                arr = arr * 255.0
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        tensor = torch.as_tensor(arr, dtype=torch.uint8)
        if tensor.shape[:2] != (self.image_size, self.image_size):
            chw = tensor.permute(2, 0, 1).unsqueeze(0).float()
            resized = F.interpolate(chw, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
            tensor = resized.squeeze(0).permute(1, 2, 0).clamp(0, 255).to(torch.uint8)
        return tensor.contiguous()

    def _extract_state(self, obs: Any) -> torch.Tensor:
        arrays: list[np.ndarray] = []
        self._collect_state_arrays(obs, arrays)
        flat = np.concatenate([arr.reshape(-1).astype(np.float32) for arr in arrays], axis=0) if arrays else np.zeros((0,), dtype=np.float32)
        out = np.zeros((self.state_dim,), dtype=np.float32)
        count = min(self.state_dim, flat.shape[0])
        if count:
            out[:count] = flat[:count]
        return torch.as_tensor(out, dtype=torch.float32)

    def _find_image_array(self, value: Any) -> Any | None:
        if isinstance(value, dict):
            for key in ("rgb", "image", "pixels"):
                if key in value and self._looks_like_image(value[key]):
                    return value[key]
            for child in value.values():
                found = self._find_image_array(child)
                if found is not None:
                    return found
        elif isinstance(value, (list, tuple)):
            for child in value:
                found = self._find_image_array(child)
                if found is not None:
                    return found
        elif self._looks_like_image(value):
            return value
        return None

    def _collect_state_arrays(self, value: Any, arrays: list[np.ndarray]) -> None:
        if isinstance(value, dict):
            for child in value.values():
                self._collect_state_arrays(child, arrays)
        elif isinstance(value, (list, tuple)):
            for child in value:
                self._collect_state_arrays(child, arrays)
        else:
            try:
                arr = np.asarray(value)
            except Exception:
                return
            if arr.size == 0 or not np.issubdtype(arr.dtype, np.number) or self._looks_like_image(arr):
                return
            arrays.append(arr.astype(np.float32))

    def _raw_state_array(self, obs: Any) -> np.ndarray:
        if isinstance(obs, dict):
            raw = obs.get("agent")
            if raw is not None:
                return np.asarray(raw, dtype=np.float32).reshape(-1)
        if isinstance(obs, np.ndarray):
            return obs.astype(np.float32).reshape(-1)
        if torch.is_tensor(obs):
            return obs.detach().cpu().float().numpy().reshape(-1)
        return np.zeros(0, dtype=np.float32)

    def _synthesize_image_from_state(self, state_arr: np.ndarray) -> np.ndarray:
        from PIL import Image, ImageDraw

        size = self.image_size
        img = Image.new("RGB", (size, size), color=(20, 20, 40))
        draw = ImageDraw.Draw(img)
        step = max(1, len(state_arr) // 48)
        cell = size / 8.0
        for i in range(0, len(state_arr), step):
            idx = i // step
            if idx >= 64:
                break
            row = idx // 8
            col = idx % 8
            val = float(np.clip((state_arr[i] + 1.0) / 2.0, 0.0, 1.0))
            r = int(val * 200 + 30)
            g = int((1.0 - val) * 180 + 30)
            b = int(abs(val - 0.5) * 400 + 30)
            x0 = int(col * cell) + 1
            y0 = int(row * cell) + 1
            x1 = int((col + 1) * cell) - 1
            y1 = int((row + 1) * cell) - 1
            draw.rectangle([x0, y0, x1, y1], fill=(r, g, b))
        return np.array(img, dtype=np.uint8)

    def _looks_like_image(self, value: Any) -> bool:
        try:
            arr = np.asarray(value)
        except Exception:
            return False
        return arr.ndim >= 3 and (arr.shape[-1] in (3, 4) or arr.shape[-3] in (3, 4))
