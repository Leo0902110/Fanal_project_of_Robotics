import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs

import src.env.material_pick_cube  # noqa: F401
from src.perception import PseudoBlurConfig, VisualUncertaintyDetector, apply_pseudo_blur


SENSOR_OBS_MODES = {
    "sensor_data",
    "rgb",
    "depth",
    "rgbd",
    "rgb+depth",
    "rgb+depth+segmentation",
    "rgb+segmentation",
    "depth+segmentation",
    "pointcloud",
}

class ManiSkillAgent:
    """
    统一机器人仿真接口类 (Vision-Tactile Fusion DP)
    集成了：环境初始化、观测值递归解析、视触觉数据提取、自动视频录制。
    """
    def __init__(
        self,
        env_id="PickCube-v1",
        obs_mode="rgbd",
        control_mode=None,
        render_mode="rgb_array",
        render_backend=None,
        object_profile: str = "default",
        pseudo_blur: PseudoBlurConfig | None = None,
        uncertainty_threshold=0.18,
    ):
        object_profile = object_profile.strip().lower()
        if object_profile != "default" and env_id == "PickCube-v1":
            env_id = "MaterialPickCube-v1"
        self.env_id = env_id
        self.obs_mode = obs_mode
        self.render_mode = render_mode
        self.render_backend = render_backend
        self.object_profile = object_profile
        self.pseudo_blur = pseudo_blur or PseudoBlurConfig(enabled=False)
        self.detector = VisualUncertaintyDetector(threshold=uncertainty_threshold)
        self._pseudo_blur_rng = np.random.default_rng(self.pseudo_blur.seed)
        normalized_render_backend = self._normalize_render_backend(obs_mode, render_backend)
        if normalized_render_backend != render_backend:
            print(
                f"obs_mode={obs_mode} 需要渲染后端；将 render_backend 从 {render_backend!r} 调整为 {normalized_render_backend!r}。"
            )
        self.render_backend = normalized_render_backend
        kwargs = {"obs_mode": obs_mode}
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        if self.render_backend is not None:
            kwargs["render_backend"] = self.render_backend
        if control_mode:
            kwargs["control_mode"] = control_mode
        if env_id == "MaterialPickCube-v1":
            kwargs["object_profile"] = object_profile
        try:
            self.env = gym.make(env_id, **kwargs)
            print(f"环境 {env_id} 初始化成功，obs_mode={obs_mode}")
        except Exception as exc:
            print(f"环境 {env_id} 初始化失败：{exc}")
            print("回退至 PickCube-v1 + state + 无渲染模式，确保 Colab smoke test 能继续。")
            self.obs_mode = "state"
            self.render_mode = None
            self.render_backend = "none"
            self.env = gym.make("PickCube-v1", obs_mode="state", render_backend="none")

        self.frames = []
        self.last_info = {}

    def reset(self, seed=None):
        episode_seed = self.pseudo_blur.seed if seed is None else int(self.pseudo_blur.seed + seed)
        self._pseudo_blur_rng = np.random.default_rng(episode_seed)
        obs, info = self.env.reset(seed=seed)
        self.frames = []
        self.last_info = self._process_info(info)
        return self._prepare_obs(obs)

    def _process_obs(self, obs):
        """递归处理嵌套字典，确保底层数据都是 Numpy 数组"""
        if isinstance(obs, dict):
            return {k: self._process_obs(v) for k, v in obs.items()}
        elif isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        
        if isinstance(obs, np.ndarray) and obs.ndim >= 4:
            obs = obs[0]
        return obs

    def _process_info(self, info):
        return self._process_obs(info) if isinstance(info, dict) else {}

    def _prepare_obs(self, obs):
        obs = self._process_obs(obs)
        return apply_pseudo_blur(obs, self.pseudo_blur, rng=self._pseudo_blur_rng)

    def get_visual_uncertainty(self, obs):
        """视觉伪模糊检测 (Visual Pseudo-blur)"""
        return self.detector.estimate(obs)

    def get_task_state(self):
        env = getattr(self.env, "unwrapped", self.env)

        def to_numpy(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            return np.asarray(value, dtype=np.float32)

        state = {}
        if hasattr(env, "agent") and hasattr(env.agent, "tcp"):
            state["tcp_pos"] = to_numpy(env.agent.tcp.pose.p)
        if hasattr(env, "agent") and hasattr(env.agent, "robot"):
            try:
                state["qpos"] = to_numpy(env.agent.robot.get_qpos())
            except Exception:
                pass
        if hasattr(env, "cube"):
            state["obj_pos"] = to_numpy(env.cube.pose.p)
        if hasattr(env, "goal_site"):
            state["goal_pos"] = to_numpy(env.goal_site.pose.p)
        return state

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_info = self._process_info(info)
        self._append_frame()
        done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        return self._prepare_obs(obs), float(np.asarray(reward).mean()), done, self.last_info

    def _append_frame(self):
        if self.render_mode is None:
            return
        try:
            frame = self._process_obs(self.env.render())
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                self.frames.append(frame.astype(np.uint8))
        except Exception:
            pass

    def save_video(self, filename="output.mp4", fps=25):
        if not self.frames:
            print("没有可保存的视频帧，可能是当前 Colab 渲染不可用或使用了 state 模式。")
            return None

        imageio.mimsave(filename, self.frames, fps=fps)
        print(f"视频已存至: {filename}")
        return filename

    def close(self):
        self.env.close()

    def _normalize_render_backend(self, obs_mode, render_backend):
        if obs_mode not in SENSOR_OBS_MODES:
            return render_backend
        if render_backend in (None, "none"):
            return "cpu"
        return render_backend
