import numpy as np
import torch
import gymnasium as gym
import imageio.v2 as imageio
import mani_skill.envs

from src.perception import PseudoBlurConfig, VisualUncertaintyDetector, apply_pseudo_blur

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
        pseudo_blur: PseudoBlurConfig | None = None,
        uncertainty_threshold=0.18,
    ):
        self.env_id = env_id
        self.obs_mode = obs_mode
        self.pseudo_blur = pseudo_blur or PseudoBlurConfig(enabled=False)
        self.detector = VisualUncertaintyDetector(threshold=uncertainty_threshold)
        kwargs = {"render_mode": render_mode, "obs_mode": obs_mode}
        if control_mode:
            kwargs["control_mode"] = control_mode
        try:
            self.env = gym.make(env_id, **kwargs)
            print(f"环境 {env_id} 初始化成功，obs_mode={obs_mode}")
        except Exception as exc:
            print(f"环境 {env_id} 初始化失败：{exc}")
            print("回退至 PickCube-v1 + state 模式，确保 Colab smoke test 能继续。")
            self.obs_mode = "state"
            self.env = gym.make("PickCube-v1", render_mode=render_mode, obs_mode="state")

        self.frames = []
        self.last_info = {}

    def reset(self, seed=None):
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
        return apply_pseudo_blur(obs, self.pseudo_blur)

    def get_visual_uncertainty(self, obs):
        """视觉伪模糊检测 (Visual Pseudo-blur)"""
        return self.detector.estimate(obs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_info = self._process_info(info)
        self._append_frame()
        done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
        return self._prepare_obs(obs), float(np.asarray(reward).mean()), done, self.last_info

    def _append_frame(self):
        try:
            frame = self._process_obs(self.env.render())
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                self.frames.append(frame.astype(np.uint8))
        except Exception:
            pass

    def save_video(self, filename="output.mp4"):
        if not self.frames:
            print("没有可保存的视频帧，可能是当前 Colab 渲染不可用或使用了 state 模式。")
            return None

        imageio.mimsave(filename, self.frames, fps=25)
        print(f"视频已存至: {filename}")
        return filename

    def close(self):
        self.env.close()
