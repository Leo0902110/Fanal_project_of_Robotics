import gymnasium as gym
import mani_skill.envs
import numpy as np
import cv2
import torch

class ManiSkillAgent:
    """
    统一机器人仿真接口类 (Vision-Tactile Fusion DP)
    集成了：环境初始化、观测值递归解析、视触觉数据提取、自动视频录制。
    """
    def __init__(self, env_id="PickCube-v1", render_mode="rgb_array"):
        self.env_id = env_id
        try:
            # 明确要求环境输出 RGB-D 视觉数据
            self.env = gym.make(env_id, render_mode=render_mode, obs_mode="rgbd")
            print(f"✅ 环境 {env_id} 初始化成功 (已开启视觉传感器)")
        except Exception:
            print(f"⚠️ 未找到 {env_id}，回退至基础任务 PickCube-v1")
            self.env = gym.make("PickCube-v1", render_mode=render_mode, obs_mode="rgbd")
        
        self.frames = []

    def reset(self):
        obs, info = self.env.reset()
        self.frames = []
        return self._process_obs(obs)

    def _process_obs(self, obs):
        """递归处理嵌套字典，确保底层数据都是 Numpy 数组"""
        if isinstance(obs, dict):
            return {k: self._process_obs(v) for k, v in obs.items()}
        elif isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        
        if isinstance(obs, np.ndarray) and obs.ndim >= 4:
            obs = obs[0]
        return obs

    def get_visual_uncertainty(self, obs):
        """视觉伪模糊检测 (Visual Pseudo-blur)"""
        if not isinstance(obs, dict):
            return 0.0
            
        try:
            sensor_data = obs.get('sensor_data', obs.get('image', {}))
            camera_data = sensor_data.get('base_camera', {})
            
            if 'depth' in camera_data:
                depth = camera_data['depth']
                return np.var(depth)
            return 0.0
        except Exception as e:
            print(f"⚠️ 视觉不确定性计算出错: {e}")
            return 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self.env.render())
        return self._process_obs(obs), reward, terminated or truncated

    def save_video(self, filename="output.mp4"):
        if not self.frames:
            return
        
        f0 = self._process_obs(self.frames[0])
        h, w, _ = f0.shape
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'mp4v'), 25, (w, h))
        
        for f in self.frames:
            f = self._process_obs(f)
            out.write(cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_RGB2BGR))
        
        out.release()
        print(f"🎬 视频已存至: {filename}")

    def close(self):
        self.env.close()