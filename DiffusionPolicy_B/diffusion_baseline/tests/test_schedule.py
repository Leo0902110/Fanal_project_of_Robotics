# 用途: 测试扩散 q_sample 前向加噪函数保持动作张量 shape。
# Purpose: Test diffusion q_sample keeps the action tensor shape unchanged.

import torch

from diffusion_baseline.models.schedule import DiffusionSchedule, q_sample


def test_diffusion_schedule_q_sample_shape() -> None:
    batch_size = 2
    action_horizon = 8
    action_dim = 4
    actions = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32)
    noise = torch.randn_like(actions)
    timesteps = torch.randint(0, 100, (batch_size,), dtype=torch.long)

    schedule = DiffusionSchedule(num_timesteps=100)
    noisy_actions = schedule.q_sample(actions, timesteps, noise)
    assert actions.dtype == torch.float32
    assert noise.dtype == torch.float32
    assert timesteps.dtype == torch.long
    assert noisy_actions.dtype == torch.float32
    assert noisy_actions.shape == actions.shape

    noisy_actions_from_function = q_sample(actions, timesteps, noise, num_timesteps=100)
    assert noisy_actions_from_function.shape == actions.shape
