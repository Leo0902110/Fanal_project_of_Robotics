# 用途: 测试 DiffusionPolicyNet 前向传播输出 shape 与 noisy_actions 完全一致。
# Purpose: Test DiffusionPolicyNet forward output shape matches noisy_actions.

import torch

from diffusion_baseline.models.diffusion_net import DiffusionPolicyNet


def test_diffusion_policy_net_forward_shape() -> None:
    batch_size = 2
    repr_dim = 128
    state_dim = 7
    action_horizon = 8
    action_dim = 4

    net = DiffusionPolicyNet(
        repr_dim=repr_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
    )
    encoded = torch.randn(batch_size, repr_dim, dtype=torch.float32)
    state = torch.randn(batch_size, state_dim, dtype=torch.float32)
    noisy_actions = torch.randn(batch_size, action_horizon, action_dim, dtype=torch.float32)
    timesteps = torch.randint(0, 100, (batch_size,), dtype=torch.long)

    output = net(encoded, state, noisy_actions, timesteps)
    assert encoded.dtype == torch.float32
    assert state.dtype == torch.float32
    assert noisy_actions.dtype == torch.float32
    assert timesteps.dtype == torch.long
    assert output.dtype == torch.float32
    assert output.shape == noisy_actions.shape
