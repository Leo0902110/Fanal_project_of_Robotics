# 用途: 测试 ReplayBuffer 的 push/sample 以及采样 batch 的 shape。
# Purpose: Test ReplayBuffer push/sample behavior and sampled batch shapes.

import torch

from diffusion_baseline.utils.buffer import ReplayBuffer


def test_replay_buffer_sample_shapes() -> None:
    buffer = ReplayBuffer(capacity=16)
    repr_dim = 128
    state_dim = 7
    action_horizon = 8
    action_dim = 4

    for _ in range(10):
        buffer.push(
            encoded=torch.randn(repr_dim, dtype=torch.float32),
            state=torch.randn(state_dim, dtype=torch.float32),
            action=torch.randn(action_horizon, action_dim, dtype=torch.float32),
        )

    batch = buffer.sample(batch_size=4)
    assert len(buffer) == 10
    assert batch["encoded"].dtype == torch.float32
    assert batch["state"].dtype == torch.float32
    assert batch["action"].dtype == torch.float32
    assert batch["encoded"].shape == (4, repr_dim)
    assert batch["state"].shape == (4, state_dim)
    assert batch["action"].shape == (4, action_horizon, action_dim)
