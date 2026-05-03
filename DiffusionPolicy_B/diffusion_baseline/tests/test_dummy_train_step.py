# 用途: 测试 encoder、diffusion_net、schedule、buffer 能完成一次 forward/loss/backward/step。
# Purpose: Test one minimal forward/loss/backward/optimizer step using all baseline components.

import torch

from diffusion_baseline.models.diffusion_net import DiffusionPolicyNet
from diffusion_baseline.models.encoder import CNNEncoder
from diffusion_baseline.models.schedule import DiffusionSchedule
from diffusion_baseline.utils.buffer import ReplayBuffer


def test_dummy_train_step_backward() -> None:
    torch.manual_seed(0)
    batch_size = 2
    repr_dim = 128
    state_dim = 7
    action_horizon = 8
    action_dim = 4

    encoder = CNNEncoder(in_channels=3, repr_dim=repr_dim)
    net = DiffusionPolicyNet(
        repr_dim=repr_dim,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
    )
    schedule = DiffusionSchedule(num_timesteps=100)
    buffer = ReplayBuffer(capacity=8)

    for _ in range(4):
        buffer.push(
            image=torch.randint(0, 256, (64, 64, 3), dtype=torch.uint8),
            state=torch.randn(state_dim, dtype=torch.float32),
            action=torch.randn(action_horizon, action_dim, dtype=torch.float32),
        )

    batch = buffer.sample(batch_size=batch_size)
    images = batch["image"]
    states = batch["state"]
    actions = batch["action"]
    noise = torch.randn_like(actions)
    timesteps = torch.randint(0, schedule.num_timesteps, (batch_size,), dtype=torch.long)

    assert images.shape == (batch_size, 64, 64, 3)
    assert images.dtype == torch.uint8
    assert states.shape == (batch_size, state_dim)
    assert states.dtype == torch.float32
    assert actions.shape == (batch_size, action_horizon, action_dim)
    assert actions.dtype == torch.float32
    assert timesteps.dtype == torch.long

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(net.parameters()), lr=1e-3)
    encoded = encoder(images)
    noisy_actions = schedule.q_sample(actions, timesteps, noise)
    pred_noise = net(encoded, states, noisy_actions, timesteps)
    loss = torch.nn.functional.mse_loss(pred_noise, noise)

    assert encoded.shape == (batch_size, repr_dim)
    assert noisy_actions.shape == actions.shape
    assert pred_noise.shape == actions.shape
    assert loss.ndim == 0
    assert loss.dtype == torch.float32

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
