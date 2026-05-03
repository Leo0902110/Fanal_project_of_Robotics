# 用途: 提供简单 replay buffer，用于测试最小训练闭环。
# Purpose: Provide a simple replay buffer for smoke tests and minimal training loops.

from __future__ import annotations

import random
from collections import deque
from typing import Any

import torch


class ReplayBuffer:
    """Fixed-capacity replay buffer storing tensor transitions."""

    def __init__(self, capacity: int = 10000) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._storage: deque[dict[str, torch.Tensor]] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._storage)

    def push(self, **transition: torch.Tensor) -> None:
        if not transition:
            raise ValueError("transition cannot be empty")
        cleaned: dict[str, torch.Tensor] = {}
        for key, value in transition.items():
            if not torch.is_tensor(value):
                value = torch.as_tensor(value)
            cleaned[key] = value.detach().clone()
        self._storage.append(cleaned)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if batch_size > len(self._storage):
            raise ValueError("batch_size cannot exceed current buffer length")
        samples = random.sample(list(self._storage), batch_size)
        keys = samples[0].keys()
        return {key: torch.stack([sample[key] for sample in samples], dim=0) for key in keys}


def make_transition(**kwargs: Any) -> dict[str, torch.Tensor]:
    """Convert keyword values into a tensor transition dictionary."""

    return {key: torch.as_tensor(value) for key, value in kwargs.items()}
