from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PseudoBlurConfig:
    """Configuration for lightweight pseudo-blur simulation in RGB-D observations."""

    enabled: bool = False
    depth_noise_std: float = 0.015
    dropout_prob: float = 0.08
    state_uncertainty: float = 0.65
    seed: int = 0


def _as_numpy(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return None


def find_depth_array(obs: Any) -> np.ndarray | None:
    """Find the first depth-like array inside ManiSkill nested observations."""
    if isinstance(obs, dict):
        for key in ("depth", "Position", "position"):
            arr = _as_numpy(obs.get(key))
            if arr is not None and arr.size > 0:
                return arr
        for value in obs.values():
            depth = find_depth_array(value)
            if depth is not None:
                return depth
    return None


def apply_pseudo_blur(obs: Any, config: PseudoBlurConfig) -> Any:
    """Apply in-place depth degradation to simulate transparent/dark-object ambiguity."""
    if not config.enabled:
        return obs
    if not isinstance(obs, dict):
        return {
            "state": obs,
            "_pseudo_blur_uncertainty": float(config.state_uncertainty),
        }

    rng = np.random.default_rng(config.seed)

    def degrade(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, value in list(node.items()):
            arr = _as_numpy(value)
            if key.lower() == "depth" and arr is not None and arr.size > 0:
                degraded = arr.astype(np.float32, copy=True)
                valid = np.isfinite(degraded) & (degraded > 0)
                noise = rng.normal(0.0, config.depth_noise_std, degraded.shape)
                dropout = rng.random(degraded.shape) < config.dropout_prob
                degraded[valid] = degraded[valid] + noise[valid]
                degraded[dropout] = 0.0
                node[key] = degraded
            else:
                degrade(value)

    degrade(obs)
    return obs

