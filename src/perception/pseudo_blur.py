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
    if not config.enabled or not isinstance(obs, dict):
        return obs

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


class VisualUncertaintyDetector:
    """Depth-based pseudo-blur detector with normalized, threshold-friendly output."""

    def __init__(self, threshold: float = 0.18):
        self.threshold = threshold

    def _normalize_depth_units(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.size and np.nanmax(depth) > 20.0:
            return depth / 1000.0
        return depth

    def estimate(self, obs: Any) -> dict[str, float | bool]:
        depth = find_depth_array(obs)
        if depth is None:
            return {
                "uncertainty": 0.0,
                "depth_variance": 0.0,
                "missing_ratio": 0.0,
                "triggered": False,
            }

        depth = self._normalize_depth_units(depth)
        valid = np.isfinite(depth) & (depth > 0)
        missing_ratio = float(1.0 - valid.mean()) if depth.size else 1.0

        if valid.any():
            values = depth[valid]
            mean = float(np.mean(values))
            variance = float(np.var(values))
        else:
            variance = 0.0

        # For the MVP, pseudo-blur is simulated mostly as depth dropout. Global
        # depth variance is dominated by camera perspective/background distance,
        # so it is reported but not used as the primary trigger.
        uncertainty = float(np.clip(missing_ratio / 0.12, 0.0, 1.0))
        return {
            "uncertainty": uncertainty,
            "depth_variance": variance,
            "missing_ratio": missing_ratio,
            "triggered": uncertainty >= self.threshold,
        }
