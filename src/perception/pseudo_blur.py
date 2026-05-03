from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class PseudoBlurConfig:
    """Configuration for lightweight pseudo-blur simulation in RGB-D observations."""

    enabled: bool = False
    profile: str = "mild"
    severity: float = 1.0
    depth_noise_std: float = 0.015
    dropout_prob: float = 0.08
    center_dropout_prob: float = 0.04
    boundary_dropout_prob: float = 0.06
    boundary_noise_std: float = 0.01
    degrade_position: bool = False
    rgb_noise_std: float = 0.0
    rgb_darken: float = 0.0
    rgb_desaturate: float = 0.0
    rgb_highlight_prob: float = 0.0
    state_uncertainty: float = 0.65
    seed: int = 0


PSEUDO_BLUR_PROFILES: dict[str, dict[str, float]] = {
    "mild": {
        "depth_noise_std": 0.015,
        "dropout_prob": 0.08,
        "center_dropout_prob": 0.0,
        "boundary_dropout_prob": 0.0,
        "boundary_noise_std": 0.0,
        "degrade_position": False,
        "state_uncertainty": 0.65,
    },
    "transparent": {
        "depth_noise_std": 0.028,
        "dropout_prob": 0.18,
        "center_dropout_prob": 0.22,
        "boundary_dropout_prob": 0.24,
        "boundary_noise_std": 0.025,
        "degrade_position": True,
        "rgb_desaturate": 0.25,
        "state_uncertainty": 0.82,
    },
    "dark": {
        "depth_noise_std": 0.022,
        "dropout_prob": 0.14,
        "center_dropout_prob": 0.18,
        "boundary_dropout_prob": 0.18,
        "boundary_noise_std": 0.018,
        "degrade_position": True,
        "rgb_noise_std": 0.03,
        "rgb_darken": 0.55,
        "state_uncertainty": 0.78,
    },
    "reflective": {
        "depth_noise_std": 0.035,
        "dropout_prob": 0.16,
        "center_dropout_prob": 0.18,
        "boundary_dropout_prob": 0.30,
        "boundary_noise_std": 0.035,
        "degrade_position": True,
        "rgb_highlight_prob": 0.05,
        "state_uncertainty": 0.86,
    },
    "low_texture": {
        "depth_noise_std": 0.018,
        "dropout_prob": 0.10,
        "center_dropout_prob": 0.10,
        "boundary_dropout_prob": 0.14,
        "boundary_noise_std": 0.016,
        "degrade_position": True,
        "rgb_desaturate": 0.65,
        "state_uncertainty": 0.72,
    },
}


def build_pseudo_blur_config(
    *,
    enabled: bool,
    seed: int,
    profile: str = "mild",
    severity: float = 1.0,
) -> PseudoBlurConfig:
    """Create a pseudo-blur config for proposal-style object appearance failures."""
    profile = profile.strip().lower()
    severity = max(0.0, float(severity))
    config = PseudoBlurConfig(enabled=enabled, profile=profile, severity=severity, seed=seed)
    if not enabled:
        return config
    if profile not in PSEUDO_BLUR_PROFILES:
        raise ValueError(f"Unknown pseudo-blur profile {profile!r}; choose from {sorted(PSEUDO_BLUR_PROFILES)}")
    for key, value in PSEUDO_BLUR_PROFILES[profile].items():
        if isinstance(value, bool):
            pass
        elif key.endswith("prob") or key.endswith("dropout") or key.endswith("dropout_prob"):
            value = min(1.0, value * severity)
        elif key.endswith("std"):
            value = value * severity
        elif key in {"rgb_darken", "rgb_desaturate", "state_uncertainty"}:
            value = min(1.0, value * severity)
        setattr(config, key, value)
    return config


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


def _is_depth_like(key: str, arr: np.ndarray, *, degrade_position: bool) -> bool:
    key = key.lower()
    if key == "depth" or key.endswith("depth"):
        return True
    return degrade_position and (key == "position" or (key.endswith("position") and arr.ndim >= 3))


def _is_rgb_like(key: str, arr: np.ndarray) -> bool:
    key = key.lower()
    return key in {"rgb", "color", "image"} and arr.ndim >= 3 and arr.shape[-1] in (3, 4)


def _spatial_mask(arr: np.ndarray, *, kind: str) -> np.ndarray:
    if arr.ndim < 2:
        return np.ones(arr.shape, dtype=bool)
    if arr.ndim >= 3 and arr.shape[-1] in (1, 3, 4):
        height_axis, width_axis = arr.ndim - 3, arr.ndim - 2
    else:
        height_axis, width_axis = arr.ndim - 2, arr.ndim - 1
    height, width = arr.shape[height_axis], arr.shape[width_axis]
    rows = np.linspace(-1.0, 1.0, max(height, 1), dtype=np.float32)
    cols = np.linspace(-1.0, 1.0, max(width, 1), dtype=np.float32)
    yy, xx = np.meshgrid(rows, cols, indexing="ij")
    radius = np.maximum(np.abs(xx), np.abs(yy))
    if kind == "center":
        mask2d = radius <= 0.55
    else:
        mask2d = (radius > 0.45) & (radius <= 0.82)
    shape = [1] * arr.ndim
    shape[height_axis] = height
    shape[width_axis] = width
    return np.broadcast_to(mask2d.reshape(shape), arr.shape)


def _degrade_depth_like(arr: np.ndarray, config: PseudoBlurConfig, rng: np.random.Generator) -> np.ndarray:
    degraded = arr.astype(np.float32, copy=True)
    valid = np.isfinite(degraded) & (degraded != 0)
    random_dropout = rng.random(degraded.shape) < config.dropout_prob
    center_dropout = (rng.random(degraded.shape) < config.center_dropout_prob) & _spatial_mask(degraded, kind="center")
    boundary_mask = _spatial_mask(degraded, kind="boundary")
    boundary_dropout = (rng.random(degraded.shape) < config.boundary_dropout_prob) & boundary_mask
    noise = rng.normal(0.0, config.depth_noise_std, degraded.shape)
    boundary_noise = rng.normal(0.0, config.boundary_noise_std, degraded.shape) * boundary_mask
    degraded[valid] = degraded[valid] + noise[valid] + boundary_noise[valid]
    degraded[random_dropout | center_dropout | boundary_dropout] = 0.0
    return degraded


def _degrade_rgb_like(arr: np.ndarray, config: PseudoBlurConfig, rng: np.random.Generator) -> np.ndarray:
    original_dtype = arr.dtype
    scale = 255.0 if np.issubdtype(original_dtype, np.integer) else 1.0
    degraded = arr.astype(np.float32, copy=True) / scale
    rgb = degraded[..., :3]
    if config.rgb_darken > 0.0:
        rgb *= max(0.0, 1.0 - config.rgb_darken)
    if config.rgb_desaturate > 0.0:
        gray = rgb.mean(axis=-1, keepdims=True)
        rgb[:] = (1.0 - config.rgb_desaturate) * rgb + config.rgb_desaturate * gray
    if config.rgb_noise_std > 0.0:
        rgb += rng.normal(0.0, config.rgb_noise_std, rgb.shape)
    if config.rgb_highlight_prob > 0.0:
        highlights = rng.random(rgb.shape[:-1]) < config.rgb_highlight_prob
        rgb[highlights] = 1.0
    degraded[..., :3] = np.clip(rgb, 0.0, 1.0)
    degraded = np.clip(degraded * scale, 0.0, scale)
    return degraded.astype(original_dtype, copy=False)


def apply_pseudo_blur(
    obs: Any,
    config: PseudoBlurConfig,
    rng: np.random.Generator | None = None,
) -> Any:
    """Apply in-place depth degradation to simulate transparent/dark-object ambiguity."""
    if not config.enabled:
        return obs
    if not isinstance(obs, dict):
        return {
            "state": obs,
            "_pseudo_blur_uncertainty": float(config.state_uncertainty),
        }

    if rng is None:
        rng = np.random.default_rng(config.seed)

    def degrade(node: Any) -> None:
        if not isinstance(node, dict):
            return
        for key, value in list(node.items()):
            arr = _as_numpy(value)
            if arr is not None and arr.size > 0 and _is_depth_like(key, arr, degrade_position=config.degrade_position):
                node[key] = _degrade_depth_like(arr, config, rng)
            elif arr is not None and arr.size > 0 and _is_rgb_like(key, arr):
                node[key] = _degrade_rgb_like(arr, config, rng)
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

    def _center_missing_ratio(self, valid: np.ndarray) -> float:
        if valid.ndim < 2:
            return float(1.0 - valid.mean()) if valid.size else 1.0
        height, width = valid.shape[-2], valid.shape[-1]
        row_margin = max(height // 4, 1)
        col_margin = max(width // 4, 1)
        center = valid[..., row_margin : height - row_margin, col_margin : width - col_margin]
        if center.size == 0:
            return float(1.0 - valid.mean()) if valid.size else 1.0
        return float(1.0 - center.mean())

    def estimate(self, obs: Any) -> dict[str, float | bool]:
        if isinstance(obs, dict) and "_pseudo_blur_uncertainty" in obs:
            uncertainty = float(np.clip(obs["_pseudo_blur_uncertainty"], 0.0, 1.0))
            return {
                "uncertainty": uncertainty,
                "depth_variance": 0.0,
                "missing_ratio": 0.0,
                "center_missing_ratio": 0.0,
                "normalized_depth_std": 0.0,
                "triggered": uncertainty >= self.threshold,
            }

        depth = find_depth_array(obs)
        if depth is None:
            return {
                "uncertainty": 0.0,
                "depth_variance": 0.0,
                "missing_ratio": 0.0,
                "center_missing_ratio": 0.0,
                "normalized_depth_std": 0.0,
                "triggered": False,
            }

        depth = self._normalize_depth_units(depth)
        valid = np.isfinite(depth) & (depth > 0)
        missing_ratio = float(1.0 - valid.mean()) if depth.size else 1.0
        center_missing_ratio = self._center_missing_ratio(valid)

        if valid.any():
            values = depth[valid]
            mean = float(np.mean(values))
            variance = float(np.var(values))
            normalized_depth_std = float(np.clip(np.sqrt(variance) / max(mean, 1e-4), 0.0, 1.0))
        else:
            variance = 0.0
            normalized_depth_std = 0.0

        dropout_score = np.clip(missing_ratio / 0.12, 0.0, 1.0)
        center_score = np.clip(center_missing_ratio / 0.18, 0.0, 1.0)
        uncertainty = float(np.clip(0.7 * dropout_score + 0.2 * center_score + 0.1 * normalized_depth_std, 0.0, 1.0))
        return {
            "uncertainty": uncertainty,
            "depth_variance": variance,
            "missing_ratio": missing_ratio,
            "center_missing_ratio": center_missing_ratio,
            "normalized_depth_std": normalized_depth_std,
            "triggered": uncertainty >= self.threshold,
        }
