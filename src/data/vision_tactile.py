from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


TACTILE_FEATURE_NAMES: tuple[str, ...] = (
    "visual_uncertainty",
    "boundary_confidence",
    "contact_detected",
    "contact_strength",
    "left_force_norm",
    "right_force_norm",
    "net_force_norm",
    "pairwise_contact_used",
    "post_probe_uncertainty",
    "probe_state",
)


def _to_numpy(value: Any) -> np.ndarray | None:
    if isinstance(value, np.ndarray):
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
    except Exception:
        pass
    return None


def _normalize_rgb(arr: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(arr)
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[-1] not in (3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        return None
    arr = arr[..., :3]
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    arr = arr.astype(np.float32)
    if arr.size and np.nanmax(arr) > 1.5:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _normalize_depth(arr: np.ndarray) -> np.ndarray | None:
    arr = np.asarray(arr)
    while arr.ndim >= 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        return None
    depth = arr.astype(np.float32)
    if depth.size and np.nanmax(depth) > 20.0:
        depth = depth / 1000.0
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(depth, 0.0, 2.0) / 2.0


def camera_rgbd_tensor(obs: Any, camera: str = "hand_camera") -> np.ndarray:
    """Extract selected camera RGB-D as CHW float32 in [0, 1]."""
    if not isinstance(obs, dict):
        raise ValueError("Expected ManiSkill dict observation.")
    sensor_data = obs.get("sensor_data", {})
    if not isinstance(sensor_data, dict):
        raise ValueError("Observation has no sensor_data dict.")
    matches = [(key, value) for key, value in sensor_data.items() if camera.lower() in str(key).lower()]
    if not matches:
        available = ", ".join(str(key) for key in sensor_data)
        raise ValueError(f"Camera {camera!r} not found in sensor_data. Available: {available}")
    _, node = matches[0]
    if not isinstance(node, dict):
        raise ValueError(f"Camera node {camera!r} is not a dict.")
    rgb = _normalize_rgb(_to_numpy(node.get("rgb")))
    depth = _normalize_depth(_to_numpy(node.get("depth")))
    if rgb is None or depth is None:
        raise ValueError(f"Camera {camera!r} must provide rgb and depth arrays.")
    if rgb.shape[:2] != depth.shape[:2]:
        raise ValueError(f"RGB/depth shape mismatch: {rgb.shape} vs {depth.shape}")
    rgb_chw = np.moveaxis(rgb, -1, 0)
    depth_chw = depth[None, ...]
    return np.concatenate([rgb_chw, depth_chw], axis=0).astype(np.float32, copy=False)


def tactile_vector(
    *,
    visual_uncertainty: float,
    boundary_confidence: float,
    tactile_feature: dict[str, Any],
    post_probe_uncertainty: float,
    probe_state: float,
) -> np.ndarray:
    values = {
        "visual_uncertainty": visual_uncertainty,
        "boundary_confidence": boundary_confidence,
        "post_probe_uncertainty": post_probe_uncertainty,
        "probe_state": probe_state,
    }
    for key in (
        "contact_detected",
        "contact_strength",
        "left_force_norm",
        "right_force_norm",
        "net_force_norm",
        "pairwise_contact_used",
    ):
        values[key] = tactile_feature.get(key, 0.0)
    return np.asarray([float(values[name]) for name in TACTILE_FEATURE_NAMES], dtype=np.float32)


def load_vision_tactile_npz(root: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    root = Path(root)
    paths = sorted(root.glob("episode_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No episode_*.npz files found under {root}")
    images, tactile, actions, metadata = [], [], [], []
    image_shape = None
    action_dim = None
    for path in paths:
        data = np.load(path, allow_pickle=True)
        episode_images = data["images"].astype(np.float32)
        episode_tactile = data["tactile"].astype(np.float32)
        episode_actions = data["actions"].astype(np.float32)
        if image_shape is None:
            image_shape = episode_images.shape[1:]
            action_dim = episode_actions.shape[1]
        if episode_images.shape[1:] != image_shape or episode_actions.shape[1] != action_dim:
            raise ValueError(f"Episode shape mismatch in {path}")
        images.append(episode_images)
        tactile.append(episode_tactile)
        actions.append(episode_actions)
        metadata.append(data["metadata"].item() if "metadata" in data else {"path": str(path)})
    return np.concatenate(images), np.concatenate(tactile), np.concatenate(actions), metadata
