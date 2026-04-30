from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from utils.d_features import DFeatureBundle


TensorLike = np.ndarray | torch.Tensor


def _to_float_tensor(value: TensorLike | None) -> torch.Tensor | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.float()
    return torch.tensor(np.asarray(value), dtype=torch.float32)


def _to_2d_float_tensor(value: TensorLike | None) -> torch.Tensor | None:
    tensor = _to_float_tensor(value)
    if tensor is None:
        return None
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    return tensor


def _to_flat_float_tensor(value: TensorLike | None) -> torch.Tensor | None:
    tensor = _to_float_tensor(value)
    if tensor is None:
        return None
    return tensor.reshape(-1)


@dataclass(frozen=True)
class DModuleSpec:
    vision_dim: int
    tactile_dim: int
    action_dim: int
    supports_active_probe: bool = True
    supports_training: bool = True
    accepts_external_uncertainty: bool = True


@dataclass
class DInferenceRequest:
    feature_bundle: DFeatureBundle
    vision_features: TensorLike | None = None
    visual_uncertainty: float | TensorLike | None = None
    probe_steps_used: int = 0
    step_idx: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DInferenceResult:
    action: np.ndarray
    uncertainty: float
    probe_triggered: bool
    policy_mode: str
    fused_features: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DTrainingRequest:
    vision_features: TensorLike
    tactile_features: TensorLike
    probe_flags: TensorLike
    target_actions: TensorLike | None = None
    target_probe_flags: TensorLike | None = None
    target_uncertainty: TensorLike | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def build_training_batch_tensors(request: DTrainingRequest) -> dict[str, torch.Tensor | None]:
    return {
        "vision_features": _to_2d_float_tensor(request.vision_features),
        "tactile_features": _to_2d_float_tensor(request.tactile_features),
        "probe_flags": _to_flat_float_tensor(request.probe_flags),
        "target_actions": _to_2d_float_tensor(request.target_actions),
        "target_probe_flags": _to_flat_float_tensor(request.target_probe_flags),
        "target_uncertainty": _to_flat_float_tensor(request.target_uncertainty),
    }
