from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.tactile_fusion import VisionTactileFusionMLP
from utils.d_active_probe import ActiveProbeConfig, build_joint_probe_action
from utils.d_features import DFeatureBundle
from utils.d_interface import (
    DInferenceRequest,
    DInferenceResult,
    DModuleSpec,
    DTrainingRequest,
    build_training_batch_tensors,
)
from utils.d_uncertainty import VisualUncertaintyConfig, estimate_visual_uncertainty


@dataclass
class DPolicyOutput:
    action: np.ndarray
    fused_features: torch.Tensor
    uncertainty: float
    probe_triggered: bool
    policy_mode: str


@dataclass
class DPolicyBatch:
    vision_features: torch.Tensor
    tactile_features: torch.Tensor
    probe_flags: torch.Tensor
    target_actions: torch.Tensor | None = None
    target_probe_flags: torch.Tensor | None = None
    target_uncertainty: torch.Tensor | None = None
    target_phase_labels: torch.Tensor | None = None


@dataclass
class DPolicyTrainOutput:
    actions: torch.Tensor
    probe_logits: torch.Tensor
    uncertainty: torch.Tensor
    phase_logits: torch.Tensor | None
    fused_features: torch.Tensor


@dataclass
class DPolicyLosses:
    total_loss: torch.Tensor
    action_loss: torch.Tensor
    probe_loss: torch.Tensor
    uncertainty_loss: torch.Tensor
    phase_loss: torch.Tensor


class ActiveTactilePolicy(nn.Module):
    def __init__(
        self,
        action_dim: int,
        vision_dim: int,
        tactile_dim: int,
        hidden_dim: int = 128,
        num_phase_classes: int = 0,
        probe_config: ActiveProbeConfig | None = None,
        uncertainty_config: VisualUncertaintyConfig | None = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.tactile_dim = tactile_dim
        self.hidden_dim = hidden_dim
        self.num_phase_classes = num_phase_classes
        self.probe_config = probe_config or ActiveProbeConfig()
        self.uncertainty_config = uncertainty_config or VisualUncertaintyConfig()
        self.fusion = VisionTactileFusionMLP(
            vision_dim=vision_dim,
            tactile_dim=tactile_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
        )
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )
        self.probe_head = nn.Linear(hidden_dim, 1)
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        self.phase_head = nn.Linear(hidden_dim, num_phase_classes) if num_phase_classes > 0 else None

    @property
    def vision_dim(self) -> int:
        first_linear = self.fusion.fusion[0]
        fusion_input_dim = first_linear.in_features
        tactile_latent_dim = self.fusion.tactile_encoder.network[-1].out_features
        return fusion_input_dim - tactile_latent_dim - 1

    def _ensure_2d(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 1:
            return tensor.unsqueeze(0)
        return tensor

    def _resolve_uncertainty(
        self,
        bundle: DFeatureBundle,
        uncertainty_override: float | torch.Tensor | np.ndarray | None = None,
    ) -> float:
        if uncertainty_override is None:
            return estimate_visual_uncertainty(bundle, self.uncertainty_config)
        if isinstance(uncertainty_override, torch.Tensor):
            uncertainty_override = uncertainty_override.detach().cpu().numpy()
        if isinstance(uncertainty_override, np.ndarray):
            uncertainty_override = float(uncertainty_override.reshape(-1)[0])
        return float(uncertainty_override)

    @property
    def interface_spec(self) -> DModuleSpec:
        return DModuleSpec(
            vision_dim=self.vision_dim,
            tactile_dim=self.tactile_dim,
            action_dim=self.action_dim,
        )

    def make_batch(
        self,
        vision_features: torch.Tensor,
        tactile_features: torch.Tensor,
        probe_flags: torch.Tensor,
        target_actions: torch.Tensor | None = None,
        target_probe_flags: torch.Tensor | None = None,
        target_uncertainty: torch.Tensor | None = None,
        target_phase_labels: torch.Tensor | None = None,
    ) -> DPolicyBatch:
        return DPolicyBatch(
            vision_features=self._ensure_2d(vision_features.float()),
            tactile_features=self._ensure_2d(tactile_features.float()),
            probe_flags=probe_flags.float(),
            target_actions=None if target_actions is None else self._ensure_2d(target_actions.float()),
            target_probe_flags=target_probe_flags,
            target_uncertainty=target_uncertainty,
            target_phase_labels=target_phase_labels,
        )

    def make_batch_from_request(self, request: DTrainingRequest) -> DPolicyBatch:
        batch_tensors = build_training_batch_tensors(request)
        return self.make_batch(**batch_tensors)

    def forward(
        self,
        vision_features: torch.Tensor,
        tactile_features: torch.Tensor,
        probe_flag: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused = self.fusion(vision_features, tactile_features, probe_flag)
        action = self.action_head(fused)
        return action, fused

    def forward_batch(self, batch: DPolicyBatch) -> DPolicyTrainOutput:
        actions, fused = self.forward(
            batch.vision_features,
            batch.tactile_features,
            batch.probe_flags,
        )
        probe_logits = self.probe_head(fused).squeeze(-1)
        uncertainty = self.uncertainty_head(fused).squeeze(-1)
        phase_logits = self.phase_head(fused) if self.phase_head is not None else None
        return DPolicyTrainOutput(
            actions=actions,
            probe_logits=probe_logits,
            uncertainty=uncertainty,
            phase_logits=phase_logits,
            fused_features=fused,
        )

    def compute_loss(
        self,
        batch: DPolicyBatch,
        action_weight: float = 1.0,
        probe_weight: float = 0.2,
        uncertainty_weight: float = 0.2,
        phase_weight: float = 0.5,
    ) -> DPolicyLosses:
        outputs = self.forward_batch(batch)

        zero = outputs.actions.new_tensor(0.0)
        action_loss = zero
        probe_loss = zero
        uncertainty_loss = zero
        phase_loss = zero

        if batch.target_actions is not None:
            action_loss = F.mse_loss(outputs.actions, batch.target_actions)
        if batch.target_probe_flags is not None:
            target_probe_flags = batch.target_probe_flags.float().reshape(-1)
            probe_loss = F.binary_cross_entropy_with_logits(
                outputs.probe_logits,
                target_probe_flags,
            )
        if batch.target_uncertainty is not None:
            target_uncertainty = batch.target_uncertainty.float().reshape(-1)
            uncertainty_loss = F.mse_loss(outputs.uncertainty, target_uncertainty)
        if batch.target_phase_labels is not None and outputs.phase_logits is not None:
            target_phase_labels = batch.target_phase_labels.long().reshape(-1)
            phase_loss = F.cross_entropy(outputs.phase_logits, target_phase_labels)

        total_loss = (
            action_weight * action_loss
            + probe_weight * probe_loss
            + uncertainty_weight * uncertainty_loss
            + phase_weight * phase_loss
        )
        return DPolicyLosses(
            total_loss=total_loss,
            action_loss=action_loss,
            probe_loss=probe_loss,
            uncertainty_loss=uncertainty_loss,
            phase_loss=phase_loss,
        )

    def train_step(
        self,
        batch: DPolicyBatch,
        optimizer: torch.optim.Optimizer,
        action_weight: float = 1.0,
        probe_weight: float = 0.2,
        uncertainty_weight: float = 0.2,
        phase_weight: float = 0.5,
    ) -> dict[str, float]:
        self.train()
        optimizer.zero_grad(set_to_none=True)
        losses = self.compute_loss(
            batch,
            action_weight=action_weight,
            probe_weight=probe_weight,
            uncertainty_weight=uncertainty_weight,
            phase_weight=phase_weight,
        )
        losses.total_loss.backward()
        optimizer.step()
        return {
            "total_loss": float(losses.total_loss.detach().cpu().item()),
            "action_loss": float(losses.action_loss.detach().cpu().item()),
            "probe_loss": float(losses.probe_loss.detach().cpu().item()),
            "uncertainty_loss": float(losses.uncertainty_loss.detach().cpu().item()),
            "phase_loss": float(losses.phase_loss.detach().cpu().item()),
        }

    def act(
        self,
        bundle: DFeatureBundle,
        probe_steps_used: int,
        step_idx: int,
        vision_features: torch.Tensor | None = None,
        uncertainty_override: float | torch.Tensor | np.ndarray | None = None,
    ) -> DPolicyOutput:
        if vision_features is None:
            vision_features = torch.zeros((1, self.vision_dim), dtype=torch.float32)

        tactile_tensor = torch.from_numpy(bundle.feature_vector).float().unsqueeze(0)
        uncertainty = self._resolve_uncertainty(bundle, uncertainty_override)
        probe_allowed = probe_steps_used < self.probe_config.max_probe_steps
        probe_triggered = (
            uncertainty >= self.uncertainty_config.uncertainty_threshold
            and float(bundle.named["is_grasped"]) < 0.5
            and float(bundle.named["success"]) < 0.5
            and probe_allowed
        )
        probe_flag = torch.tensor([1.0 if probe_triggered else 0.0], dtype=torch.float32)
        learned_action, fused = self.forward(vision_features, tactile_tensor, probe_flag)

        if probe_triggered:
            action = build_joint_probe_action(self.action_dim, step_idx, self.probe_config)
            policy_mode = "probe"
        else:
            action = learned_action.detach().cpu().numpy()[0].astype(np.float32, copy=False)
            policy_mode = "fused_policy"

        return DPolicyOutput(
            action=action,
            fused_features=fused,
            uncertainty=float(uncertainty),
            probe_triggered=probe_triggered,
            policy_mode=policy_mode,
        )

    def act_from_request(self, request: DInferenceRequest) -> DInferenceResult:
        output = self.act(
            bundle=request.feature_bundle,
            probe_steps_used=request.probe_steps_used,
            step_idx=request.step_idx,
            vision_features=request.vision_features,
            uncertainty_override=request.visual_uncertainty,
        )
        return DInferenceResult(
            action=output.action,
            fused_features=output.fused_features,
            uncertainty=output.uncertainty,
            probe_triggered=output.probe_triggered,
            policy_mode=output.policy_mode,
            metadata=dict(request.metadata),
        )