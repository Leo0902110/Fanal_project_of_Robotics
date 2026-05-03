from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class BoundaryRefinement:
    step: int
    refined: bool
    boundary_confidence: float
    confidence_delta: float
    post_probe_uncertainty: float
    reason: str
    contact_evidence: int
    empty_evidence: int
    refined_grasp_target: tuple[float, float, float] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TactileBoundaryRefiner:
    """Lightweight tactile boundary update for the MVP active-perception loop."""

    def __init__(
        self,
        probe_confidence_gain: float = 0.35,
        contact_confidence_gain: float = 0.55,
        miss_confidence_penalty: float = 0.20,
        uncertainty_relief: float = 0.35,
        confidence_decay: float = 0.05,
        contact_shrink_gain: float = 0.45,
        empty_push_gain: float = 0.60,
    ):
        self.probe_confidence_gain = probe_confidence_gain
        self.contact_confidence_gain = contact_confidence_gain
        self.miss_confidence_penalty = miss_confidence_penalty
        self.uncertainty_relief = uncertainty_relief
        self.confidence_decay = confidence_decay
        self.contact_shrink_gain = contact_shrink_gain
        self.empty_push_gain = empty_push_gain
        self.boundary_confidence = 0.0
        self.refinement_count = 0
        self.contact_evidence = 0
        self.empty_evidence = 0
        self.boundary_shift = np.zeros(3, dtype=np.float32)
        self.surface_normal = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def update(
        self,
        *,
        step: int,
        decision: dict[str, Any],
        tactile_feature: dict[str, Any],
        visual_uncertainty: float,
        probe_plan: dict[str, Any] | None = None,
        oracle_state: dict[str, Any] | None = None,
    ) -> BoundaryRefinement:
        oracle_state = oracle_state or {}
        probe_plan = probe_plan or {}
        should_probe = bool(decision.get("should_probe", False))
        contact_detected = self._float(tactile_feature.get("contact_detected", 0.0)) > 0.0
        contact_strength = max(
            self._float(tactile_feature.get("contact_strength", 0.0)),
            self._float(tactile_feature.get("net_force_norm", 0.0)),
        )
        pairwise_contact_used = min(max(self._float(tactile_feature.get("pairwise_contact_used", 0.0)), 0.0), 1.0)
        left_force_norm = max(self._float(tactile_feature.get("left_force_norm", 0.0)), 0.0)
        right_force_norm = max(self._float(tactile_feature.get("right_force_norm", 0.0)), 0.0)
        normalized_contact = math.tanh(max(contact_strength, 0.0))
        force_balance = self._force_balance(left_force_norm, right_force_norm)
        contact_evidence = max(float(contact_detected), normalized_contact)

        offset = self._vector3(probe_plan.get("boundary_offset_xyz"))
        obj_pos = self._vector3(oracle_state.get("obj_pos"))
        target_xyz = self._vector3(probe_plan.get("target_xyz"))
        normal = self._unit_vector(offset)
        offset_norm = float(np.linalg.norm(offset)) if offset is not None else 0.0
        if normal is not None:
            self.surface_normal = normal

        if contact_detected:
            confidence_gain = min(
                1.0,
                max(normalized_contact, 0.35)
                + 0.15 * pairwise_contact_used * contact_evidence
                + 0.15 * force_balance * contact_evidence,
            )
            delta = self.contact_confidence_gain * confidence_gain
            reason = "contact_boundary_observed"
            self.contact_evidence += 1
            if obj_pos is not None and target_xyz is not None:
                penetration = target_xyz - obj_pos
                self.boundary_shift += self.contact_shrink_gain * penetration
            elif offset is not None:
                self.boundary_shift += self.contact_shrink_gain * offset
        elif should_probe:
            hypothesis_gain = self.probe_confidence_gain * (0.75 + 0.25 * min(max(visual_uncertainty, 0.0), 1.0))
            delta = hypothesis_gain - self.miss_confidence_penalty
            reason = "probe_found_empty_space" if delta < 0.0 else "probe_boundary_hypothesis"
            self.empty_evidence += 1
            push = self.empty_push_gain * max(offset_norm, 0.01) * self.surface_normal
            self.boundary_shift -= push
        elif self.boundary_confidence > 0.0:
            delta = -min(self.boundary_confidence, self.confidence_decay)
            reason = "confidence_decay"
        else:
            delta = 0.0
            reason = "no_boundary_update"

        if delta != 0.0:
            self.refinement_count += 1
        self.boundary_confidence = float(np.clip(self.boundary_confidence + delta, 0.0, 1.0))

        relief = self.uncertainty_relief * self.boundary_confidence
        post_probe_uncertainty = max(0.0, visual_uncertainty - relief)
        refined_target = None
        if obj_pos is not None:
            refined = obj_pos + self.boundary_shift
            refined_target = (float(refined[0]), float(refined[1]), float(refined[2]))
        return BoundaryRefinement(
            step=step,
            refined=delta != 0.0,
            boundary_confidence=self.boundary_confidence,
            confidence_delta=delta,
            post_probe_uncertainty=post_probe_uncertainty,
            reason=reason,
            contact_evidence=self.contact_evidence,
            empty_evidence=self.empty_evidence,
            refined_grasp_target=refined_target,
        )

    def summary(self) -> dict[str, float | int]:
        return {
            "refinement_count": self.refinement_count,
            "final_boundary_confidence": self.boundary_confidence,
            "contact_evidence": self.contact_evidence,
            "empty_evidence": self.empty_evidence,
        }

    def _float(self, value: Any) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    def _force_balance(self, left_force_norm: float, right_force_norm: float) -> float:
        total = left_force_norm + right_force_norm
        if total <= 1e-6:
            return 0.0
        imbalance = abs(left_force_norm - right_force_norm) / total
        return max(0.0, 1.0 - min(imbalance, 1.0))

    def _vector3(self, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            return None
        return arr[:3]

    def _unit_vector(self, value: Any) -> np.ndarray | None:
        arr = self._vector3(value)
        if arr is None:
            return None
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-6:
            return None
        return arr / norm
