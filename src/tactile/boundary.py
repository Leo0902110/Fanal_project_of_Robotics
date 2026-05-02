from __future__ import annotations

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
    """Lightweight tactile boundary update for the MVP active-perception loop.

    The refiner turns a probing decision plus contact features into a compact
    confidence update. It is intentionally simple now, but gives a stable place
    for future tactile image or force-based boundary estimation.
    """

    def __init__(
        self,
        probe_confidence_gain: float = 0.35,
        contact_confidence_gain: float = 0.55,
        miss_confidence_penalty: float = 0.20,
        uncertainty_relief: float = 0.35,
        contact_shrink_gain: float = 0.45,
        empty_push_gain: float = 0.60,
    ):
        self.probe_confidence_gain = probe_confidence_gain
        self.contact_confidence_gain = contact_confidence_gain
        self.miss_confidence_penalty = miss_confidence_penalty
        self.uncertainty_relief = uncertainty_relief
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
        contact_strength = min(self._float(tactile_feature.get("contact_strength", 0.0)), 1.0)
        offset = self._vector3(probe_plan.get("boundary_offset_xyz"))
        obj_pos = self._vector3(oracle_state.get("obj_pos"))
        target_xyz = self._vector3(probe_plan.get("target_xyz"))
        normal = self._unit_vector(offset)
        offset_norm = float(np.linalg.norm(offset)) if offset is not None else 0.0
        if normal is not None:
            self.surface_normal = normal

        if contact_detected:
            delta = self.contact_confidence_gain * max(contact_strength, 0.5)
            reason = "contact_boundary_observed"
            self.contact_evidence += 1
            if obj_pos is not None and target_xyz is not None:
                penetration = target_xyz - obj_pos
                self.boundary_shift += self.contact_shrink_gain * penetration
            elif offset is not None:
                self.boundary_shift += self.contact_shrink_gain * offset
        elif should_probe:
            delta = -self.miss_confidence_penalty
            reason = "probe_found_empty_space"
            self.empty_evidence += 1
            push = self.empty_push_gain * max(offset_norm, 0.01) * self.surface_normal
            self.boundary_shift -= push
        else:
            delta = 0.0
            reason = "no_boundary_update"

        if delta != 0.0:
            self.refinement_count += 1
        if should_probe and not contact_detected:
            self.boundary_confidence = max(0.0, self.boundary_confidence + delta)
        elif should_probe:
            self.boundary_confidence = min(1.0, self.boundary_confidence + delta)
        elif delta > 0.0:
            self.boundary_confidence = min(1.0, self.boundary_confidence + delta)

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
