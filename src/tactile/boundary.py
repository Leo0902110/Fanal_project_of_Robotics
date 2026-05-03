from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class BoundaryRefinement:
    step: int
    refined: bool
    boundary_confidence: float
    confidence_delta: float
    post_probe_uncertainty: float
    reason: str

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
        uncertainty_relief: float = 0.35,
        confidence_decay: float = 0.05,
    ):
        self.probe_confidence_gain = probe_confidence_gain
        self.contact_confidence_gain = contact_confidence_gain
        self.uncertainty_relief = uncertainty_relief
        self.confidence_decay = confidence_decay
        self.boundary_confidence = 0.0
        self.refinement_count = 0

    def update(
        self,
        *,
        step: int,
        decision: dict[str, Any],
        tactile_feature: dict[str, Any],
        visual_uncertainty: float,
    ) -> BoundaryRefinement:
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

        if contact_detected:
            confidence_gain = min(
                1.0,
                max(normalized_contact, 0.35)
                + 0.15 * pairwise_contact_used * contact_evidence
                + 0.15 * force_balance * contact_evidence,
            )
            delta = self.contact_confidence_gain * confidence_gain
            reason = "contact_boundary_observed"
        elif should_probe:
            delta = self.probe_confidence_gain * (0.75 + 0.25 * min(max(visual_uncertainty, 0.0), 1.0))
            reason = "probe_boundary_hypothesis"
        elif self.boundary_confidence > 0.0:
            delta = -min(self.boundary_confidence, self.confidence_decay)
            reason = "confidence_decay"
        else:
            delta = 0.0
            reason = "no_boundary_update"

        if delta > 0:
            self.refinement_count += 1
        self.boundary_confidence = min(1.0, self.boundary_confidence + delta)

        relief = self.uncertainty_relief * self.boundary_confidence
        post_probe_uncertainty = max(0.0, visual_uncertainty - relief)
        return BoundaryRefinement(
            step=step,
            refined=delta > 0,
            boundary_confidence=self.boundary_confidence,
            confidence_delta=delta,
            post_probe_uncertainty=post_probe_uncertainty,
            reason=reason,
        )

    def summary(self) -> dict[str, float | int]:
        return {
            "refinement_count": self.refinement_count,
            "final_boundary_confidence": self.boundary_confidence,
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
