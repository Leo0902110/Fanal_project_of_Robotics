from __future__ import annotations

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
    ):
        self.probe_confidence_gain = probe_confidence_gain
        self.contact_confidence_gain = contact_confidence_gain
        self.uncertainty_relief = uncertainty_relief
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
        contact_strength = min(self._float(tactile_feature.get("contact_strength", 0.0)), 1.0)

        if contact_detected:
            delta = self.contact_confidence_gain * max(contact_strength, 0.5)
            reason = "contact_boundary_observed"
        elif should_probe:
            delta = self.probe_confidence_gain
            reason = "probe_boundary_hypothesis"
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
