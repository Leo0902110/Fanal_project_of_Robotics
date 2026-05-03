from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ActivePerceptionConfig:
    """Thresholds for the decision-ambiguity to tactile-probing loop."""

    enabled: bool = False
    uncertainty_threshold: float = 0.5
    contact_confidence_threshold: float = 0.2
    probe_budget: int = 2
    probe_phases: tuple[str, ...] = ("approach", "fallback")


@dataclass
class ActivePerceptionDecision:
    step: int
    phase: str
    ambiguity_score: float
    visual_uncertainty: float
    tactile_confidence: float
    state: str
    should_probe: bool
    probe_index: int
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ActivePerceptionCoordinator:
    """Decision layer between uncertain perception and tactile-style probing.

    The coordinator keeps the probing budget and exposes a compact decision
    object that policies can consume without knowing how ambiguity was scored.
    """

    def __init__(self, config: ActivePerceptionConfig | None = None):
        self.config = config or ActivePerceptionConfig()
        self.probes_requested = 0
        self.ambiguity_count = 0
        self.contact_resolved_count = 0

    def decide(
        self,
        *,
        step: int,
        phase: str,
        uncertainty: dict[str, Any],
        tactile_feature: dict[str, Any],
    ) -> ActivePerceptionDecision:
        visual_uncertainty = self._float(uncertainty.get("uncertainty", 0.0))
        contact_detected = self._float(tactile_feature.get("contact_detected", 0.0))
        contact_strength = max(
            self._float(tactile_feature.get("contact_strength", 0.0)),
            self._float(tactile_feature.get("net_force_norm", 0.0)),
        )
        pairwise_contact_used = min(max(self._float(tactile_feature.get("pairwise_contact_used", 0.0)), 0.0), 1.0)
        left_force_norm = max(self._float(tactile_feature.get("left_force_norm", 0.0)), 0.0)
        right_force_norm = max(self._float(tactile_feature.get("right_force_norm", 0.0)), 0.0)
        normalized_contact = math.tanh(max(contact_strength, 0.0))
        force_balance = self._force_balance(left_force_norm, right_force_norm)
        contact_evidence = max(contact_detected, normalized_contact)
        tactile_confidence = min(
            1.0,
            max(
                contact_detected,
                contact_evidence * (0.65 + 0.2 * pairwise_contact_used + 0.15 * force_balance),
            ),
        )
        center_missing_ratio = self._float(uncertainty.get("center_missing_ratio", uncertainty.get("missing_ratio", 0.0)))
        normalized_depth_std = self._float(uncertainty.get("normalized_depth_std", 0.0))
        visual_pressure = min(
            1.0,
            max(visual_uncertainty, 0.65 * visual_uncertainty + 0.25 * center_missing_ratio + 0.1 * normalized_depth_std),
        )
        ambiguity_score = max(0.0, visual_pressure - tactile_confidence)

        ambiguous = ambiguity_score >= self.config.uncertainty_threshold
        if ambiguous:
            self.ambiguity_count += 1

        if tactile_confidence >= self.config.contact_confidence_threshold and visual_uncertainty > 0:
            self.contact_resolved_count += 1
            return ActivePerceptionDecision(
                step=step,
                phase=phase,
                ambiguity_score=ambiguity_score,
                visual_uncertainty=visual_uncertainty,
                tactile_confidence=tactile_confidence,
                state="resolved_by_tactile",
                should_probe=False,
                probe_index=self.probes_requested,
                reason="contact_confidence_available",
            )

        if not self.config.enabled:
            return self._idle_decision(
                step, phase, ambiguity_score, visual_uncertainty, tactile_confidence, "disabled"
            )

        if not ambiguous:
            return self._idle_decision(
                step, phase, ambiguity_score, visual_uncertainty, tactile_confidence, "confident"
            )

        if phase not in self.config.probe_phases:
            return self._idle_decision(
                step, phase, ambiguity_score, visual_uncertainty, tactile_confidence, "phase_not_probeable"
            )

        if self.probes_requested >= self.config.probe_budget:
            return ActivePerceptionDecision(
                step=step,
                phase=phase,
                ambiguity_score=ambiguity_score,
                visual_uncertainty=visual_uncertainty,
                tactile_confidence=tactile_confidence,
                state="budget_exhausted",
                should_probe=False,
                probe_index=self.probes_requested,
                reason="probe_budget_exhausted",
            )

        self.probes_requested += 1
        return ActivePerceptionDecision(
            step=step,
            phase=phase,
            ambiguity_score=ambiguity_score,
            visual_uncertainty=visual_uncertainty,
            tactile_confidence=tactile_confidence,
            state="request_probe",
            should_probe=True,
            probe_index=self.probes_requested,
            reason="visual_ambiguity_without_contact",
        )

    def _idle_decision(
        self,
        step: int,
        phase: str,
        ambiguity_score: float,
        visual_uncertainty: float,
        tactile_confidence: float,
        reason: str,
    ) -> ActivePerceptionDecision:
        state = "ambiguous_hold" if reason == "phase_not_probeable" else "observe"
        return ActivePerceptionDecision(
            step=step,
            phase=phase,
            ambiguity_score=ambiguity_score,
            visual_uncertainty=visual_uncertainty,
            tactile_confidence=tactile_confidence,
            state=state,
            should_probe=False,
            probe_index=self.probes_requested,
            reason=reason,
        )

    def summary(self) -> dict[str, int]:
        return {
            "decision_ambiguity_count": self.ambiguity_count,
            "probe_request_count": self.probes_requested,
            "contact_resolved_count": self.contact_resolved_count,
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
