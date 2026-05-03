from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass
class ProbePlan:
    should_probe: bool
    probe_point: tuple[float, float]
    reason: str
    boundary_offset_xyz: tuple[float, float, float]
    target_xyz: tuple[float, float, float] | None
    uncertainty_score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TactileProbePlanner:
    """Choose a concrete tactile probe target from visual ambiguity points."""

    def __init__(self, lateral_scale: float = 0.03, vertical_offset: float = 0.015):
        self.lateral_scale = lateral_scale
        self.vertical_offset = vertical_offset

    def plan(
        self,
        *,
        decision: dict[str, Any],
        uncertainty: dict[str, Any],
        oracle_state: dict[str, Any] | None = None,
    ) -> ProbePlan:
        oracle_state = oracle_state or {}
        should_probe = bool(decision.get("should_probe", False))
        points = uncertainty.get("uncertain_points", []) if isinstance(uncertainty, dict) else []
        point = points[0] if points else None
        if not should_probe or point is None:
            return ProbePlan(
                should_probe=False,
                probe_point=(0.0, 0.0),
                reason="no_visual_boundary_point",
                boundary_offset_xyz=(0.0, 0.0, 0.0),
                target_xyz=self._tuple3(oracle_state.get("obj_pos")),
                uncertainty_score=float(decision.get("ambiguity_score", 0.0)),
            )

        obj_pos = self._array3(oracle_state.get("obj_pos"))
        offset = np.array(
            [
                self.lateral_scale * float(point.get("normalized_x", 0.0)),
                self.lateral_scale * float(point.get("normalized_y", 0.0)),
                self.vertical_offset,
            ],
            dtype=np.float32,
        )
        target = obj_pos + offset if obj_pos is not None else None
        return ProbePlan(
            should_probe=True,
            probe_point=(
                float(point.get("normalized_x", 0.0)),
                float(point.get("normalized_y", 0.0)),
            ),
            reason=str(point.get("reason", decision.get("reason", "visual_ambiguity"))),
            boundary_offset_xyz=(float(offset[0]), float(offset[1]), float(offset[2])),
            target_xyz=self._tuple3(target),
            uncertainty_score=float(point.get("score", decision.get("ambiguity_score", 0.0))),
        )

    def _array3(self, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size < 3:
            return None
        return arr[:3]

    def _tuple3(self, value: Any) -> tuple[float, float, float] | None:
        arr = self._array3(value)
        if arr is None:
            return None
        return (float(arr[0]), float(arr[1]), float(arr[2]))
