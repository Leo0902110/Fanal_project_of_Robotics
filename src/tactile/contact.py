from __future__ import annotations

from typing import Any

import numpy as np


class ContactFeatureExtractor:
    """MVP tactile placeholder based on simulator info and robot state.

    ManiSkill does not expose a real tactile image by default for PickCube-v1.
    This class keeps the project interface stable now, so a future tactile module
    can replace the internals without changing the experiment runner.
    """

    def extract(self, obs: Any, info: dict[str, Any] | None = None) -> dict[str, float]:
        info = info or {}
        success = self._to_float(info.get("success", 0.0))
        contact_hint = self._search_numeric(obs, ("force", "contact", "tcp_wrench"))
        return {
            "contact_detected": float(contact_hint > 1e-6 or success > 0.0),
            "contact_strength": contact_hint,
            "success_hint": success,
        }

    def _to_float(self, value: Any) -> float:
        try:
            arr = np.asarray(value, dtype=np.float32)
            return float(arr.mean())
        except Exception:
            return 0.0

    def _search_numeric(self, node: Any, keywords: tuple[str, ...]) -> float:
        if isinstance(node, dict):
            best = 0.0
            for key, value in node.items():
                if any(word in key.lower() for word in keywords):
                    try:
                        best = max(best, float(np.linalg.norm(np.asarray(value, dtype=np.float32))))
                    except Exception:
                        pass
                best = max(best, self._search_numeric(value, keywords))
            return best
        return 0.0
