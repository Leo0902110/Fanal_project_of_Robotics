from __future__ import annotations

from typing import Any

import numpy as np


class ContactFeatureExtractor:
    """Compatibility wrapper for tactile-style contact features.

    When the simulator environment is available, prefer D's pairwise finger-cube
    contact forces. Otherwise fall back to the previous observation scan so the
    rest of the pipeline keeps working in lightweight contexts.
    """

    def __init__(self, contact_threshold: float = 1e-6):
        self.contact_threshold = contact_threshold

    def extract(
        self,
        obs: Any,
        info: dict[str, Any] | None = None,
        *,
        env: Any | None = None,
    ) -> dict[str, float]:
        info = info or {}
        success = self._to_float(info.get("success", 0.0))
        tactile_reading = self._extract_pairwise_contact(env)
        if tactile_reading is not None:
            left_force_norm = self._to_float(tactile_reading.named.get("left_force_norm", 0.0))
            right_force_norm = self._to_float(tactile_reading.named.get("right_force_norm", 0.0))
            net_force_norm = self._to_float(tactile_reading.named.get("net_force_norm", 0.0))
            finger_links_found = self._to_float(tactile_reading.named.get("finger_links_found", 0.0))
            return {
                "contact_detected": float(net_force_norm > self.contact_threshold or success > 0.0),
                "contact_strength": net_force_norm,
                "success_hint": success,
                "left_force_norm": left_force_norm,
                "right_force_norm": right_force_norm,
                "net_force_norm": net_force_norm,
                "finger_links_found": finger_links_found,
                "pairwise_contact_used": 1.0,
            }

        contact_hint = self._search_numeric(obs, ("force", "contact", "tcp_wrench"))
        return {
            "contact_detected": float(contact_hint > self.contact_threshold or success > 0.0),
            "contact_strength": contact_hint,
            "success_hint": success,
            "left_force_norm": contact_hint,
            "right_force_norm": 0.0,
            "net_force_norm": contact_hint,
            "finger_links_found": 0.0,
            "pairwise_contact_used": 0.0,
        }

    def _extract_pairwise_contact(self, env: Any | None) -> Any | None:
        if env is None:
            return None
        try:
            from utils.d_features import extract_contact_reading
        except Exception:
            return None

        try:
            return extract_contact_reading(env)
        except Exception:
            return None

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
