from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .pseudo_blur import _as_numpy, find_depth_array


@dataclass
class AmbiguityPoint:
    pixel_y: int
    pixel_x: int
    score: float
    reason: str
    normalized_x: float
    normalized_y: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "pixel_y": self.pixel_y,
            "pixel_x": self.pixel_x,
            "score": self.score,
            "reason": self.reason,
            "normalized_x": self.normalized_x,
            "normalized_y": self.normalized_y,
        }


def find_rgb_array(obs: Any) -> np.ndarray | None:
    if isinstance(obs, dict):
        for key in ("rgb", "Color", "color"):
            arr = _as_numpy(obs.get(key))
            if arr is not None and arr.size > 0:
                return arr
        for value in obs.values():
            rgb = find_rgb_array(value)
            if rgb is not None:
                return rgb
    return None


class VisualAmbiguityDetector:
    """Structured visual ambiguity detector for MVP active grasping.

    The detector stays lightweight, but upgrades the old single uncertainty
    scalar into:
    - an uncertainty map
    - candidate uncertain boundary points
    - a dominant reason for ambiguity
    """

    def __init__(
        self,
        threshold: float = 0.18,
        dark_threshold: float = 0.18,
        edge_threshold: float = 0.05,
        top_k_points: int = 6,
    ):
        self.threshold = threshold
        self.dark_threshold = dark_threshold
        self.edge_threshold = edge_threshold
        self.top_k_points = top_k_points

    def _normalize_depth_units(self, depth: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.size and np.nanmax(depth) > 20.0:
            return depth / 1000.0
        return depth

    def _normalize_rgb(self, rgb: np.ndarray) -> np.ndarray:
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.size and np.nanmax(rgb) > 1.5:
            rgb = rgb / 255.0
        return np.clip(rgb, 0.0, 1.0)

    def _gradient_mask(self, depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
        if depth.ndim < 2:
            return np.zeros_like(depth, dtype=np.float32)
        grad_y = np.zeros_like(depth, dtype=np.float32)
        grad_x = np.zeros_like(depth, dtype=np.float32)
        grad_y[1:, :] = np.abs(depth[1:, :] - depth[:-1, :])
        grad_x[:, 1:] = np.abs(depth[:, 1:] - depth[:, :-1])
        grad = np.maximum(grad_x, grad_y)
        return ((grad > self.edge_threshold) & valid).astype(np.float32)

    def _boundary_missing_mask(self, valid: np.ndarray) -> np.ndarray:
        if valid.ndim < 2:
            return (~valid).astype(np.float32)
        invalid = ~valid
        neighbor_invalid = invalid.copy()
        neighbor_invalid[:-1, :] |= invalid[1:, :]
        neighbor_invalid[1:, :] |= invalid[:-1, :]
        neighbor_invalid[:, :-1] |= invalid[:, 1:]
        neighbor_invalid[:, 1:] |= invalid[:, :-1]
        return (valid & neighbor_invalid).astype(np.float32)

    def _candidate_points(
        self,
        uncertainty_map: np.ndarray,
        reason_map: np.ndarray,
    ) -> list[AmbiguityPoint]:
        if uncertainty_map.ndim < 2:
            return []
        flat_indices = np.argsort(uncertainty_map.reshape(-1))[::-1]
        points: list[AmbiguityPoint] = []
        height, width = uncertainty_map.shape[:2]
        for flat_index in flat_indices:
            score = float(uncertainty_map.reshape(-1)[flat_index])
            if score < self.threshold:
                break
            y, x = np.unravel_index(int(flat_index), uncertainty_map.shape)
            reason = str(reason_map[y, x])
            if points and any(abs(p.pixel_y - y) <= 4 and abs(p.pixel_x - x) <= 4 for p in points):
                continue
            points.append(
                AmbiguityPoint(
                    pixel_y=int(y),
                    pixel_x=int(x),
                    score=score,
                    reason=reason,
                    normalized_x=(float(x) / max(width - 1, 1)) * 2.0 - 1.0,
                    normalized_y=(float(y) / max(height - 1, 1)) * 2.0 - 1.0,
                )
            )
            if len(points) >= self.top_k_points:
                break
        return points

    def estimate(self, obs: Any) -> dict[str, Any]:
        if isinstance(obs, dict) and "_pseudo_blur_uncertainty" in obs:
            uncertainty = float(np.clip(obs["_pseudo_blur_uncertainty"], 0.0, 1.0))
            return {
                "uncertainty": uncertainty,
                "depth_variance": 0.0,
                "missing_ratio": 0.0,
                "triggered": uncertainty >= self.threshold,
                "ambiguity_score": uncertainty,
                "ambiguity_map": [[uncertainty]],
                "uncertain_points": [
                    {
                        "pixel_y": 0,
                        "pixel_x": 0,
                        "score": uncertainty,
                        "reason": "state_fallback",
                        "normalized_x": 0.0,
                        "normalized_y": 0.0,
                    }
                ],
                "dominant_reason": "state_fallback",
                "reason_scores": {"state_fallback": uncertainty},
                "uncertain_boundary_points": 1,
            }

        depth = find_depth_array(obs)
        if depth is None:
            return {
                "uncertainty": 0.0,
                "depth_variance": 0.0,
                "missing_ratio": 0.0,
                "triggered": False,
                "ambiguity_score": 0.0,
                "ambiguity_map": [[0.0]],
                "uncertain_points": [],
                "dominant_reason": "none",
                "reason_scores": {},
                "uncertain_boundary_points": 0,
            }

        depth = self._normalize_depth_units(depth)
        if depth.ndim == 3:
            depth = depth[..., 0]
        valid = np.isfinite(depth) & (depth > 0)
        missing_ratio = float(1.0 - valid.mean()) if depth.size else 1.0
        variance = float(np.var(depth[valid])) if valid.any() else 0.0

        rgb = find_rgb_array(obs)
        dark_mask = np.zeros_like(depth, dtype=np.float32)
        if rgb is not None:
            rgb = self._normalize_rgb(rgb)
            if rgb.ndim == 3:
                intensity = rgb[..., :3].mean(axis=-1)
                dark_mask = (intensity < self.dark_threshold).astype(np.float32)

        edge_break_mask = self._gradient_mask(depth, valid)
        boundary_missing_mask = self._boundary_missing_mask(valid)

        reason_scores = {
            "depth_missing": float(boundary_missing_mask.mean()),
            "dark_region": float(dark_mask.mean()),
            "edge_break": float(edge_break_mask.mean()),
        }
        dominant_reason = max(reason_scores, key=reason_scores.get) if any(reason_scores.values()) else "none"

        uncertainty_map = np.clip(
            0.55 * boundary_missing_mask + 0.25 * dark_mask + 0.20 * edge_break_mask,
            0.0,
            1.0,
        ).astype(np.float32)

        reason_map = np.full(depth.shape, "depth_missing", dtype=object)
        reason_map[dark_mask > boundary_missing_mask] = "dark_region"
        reason_map[edge_break_mask > np.maximum(boundary_missing_mask, dark_mask)] = "edge_break"

        points = self._candidate_points(uncertainty_map, reason_map)
        uncertainty = float(np.clip(max(float(uncertainty_map.mean()) * 3.0, missing_ratio / 0.12), 0.0, 1.0))

        return {
            "uncertainty": uncertainty,
            "depth_variance": variance,
            "missing_ratio": missing_ratio,
            "triggered": uncertainty >= self.threshold,
            "ambiguity_score": uncertainty,
            "ambiguity_map": uncertainty_map.tolist(),
            "uncertain_points": [point.to_dict() for point in points],
            "dominant_reason": dominant_reason,
            "reason_scores": reason_scores,
            "uncertain_boundary_points": len(points),
        }
