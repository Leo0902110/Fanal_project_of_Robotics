from __future__ import annotations

from dataclasses import dataclass

from utils.d_features import DFeatureBundle


@dataclass
class VisualUncertaintyConfig:
    distance_weight: float = 3.0
    alignment_weight: float = 1.5
    low_contact_bonus: float = 0.5
    grasped_discount: float = 0.8
    uncertainty_threshold: float = 0.35


def estimate_visual_uncertainty(
    bundle: DFeatureBundle,
    config: VisualUncertaintyConfig,
) -> float:
    distance_term = min(1.0, float(bundle.named["tcp_to_obj_dist"]) * config.distance_weight)
    goal_term = min(1.0, float(bundle.named["obj_to_goal_dist"]) * config.alignment_weight)
    low_contact_term = config.low_contact_bonus * (1.0 - float(bundle.named["contact_active"]))
    grasp_discount = config.grasped_discount * float(bundle.named["is_grasped"])
    uncertainty = distance_term * 0.6 + goal_term * 0.2 + low_contact_term - grasp_discount
    return max(0.0, min(1.0, uncertainty))


def should_trigger_probe_from_uncertainty(
    bundle: DFeatureBundle,
    config: VisualUncertaintyConfig,
) -> bool:
    if float(bundle.named["success"]) > 0.0:
        return False
    return estimate_visual_uncertainty(bundle, config) >= config.uncertainty_threshold