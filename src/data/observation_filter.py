from __future__ import annotations

from typing import Any


SENSOR_ROOTS = {"sensor_data", "sensor_param"}


def filter_observation_for_camera(
    obs: Any,
    *,
    camera: str | None = None,
    keep_state: bool = True,
) -> Any:
    """Keep a selected observation camera while optionally retaining proprioception.

    ManiSkill RGB-D observations may contain both a fixed base camera and a
    gripper-mounted hand camera. For camera-specific BC/DP demos, we filter the
    sensor branches to one camera so flattened observations cannot silently mix
    third-person and wrist-camera pixels.
    """

    camera = (camera or "").strip().lower()
    if not camera:
        return obs

    def visit(node: Any, path: tuple[str, ...], in_sensor_branch: bool) -> Any:
        if isinstance(node, dict):
            filtered: dict[str, Any] = {}
            for key, value in node.items():
                key_str = str(key)
                key_lower = key_str.lower()
                child_path = (*path, key_str)
                child_in_sensor = in_sensor_branch or key_lower in SENSOR_ROOTS
                if child_in_sensor and key_lower == camera:
                    filtered[key] = value
                    continue
                child = visit(value, child_path, child_in_sensor)
                if child is not None:
                    filtered[key] = child
            if filtered:
                return filtered
            return None if in_sensor_branch or not keep_state else {}

        if in_sensor_branch:
            return node if any(part.lower() == camera for part in path) else None
        return node if keep_state else None

    filtered_obs = visit(obs, tuple(), False)
    return filtered_obs if filtered_obs is not None else {}
