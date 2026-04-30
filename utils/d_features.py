from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _squeeze_batch(value: Any) -> np.ndarray:
    array = _to_numpy(value)
    if array.ndim > 1 and array.shape[0] == 1:
        return array[0]
    return array


def _as_float(value: Any) -> float:
    array = _squeeze_batch(value)
    return float(np.asarray(array).reshape(-1)[0])


@dataclass
class DFeatureBundle:
    feature_vector: np.ndarray
    named: dict[str, Any]


@dataclass
class TactileContactReading:
    feature_vector: np.ndarray
    named: dict[str, Any]


def _empty_tactile_reading(finger_links_found: float = 0.0) -> TactileContactReading:
    zeros = np.zeros(3, dtype=np.float32)
    feature_vector = np.concatenate(
        [
            zeros,
            zeros,
            np.asarray(
                [0.0, 0.0, 0.0, 0.0, finger_links_found],
                dtype=np.float32,
            ),
        ]
    )
    return TactileContactReading(
        feature_vector=feature_vector,
        named={
            "left_force": zeros,
            "right_force": zeros,
            "net_force": zeros,
            "left_force_norm": 0.0,
            "right_force_norm": 0.0,
            "net_force_norm": 0.0,
            "contact_active": 0.0,
            "finger_links_found": finger_links_found,
        },
    )


def _find_first_link_name(link_names: list[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in link_names:
            return candidate
    return None


def extract_contact_reading(env: Any) -> TactileContactReading:
    unwrapped = env.unwrapped
    robot = unwrapped.agent.robot
    scene = unwrapped.scene
    cube = unwrapped.cube

    link_names = [link.name for link in robot.links]
    left_name = _find_first_link_name(
        link_names,
        [
            "panda_leftfinger_pad",
            "panda_leftfinger",
            "left_finger_pad",
            "left_finger",
        ],
    )
    right_name = _find_first_link_name(
        link_names,
        [
            "panda_rightfinger_pad",
            "panda_rightfinger",
            "right_finger_pad",
            "right_finger",
        ],
    )

    if left_name is None or right_name is None:
        return _empty_tactile_reading(finger_links_found=0.0)

    left_link = robot.links_map[left_name]
    right_link = robot.links_map[right_name]

    left_force = _squeeze_batch(scene.get_pairwise_contact_forces(left_link, cube)).astype(
        np.float32, copy=False
    )
    right_force = _squeeze_batch(
        scene.get_pairwise_contact_forces(right_link, cube)
    ).astype(np.float32, copy=False)
    net_force = left_force + right_force

    left_force_norm = float(np.linalg.norm(left_force))
    right_force_norm = float(np.linalg.norm(right_force))
    net_force_norm = float(np.linalg.norm(net_force))
    contact_active = float(net_force_norm > 1e-6)

    feature_vector = np.concatenate(
        [
            left_force.astype(np.float32, copy=False),
            right_force.astype(np.float32, copy=False),
            np.asarray(
                [
                    left_force_norm,
                    right_force_norm,
                    net_force_norm,
                    contact_active,
                    1.0,
                ],
                dtype=np.float32,
            ),
        ]
    )

    return TactileContactReading(
        feature_vector=feature_vector,
        named={
            "left_force": left_force,
            "right_force": right_force,
            "net_force": net_force,
            "left_force_norm": left_force_norm,
            "right_force_norm": right_force_norm,
            "net_force_norm": net_force_norm,
            "contact_active": contact_active,
            "finger_links_found": 1.0,
        },
    )


def extract_d_features(
    obs: Mapping[str, Any],
    info: Mapping[str, Any],
    tactile_reading: TactileContactReading | None = None,
) -> DFeatureBundle:
    agent_obs = obs.get("agent", {})
    extra_obs = obs.get("extra", {})

    qpos = _squeeze_batch(agent_obs.get("qpos", np.zeros(0, dtype=np.float32))).astype(
        np.float32, copy=False
    )
    qvel = _squeeze_batch(agent_obs.get("qvel", np.zeros(0, dtype=np.float32))).astype(
        np.float32, copy=False
    )
    tcp_pose = _squeeze_batch(extra_obs.get("tcp_pose", np.zeros(7, dtype=np.float32))).astype(
        np.float32, copy=False
    )
    obj_pose = _squeeze_batch(extra_obs.get("obj_pose", np.zeros(7, dtype=np.float32))).astype(
        np.float32, copy=False
    )
    goal_pos = _squeeze_batch(extra_obs.get("goal_pos", np.zeros(3, dtype=np.float32))).astype(
        np.float32, copy=False
    )
    tcp_to_obj_pos = _squeeze_batch(
        extra_obs.get("tcp_to_obj_pos", np.zeros(3, dtype=np.float32))
    ).astype(np.float32, copy=False)
    obj_to_goal_pos = _squeeze_batch(
        extra_obs.get("obj_to_goal_pos", np.zeros(3, dtype=np.float32))
    ).astype(np.float32, copy=False)

    is_grasped = _as_float(extra_obs.get("is_grasped", info.get("is_grasped", 0.0)))
    success = _as_float(info.get("success", 0.0))
    is_robot_static = _as_float(info.get("is_robot_static", 0.0))

    if tactile_reading is None:
        tactile_reading = _empty_tactile_reading(finger_links_found=0.0)

    tcp_to_obj_dist = float(np.linalg.norm(tcp_to_obj_pos))
    obj_to_goal_dist = float(np.linalg.norm(obj_to_goal_pos))
    qvel_norm = float(np.linalg.norm(qvel))

    tactile_surrogate = np.asarray(
        [
            is_grasped,
            tcp_to_obj_dist,
            obj_to_goal_dist,
            qvel_norm,
            success,
            is_robot_static,
            float(tactile_reading.named["left_force_norm"]),
            float(tactile_reading.named["right_force_norm"]),
            float(tactile_reading.named["net_force_norm"]),
            float(tactile_reading.named["contact_active"]),
        ],
        dtype=np.float32,
    )

    feature_vector = np.concatenate(
        [
            tactile_surrogate,
            tactile_reading.feature_vector.astype(np.float32, copy=False),
            tcp_to_obj_pos.astype(np.float32, copy=False),
            obj_to_goal_pos.astype(np.float32, copy=False),
        ]
    )

    return DFeatureBundle(
        feature_vector=feature_vector,
        named={
            "qpos": qpos,
            "qvel": qvel,
            "tcp_pose": tcp_pose,
            "obj_pose": obj_pose,
            "goal_pos": goal_pos,
            "tcp_to_obj_pos": tcp_to_obj_pos,
            "obj_to_goal_pos": obj_to_goal_pos,
            "is_grasped": is_grasped,
            "success": success,
            "is_robot_static": is_robot_static,
            "tcp_to_obj_dist": tcp_to_obj_dist,
            "obj_to_goal_dist": obj_to_goal_dist,
            "qvel_norm": qvel_norm,
            "tactile_surrogate": tactile_surrogate,
            **tactile_reading.named,
        },
    )


def summarize_feature_bundle(bundle: DFeatureBundle) -> dict[str, Any]:
    return {
        "feature_dim": int(bundle.feature_vector.shape[0]),
        "is_grasped": float(bundle.named["is_grasped"]),
        "success": float(bundle.named["success"]),
        "tcp_to_obj_dist": round(float(bundle.named["tcp_to_obj_dist"]), 4),
        "obj_to_goal_dist": round(float(bundle.named["obj_to_goal_dist"]), 4),
        "qvel_norm": round(float(bundle.named["qvel_norm"]), 4),
        "net_force_norm": round(float(bundle.named["net_force_norm"]), 4),
        "contact_active": float(bundle.named["contact_active"]),
    }