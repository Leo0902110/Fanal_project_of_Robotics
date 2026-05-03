from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


LEGACY_BC_FEATURE_NAMES: tuple[str, ...] = (
    "flattened_observation",
    "uncertainty",
    "boundary_confidence",
)

DEFAULT_BC_FEATURE_NAMES: tuple[str, ...] = LEGACY_BC_FEATURE_NAMES + (
    "contact_detected",
    "contact_strength",
    "left_force_norm",
    "right_force_norm",
    "net_force_norm",
    "pairwise_contact_used",
    "post_probe_uncertainty",
)

ORACLE_GEOMETRY_FEATURE_NAMES: tuple[str, ...] = (
    "tcp_to_obj_x",
    "tcp_to_obj_y",
    "tcp_to_obj_z",
    "obj_to_goal_x",
    "obj_to_goal_y",
    "obj_to_goal_z",
)

REMOTE_POLICY_FEATURE_NAMES: tuple[str, ...] = (
    "flattened_observation",
    "uncertainty",
    "boundary_confidence",
    "probe_state",
    "dominant_reason_one_hot",
    "probe_point_xy",
    "refined_grasp_target_xyz",
)

ORACLE_BC_FEATURE_NAMES: tuple[str, ...] = DEFAULT_BC_FEATURE_NAMES + ORACLE_GEOMETRY_FEATURE_NAMES


def flatten_observation(obs: Any) -> np.ndarray:
    """Convert nested simulator observations into a 1D float feature vector."""
    values: list[np.ndarray] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key in sorted(node):
                if key.startswith("_"):
                    continue
                visit(node[key])
            return
        try:
            arr = np.asarray(node, dtype=np.float32)
        except Exception:
            return
        if arr.size:
            values.append(arr.reshape(-1))

    visit(obs)
    if not values:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(values).astype(np.float32, copy=False)


REASON_VOCAB = ("none", "depth_missing", "dark_region", "edge_break", "state_fallback")


def reason_to_one_hot(reason: str) -> np.ndarray:
    vector = np.zeros(len(REASON_VOCAB), dtype=np.float32)
    try:
        index = REASON_VOCAB.index(str(reason))
    except ValueError:
        index = 0
    vector[index] = 1.0
    return vector


def build_policy_features(
    observation: np.ndarray,
    *,
    uncertainty: float,
    boundary_confidence: float,
    dominant_reason: str,
    probe_state: float,
    probe_point: np.ndarray | None,
    refined_grasp_target: np.ndarray | None,
) -> np.ndarray:
    probe_point_arr = np.asarray(probe_point if probe_point is not None else np.zeros(2), dtype=np.float32).reshape(-1)
    if probe_point_arr.size < 2:
        probe_point_arr = np.pad(probe_point_arr, (0, 2 - probe_point_arr.size))
    refined_arr = np.asarray(
        refined_grasp_target if refined_grasp_target is not None else np.zeros(3),
        dtype=np.float32,
    ).reshape(-1)
    if refined_arr.size < 3:
        refined_arr = np.pad(refined_arr, (0, 3 - refined_arr.size))

    extra = np.concatenate(
        [
            np.asarray([uncertainty, boundary_confidence, probe_state], dtype=np.float32),
            reason_to_one_hot(dominant_reason),
            probe_point_arr[:2].astype(np.float32),
            refined_arr[:3].astype(np.float32),
        ]
    )
    return np.concatenate([np.asarray(observation, dtype=np.float32), extra]).astype(np.float32, copy=False)


@dataclass
class DemoEpisode:
    path: Path
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    uncertainty: np.ndarray
    boundary_confidence: np.ndarray
    contact_detected: np.ndarray
    contact_strength: np.ndarray
    left_force_norm: np.ndarray
    right_force_norm: np.ndarray
    net_force_norm: np.ndarray
    pairwise_contact_used: np.ndarray
    post_probe_uncertainty: np.ndarray
    tcp_to_obj_x: np.ndarray
    tcp_to_obj_y: np.ndarray
    tcp_to_obj_z: np.ndarray
    obj_to_goal_x: np.ndarray
    obj_to_goal_y: np.ndarray
    obj_to_goal_z: np.ndarray
    dominant_reason: np.ndarray
    probe_state: np.ndarray
    probe_point: np.ndarray
    refined_grasp_target: np.ndarray
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


class DemoDataset:
    """Simple NPZ-backed dataset for BC/DP training experiments."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.paths = sorted(self.root.glob("*.npz"))
        if not self.paths:
            raise FileNotFoundError(f"No .npz demo files found under {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[DemoEpisode]:
        for path in self.paths:
            yield self.load_episode(path)

    def __getitem__(self, index: int) -> DemoEpisode:
        return self.load_episode(self.paths[index])

    def iter_transition_batches(self) -> Iterator[dict[str, np.ndarray]]:
        for episode in self:
            yield {
                "observations": episode.observations,
                "actions": episode.actions,
                "rewards": episode.rewards,
                "dones": episode.dones,
                "uncertainty": episode.uncertainty,
                "boundary_confidence": episode.boundary_confidence,
                "contact_detected": episode.contact_detected,
                "contact_strength": episode.contact_strength,
                "left_force_norm": episode.left_force_norm,
                "right_force_norm": episode.right_force_norm,
                "net_force_norm": episode.net_force_norm,
                "pairwise_contact_used": episode.pairwise_contact_used,
                "post_probe_uncertainty": episode.post_probe_uncertainty,
                "tcp_to_obj_x": episode.tcp_to_obj_x,
                "tcp_to_obj_y": episode.tcp_to_obj_y,
                "tcp_to_obj_z": episode.tcp_to_obj_z,
                "obj_to_goal_x": episode.obj_to_goal_x,
                "obj_to_goal_y": episode.obj_to_goal_y,
                "obj_to_goal_z": episode.obj_to_goal_z,
                "dominant_reason": episode.dominant_reason,
                "probe_state": episode.probe_state,
                "probe_point": episode.probe_point,
                "refined_grasp_target": episode.refined_grasp_target,
            }

    def load_episode(self, path: str | Path) -> DemoEpisode:
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        metadata = data["metadata"].item() if "metadata" in data else {}
        actions = data["actions"].astype(np.float32)
        transition_count = int(actions.shape[0])
        return DemoEpisode(
            path=path,
            observations=data["observations"].astype(np.float32),
            actions=actions,
            rewards=data["rewards"].astype(np.float32),
            dones=data["dones"].astype(bool),
            uncertainty=data["uncertainty"].astype(np.float32),
            boundary_confidence=data["boundary_confidence"].astype(np.float32),
            contact_detected=self._load_float_array(data, "contact_detected", transition_count),
            contact_strength=self._load_float_array(data, "contact_strength", transition_count),
            left_force_norm=self._load_float_array(data, "left_force_norm", transition_count),
            right_force_norm=self._load_float_array(data, "right_force_norm", transition_count),
            net_force_norm=self._load_float_array(data, "net_force_norm", transition_count),
            pairwise_contact_used=self._load_float_array(data, "pairwise_contact_used", transition_count),
            post_probe_uncertainty=self._load_float_array(data, "post_probe_uncertainty", transition_count),
            tcp_to_obj_x=self._load_scalar_feature(data, "tcp_to_obj_x", "tcp_to_obj_pos", 0, transition_count),
            tcp_to_obj_y=self._load_scalar_feature(data, "tcp_to_obj_y", "tcp_to_obj_pos", 1, transition_count),
            tcp_to_obj_z=self._load_scalar_feature(data, "tcp_to_obj_z", "tcp_to_obj_pos", 2, transition_count),
            obj_to_goal_x=self._load_scalar_feature(data, "obj_to_goal_x", "obj_to_goal_pos", 0, transition_count),
            obj_to_goal_y=self._load_scalar_feature(data, "obj_to_goal_y", "obj_to_goal_pos", 1, transition_count),
            obj_to_goal_z=self._load_scalar_feature(data, "obj_to_goal_z", "obj_to_goal_pos", 2, transition_count),
            dominant_reason=(
                data["dominant_reason"].astype(str)
                if "dominant_reason" in data
                else np.full(transition_count, "none", dtype="<U16")
            ),
            probe_state=self._load_float_array(data, "probe_state", transition_count),
            probe_point=(
                data["probe_point"].astype(np.float32)
                if "probe_point" in data
                else np.zeros((transition_count, 2), dtype=np.float32)
            ),
            refined_grasp_target=(
                data["refined_grasp_target"].astype(np.float32)
                if "refined_grasp_target" in data
                else np.zeros((transition_count, 3), dtype=np.float32)
            ),
            metadata=metadata,
        )

    def _load_float_array(self, data: Any, name: str, transition_count: int) -> np.ndarray:
        if name in data:
            return data[name].astype(np.float32)
        return np.zeros(transition_count, dtype=np.float32)

    def _load_scalar_feature(
        self,
        data: Any,
        scalar_name: str,
        vector_name: str,
        vector_index: int,
        transition_count: int,
    ) -> np.ndarray:
        if scalar_name in data:
            return data[scalar_name].astype(np.float32)
        if vector_name in data:
            values = data[vector_name].astype(np.float32).reshape(transition_count, -1)
            if values.shape[1] > vector_index:
                return values[:, vector_index].astype(np.float32)
        return np.zeros(transition_count, dtype=np.float32)


def _feature_array_from_episode(episode: DemoEpisode, name: str) -> np.ndarray:
    transition_count = episode.length
    if name == "flattened_observation":
        return episode.observations.astype(np.float32, copy=False)
    if name == "dominant_reason_one_hot":
        return np.stack([reason_to_one_hot(reason) for reason in episode.dominant_reason], axis=0)
    if name == "probe_point_xy":
        return np.asarray(episode.probe_point, dtype=np.float32).reshape(transition_count, -1)[:, :2]
    if name == "refined_grasp_target_xyz":
        return np.asarray(episode.refined_grasp_target, dtype=np.float32).reshape(transition_count, -1)[:, :3]

    values = getattr(episode, name, None)
    if values is None:
        arr = np.zeros((transition_count, 1), dtype=np.float32)
    else:
        arr = np.asarray(values, dtype=np.float32)
        if arr.ndim == 0:
            arr = np.full((transition_count, 1), float(arr), dtype=np.float32)
        elif arr.ndim == 1:
            arr = arr.reshape(transition_count, 1)
        else:
            arr = arr.reshape(transition_count, -1)
    return arr.astype(np.float32, copy=False)


def build_transition_feature_matrix(
    episode: DemoEpisode,
    feature_names: Sequence[str] = DEFAULT_BC_FEATURE_NAMES,
) -> np.ndarray:
    parts = [_feature_array_from_episode(episode, name) for name in feature_names]
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def build_policy_feature_vector(
    obs: Any,
    feature_values: Mapping[str, Any],
    feature_names: Sequence[str] = DEFAULT_BC_FEATURE_NAMES,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for name in feature_names:
        if name == "flattened_observation":
            parts.append(flatten_observation(obs).astype(np.float32, copy=False))
            continue
        if name == "dominant_reason_one_hot":
            parts.append(reason_to_one_hot(str(feature_values.get("dominant_reason", "none"))))
            continue
        if name == "probe_point_xy":
            arr = np.asarray(feature_values.get("probe_point", np.zeros(2)), dtype=np.float32).reshape(-1)
            if arr.size < 2:
                arr = np.pad(arr, (0, 2 - arr.size))
            parts.append(arr[:2].astype(np.float32, copy=False))
            continue
        if name == "refined_grasp_target_xyz":
            arr = np.asarray(feature_values.get("refined_grasp_target", np.zeros(3)), dtype=np.float32).reshape(-1)
            if arr.size < 3:
                arr = np.pad(arr, (0, 3 - arr.size))
            parts.append(arr[:3].astype(np.float32, copy=False))
            continue
        value = feature_values.get(name, 0.0)
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            arr = np.zeros(1, dtype=np.float32)
        parts.append(arr.astype(np.float32, copy=False))
    return np.concatenate(parts).astype(np.float32, copy=False)


@dataclass
class TrajectoryWindow:
    """Fixed-horizon training sample for sequence policies."""

    condition: np.ndarray
    actions: np.ndarray
    mask: np.ndarray
    episode_path: Path
    start_index: int


class TrajectoryWindowDataset:
    """Build observation-conditioned action windows for BC/DP policies."""

    def __init__(
        self,
        root: str | Path,
        *,
        horizon: int = 8,
        only_success: bool = True,
        stride: int = 1,
    ):
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.demo_dataset = DemoDataset(root)
        self.horizon = int(horizon)
        self.only_success = bool(only_success)
        self.stride = int(stride)
        self.samples: list[TrajectoryWindow] = []
        self._build()
        if not self.samples:
            raise ValueError(
                f"No trajectory windows found in {root}. "
                "Collect successful scripted demos first or pass only_success=False."
            )

    def _build(self) -> None:
        for episode in self.demo_dataset:
            success = float(episode.metadata.get("success", 0.0) or 0.0)
            if self.only_success and success < 1.0:
                continue
            if episode.length <= 0:
                continue
            for start in range(0, episode.length, self.stride):
                end = min(start + self.horizon, episode.length)
                valid = end - start
                actions = np.zeros((self.horizon, episode.actions.shape[1]), dtype=np.float32)
                mask = np.zeros((self.horizon,), dtype=np.float32)
                actions[:valid] = episode.actions[start:end]
                mask[:valid] = 1.0
                condition = build_policy_features(
                    episode.observations[start],
                    uncertainty=float(episode.uncertainty[start]),
                    boundary_confidence=float(episode.boundary_confidence[start]),
                    dominant_reason=str(episode.dominant_reason[start]),
                    probe_state=float(episode.probe_state[start]),
                    probe_point=episode.probe_point[start],
                    refined_grasp_target=episode.refined_grasp_target[start],
                )
                self.samples.append(
                    TrajectoryWindow(
                        condition=condition.astype(np.float32, copy=False),
                        actions=actions,
                        mask=mask,
                        episode_path=episode.path,
                        start_index=start,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> TrajectoryWindow:
        return self.samples[index]

    @property
    def condition_dim(self) -> int:
        return int(self.samples[0].condition.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.samples[0].actions.shape[1])
