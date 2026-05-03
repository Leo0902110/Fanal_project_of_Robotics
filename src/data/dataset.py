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
            contact_detected=(
                data["contact_detected"].astype(np.float32)
                if "contact_detected" in data
                else np.zeros(transition_count, dtype=np.float32)
            ),
            contact_strength=(
                data["contact_strength"].astype(np.float32)
                if "contact_strength" in data
                else np.zeros(transition_count, dtype=np.float32)
            ),
            left_force_norm=(
                data["left_force_norm"].astype(np.float32)
                if "left_force_norm" in data
                else np.zeros(transition_count, dtype=np.float32)
            ),
            right_force_norm=(
                data["right_force_norm"].astype(np.float32)
                if "right_force_norm" in data
                else np.zeros(transition_count, dtype=np.float32)
            ),
            net_force_norm=(
                data["net_force_norm"].astype(np.float32)
                if "net_force_norm" in data
                else np.zeros(transition_count, dtype=np.float32)
            ),
            pairwise_contact_used=(
                data["pairwise_contact_used"].astype(np.float32)
                if "pairwise_contact_used" in data
                else np.zeros(transition_count, dtype=np.float32)
            ),
            post_probe_uncertainty=(
                data["post_probe_uncertainty"].astype(np.float32)
                if "post_probe_uncertainty" in data
                else np.zeros(transition_count, dtype=np.float32)
            ),
            tcp_to_obj_x=self._load_scalar_feature(data, "tcp_to_obj_x", "tcp_to_obj_pos", 0, transition_count),
            tcp_to_obj_y=self._load_scalar_feature(data, "tcp_to_obj_y", "tcp_to_obj_pos", 1, transition_count),
            tcp_to_obj_z=self._load_scalar_feature(data, "tcp_to_obj_z", "tcp_to_obj_pos", 2, transition_count),
            obj_to_goal_x=self._load_scalar_feature(data, "obj_to_goal_x", "obj_to_goal_pos", 0, transition_count),
            obj_to_goal_y=self._load_scalar_feature(data, "obj_to_goal_y", "obj_to_goal_pos", 1, transition_count),
            obj_to_goal_z=self._load_scalar_feature(data, "obj_to_goal_z", "obj_to_goal_pos", 2, transition_count),
            metadata=metadata,
        )

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


def build_transition_feature_matrix(
    episode: DemoEpisode,
    feature_names: Sequence[str] = DEFAULT_BC_FEATURE_NAMES,
) -> np.ndarray:
    transition_count = episode.length
    parts: list[np.ndarray] = []
    for name in feature_names:
        if name == "flattened_observation":
            parts.append(episode.observations.astype(np.float32, copy=False))
            continue
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
        parts.append(arr.astype(np.float32, copy=False))
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
        value = feature_values.get(name, 0.0)
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            arr = np.zeros(1, dtype=np.float32)
        parts.append(arr.astype(np.float32, copy=False))
    return np.concatenate(parts).astype(np.float32, copy=False)
