from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


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
                "dominant_reason": episode.dominant_reason,
                "probe_state": episode.probe_state,
                "probe_point": episode.probe_point,
                "refined_grasp_target": episode.refined_grasp_target,
            }

    def load_episode(self, path: str | Path) -> DemoEpisode:
        path = Path(path)
        data = np.load(path, allow_pickle=True)
        metadata = data["metadata"].item() if "metadata" in data else {}
        return DemoEpisode(
            path=path,
            observations=data["observations"].astype(np.float32),
            actions=data["actions"].astype(np.float32),
            rewards=data["rewards"].astype(np.float32),
            dones=data["dones"].astype(bool),
            uncertainty=data["uncertainty"].astype(np.float32),
            boundary_confidence=data["boundary_confidence"].astype(np.float32),
            dominant_reason=(
                data["dominant_reason"].astype(str)
                if "dominant_reason" in data
                else np.full(data["actions"].shape[0], "none", dtype="<U16")
            ),
            probe_state=(
                data["probe_state"].astype(np.float32)
                if "probe_state" in data
                else np.zeros(data["actions"].shape[0], dtype=np.float32)
            ),
            probe_point=(
                data["probe_point"].astype(np.float32)
                if "probe_point" in data
                else np.zeros((data["actions"].shape[0], 2), dtype=np.float32)
            ),
            refined_grasp_target=(
                data["refined_grasp_target"].astype(np.float32)
                if "refined_grasp_target" in data
                else np.zeros((data["actions"].shape[0], 3), dtype=np.float32)
            ),
            metadata=metadata,
        )
