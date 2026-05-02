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


@dataclass
class DemoEpisode:
    path: Path
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    uncertainty: np.ndarray
    boundary_confidence: np.ndarray
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
            metadata=metadata,
        )
