from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Simple baseline policy for MVP smoke tests."""

    def __init__(self, action_space, seed: int = 0):
        self.action_space = action_space
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, step: int = 0, context: dict | None = None):
        del obs, step, context
        return self.action_space.sample()


class SineProbePolicy:
    """Deterministic placeholder that makes videos easier to compare."""

    def __init__(self, action_space, amplitude: float = 0.25):
        self.action_space = action_space
        self.amplitude = amplitude

    def predict(self, obs, step: int = 0, context: dict | None = None):
        del obs, context
        shape = self.action_space.shape
        if shape is None:
            return self.action_space.sample()
        action = self.amplitude * np.sin(step / 10.0) * np.ones(shape, dtype=np.float32)
        return np.clip(action, self.action_space.low, self.action_space.high)


class ActivePerceptionPolicy:
    """MVP active-perception wrapper.

    When visual uncertainty is high, it adds a small probing action before
    returning to the base policy. This is not a trained DP yet; it demonstrates
    the closed-loop interface needed by the final project.
    """

    def __init__(self, base_policy, action_space, probe_amplitude: float = 0.12):
        self.base_policy = base_policy
        self.action_space = action_space
        self.probe_amplitude = probe_amplitude

    def predict(self, obs, step: int = 0, context: dict | None = None):
        context = context or {}
        action = self.base_policy.predict(obs, step=step, context=context)
        if not context.get("active_probe", False):
            return action

        probe = np.zeros_like(action, dtype=np.float32)
        if probe.size:
            probe.flat[0] = self.probe_amplitude * ((-1.0) ** step)
        return np.clip(action + probe, self.action_space.low, self.action_space.high)
