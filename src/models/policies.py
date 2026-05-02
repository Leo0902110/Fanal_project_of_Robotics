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


class ScriptedPickCubePolicy:
    """Lightweight scripted PickCube controller for MVP comparisons.

    This policy assumes ManiSkill's Panda `pd_ee_delta_pose` control mode:
    action[0:3] are normalized end-effector position deltas, action[3:6] are
    rotation deltas, and action[-1] controls the gripper.
    """

    def __init__(
        self,
        action_space,
        use_active_probe: bool = False,
        uncertainty_threshold: float = 0.5,
        probe_steps: int = 1,
    ):
        self.action_space = action_space
        self.use_active_probe = use_active_probe
        self.uncertainty_threshold = uncertainty_threshold
        self.max_probe_steps = probe_steps
        self.phase = "approach"
        self.phase_steps = 0
        self.probe_count = 0

    def _vector(self, value, length: int) -> np.ndarray | None:
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.reshape(-1, arr.shape[-1])[0]
        arr = arr.reshape(-1)
        if arr.size < length:
            return None
        return arr[:length]

    def _tcp_obj_goal(
        self, obs, context: dict
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        extra = obs.get("extra", {}) if isinstance(obs, dict) else {}
        oracle = context.get("oracle", {}) if isinstance(context, dict) else {}
        tcp_pose = self._vector(extra.get("tcp_pose"), 7)
        obj_pos = self._vector(oracle.get("obj_pos"), 3)
        goal_pos = self._vector(oracle.get("goal_pos"), 3)
        if goal_pos is None:
            goal_pos = self._vector(extra.get("goal_pos"), 3)
        tcp_pos = tcp_pose[:3] if tcp_pose is not None else None
        return tcp_pos, obj_pos, goal_pos

    def _base_action(self, gripper: float = 1.0) -> np.ndarray:
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        if action.size:
            action[-1] = gripper
        return action

    def _move_toward(self, target: np.ndarray, tcp_pos: np.ndarray, gripper: float) -> np.ndarray:
        action = self._base_action(gripper=gripper)
        error = target - tcp_pos
        # pd_ee_delta_pose maps normalized [-1, 1] to roughly [-0.1m, 0.1m].
        action[:3] = np.clip(error / 0.08, -0.7, 0.7)
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _should_probe(self, context: dict) -> bool:
        if not self.use_active_probe or self.phase != "approach":
            return False
        if self.probe_count >= self.max_probe_steps:
            return False
        uncertainty = context.get("uncertainty", {})
        score = float(uncertainty.get("uncertainty", 0.0)) if isinstance(uncertainty, dict) else 0.0
        return score >= self.uncertainty_threshold

    def _probe_action(self) -> np.ndarray:
        action = self._base_action(gripper=1.0)
        direction = -1.0 if self.probe_count % 2 else 1.0
        action[1] = 0.08 * direction
        self.probe_count += 1
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _set_phase(self, phase: str) -> None:
        if self.phase != phase:
            self.phase = phase
            self.phase_steps = 0

    def predict(self, obs, step: int = 0, context: dict | None = None):
        del step
        context = context or {}
        tcp_pos, obj_pos, goal_pos = self._tcp_obj_goal(obs, context)
        if tcp_pos is None or obj_pos is None or goal_pos is None:
            return self.action_space.sample()

        if self._should_probe(context):
            return self._probe_action()

        approach_target = obj_pos + np.array([0.0, 0.0, 0.08], dtype=np.float32)
        grasp_target = obj_pos.copy()
        transfer_target = goal_pos + np.array([0.0, 0.0, 0.02], dtype=np.float32)

        self.phase_steps += 1
        if self.phase == "approach":
            if np.linalg.norm(tcp_pos - approach_target) < 0.035 or self.phase_steps > 45:
                self._set_phase("descend")
            else:
                return self._move_toward(approach_target, tcp_pos, gripper=1.0)

        if self.phase == "descend":
            if np.linalg.norm(tcp_pos - grasp_target) < 0.025 or self.phase_steps > 35:
                self._set_phase("close_gripper")
            else:
                return self._move_toward(grasp_target, tcp_pos, gripper=1.0)

        if self.phase == "close_gripper":
            if self.phase_steps > 15:
                self._set_phase("transfer")
            return self._move_toward(grasp_target, tcp_pos, gripper=-1.0)

        if self.phase == "transfer":
            if np.linalg.norm(tcp_pos - transfer_target) < 0.035 or self.phase_steps > 45:
                self._set_phase("release")
            else:
                return self._move_toward(transfer_target, tcp_pos, gripper=-1.0)

        return self._move_toward(transfer_target, tcp_pos, gripper=1.0)


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
