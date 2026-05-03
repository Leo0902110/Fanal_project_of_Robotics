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


class JointScriptedPickCubePolicy:
    """Heuristic PickCube controller for Panda ``pd_joint_delta_pos`` control.

    The policy follows the same high-level phases as the end-effector scripted
    controller, but maps task-space errors to small joint-delta commands. It is
    meant to provide a Windows-friendly expert-ish baseline when Pinocchio is
    unavailable for ``pd_ee_delta_pose``.
    """

    def __init__(
        self,
        action_space,
        use_active_probe: bool = False,
        uncertainty_threshold: float = 0.5,
        probe_steps: int = 2,
        arm_gain: float = 4.8,
        max_arm_command: float = 0.9,
    ):
        self.action_space = action_space
        self.use_active_probe = use_active_probe
        self.uncertainty_threshold = uncertainty_threshold
        self.max_probe_steps = probe_steps
        self.arm_gain = arm_gain
        self.max_arm_command = max_arm_command
        self.grasp_offset_z = 0.015
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

    def _task_vectors(self, obs, context: dict | None = None) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        context = context or {}
        extra = obs.get("extra", {}) if isinstance(obs, dict) else {}
        tcp_to_obj = self._vector(extra.get("tcp_to_obj_pos"), 3)
        obj_to_goal = self._vector(extra.get("obj_to_goal_pos"), 3)
        is_grasped = 0.0
        grasped_value = extra.get("is_grasped")
        if grasped_value is not None:
            try:
                is_grasped = float(np.asarray(grasped_value, dtype=np.float32).reshape(-1)[0])
            except Exception:
                is_grasped = 0.0
        if tcp_to_obj is None or obj_to_goal is None:
            oracle = context.get("oracle", {}) if isinstance(context, dict) else {}
            tcp_pos = self._vector(oracle.get("tcp_pos"), 3)
            obj_pos = self._vector(oracle.get("obj_pos"), 3)
            goal_pos = self._vector(oracle.get("goal_pos"), 3)
            if tcp_to_obj is None and tcp_pos is not None and obj_pos is not None:
                tcp_to_obj = obj_pos - tcp_pos
            if obj_to_goal is None and obj_pos is not None and goal_pos is not None:
                obj_to_goal = goal_pos - obj_pos
        return tcp_to_obj, obj_to_goal, is_grasped

    def _base_action(self, gripper: float) -> np.ndarray:
        action = np.zeros(self.action_space.shape, dtype=np.float32)
        if action.size:
            action[-1] = gripper
        return action

    def _clip(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action, self.action_space.low, self.action_space.high)

    def _set_phase(self, phase: str) -> None:
        if self.phase != phase:
            self.phase = phase
            self.phase_steps = 0

    def _should_probe(self, context: dict) -> bool:
        if not self.use_active_probe or self.phase != "approach":
            return False
        if self.probe_count >= self.max_probe_steps:
            return False
        if bool(context.get("active_probe", False)):
            return True
        uncertainty = context.get("uncertainty", {})
        if isinstance(uncertainty, dict):
            score = float(uncertainty.get("uncertainty", context.get("mean_uncertainty", 0.0)))
        else:
            score = float(context.get("mean_uncertainty", 0.0))
        return score >= self.uncertainty_threshold

    def _probe_action(self) -> np.ndarray:
        action = self._base_action(gripper=1.0)
        direction = -1.0 if self.probe_count % 2 else 1.0
        if action.size >= 3:
            action[0] = 0.18 * direction
            action[2] = -0.12 * direction
        self.probe_count += 1
        return self._clip(action)

    def _servo_to_tcp_offset(self, tcp_to_obj: np.ndarray, target_offset: np.ndarray, gripper: float) -> np.ndarray:
        action = self._base_action(gripper=gripper)
        error = target_offset - tcp_to_obj

        if action.size >= 7:
            # Empirical signs for Panda pd_joint_delta_pos in PickCube-v1:
            # joints 0/2 mainly sweep lateral y, joints 3/5 influence x, joint 1 helps vertical descent/lift.
            action[0] = -self.arm_gain * error[1]
            action[2] = -0.65 * self.arm_gain * error[1]
            action[3] = -0.95 * self.arm_gain * error[0]
            action[5] = -0.45 * self.arm_gain * error[0]
            action[1] = 0.85 * self.arm_gain * error[2]
            action[4] = 0.15 * self.arm_gain * error[1]
            action[:7] = np.clip(action[:7], -self.max_arm_command, self.max_arm_command)
        return self._clip(action)

    def _transfer_action(self, obj_to_goal: np.ndarray | None, gripper: float) -> np.ndarray:
        action = self._base_action(gripper=gripper)
        if obj_to_goal is None or action.size < 7:
            return self._clip(action)

        error = obj_to_goal.astype(np.float32, copy=False)
        action[0] = 4.0 * error[1]
        action[2] = 2.2 * error[1]
        action[3] = 5.0 * error[0]
        action[5] = 2.4 * error[0]
        action[1] = -3.2 * error[2]
        action[:7] = np.clip(action[:7], -self.max_arm_command, self.max_arm_command)
        return self._clip(action)

    def predict(self, obs, step: int = 0, context: dict | None = None):
        del step
        context = context or {}
        tcp_to_obj, obj_to_goal, is_grasped = self._task_vectors(obs, context)
        if tcp_to_obj is None:
            return self._base_action(gripper=1.0)

        if self._should_probe(context):
            return self._probe_action()

        xy_error = float(np.linalg.norm(tcp_to_obj[:2]))
        self.phase_steps += 1

        if self.phase == "approach":
            if xy_error < 0.04 or self.phase_steps > 12:
                self._set_phase("descend")
            else:
                return self._servo_to_tcp_offset(
                    tcp_to_obj,
                    np.array([0.0, 0.0, float(tcp_to_obj[2])], dtype=np.float32),
                    gripper=1.0,
                )

        if self.phase == "descend":
            if float(tcp_to_obj[2]) > self.grasp_offset_z - 0.01 or self.phase_steps > 18:
                self._set_phase("close_gripper")
            else:
                return self._servo_to_tcp_offset(
                    tcp_to_obj,
                    np.array([0.0, 0.0, self.grasp_offset_z], dtype=np.float32),
                    gripper=1.0,
                )

        if self.phase == "close_gripper":
            if is_grasped > 0.5:
                self._set_phase("transfer")
                return self._transfer_action(obj_to_goal, gripper=-1.0)
            if self.phase_steps > 5:
                self._set_phase("lift")
            return self._servo_to_tcp_offset(
                tcp_to_obj,
                np.array([0.0, 0.0, self.grasp_offset_z], dtype=np.float32),
                gripper=-1.0,
            )

        if self.phase == "lift":
            if is_grasped > 0.5 or self.phase_steps > 3:
                self._set_phase("transfer")
                return self._transfer_action(obj_to_goal, gripper=-1.0)
            return self._servo_to_tcp_offset(
                tcp_to_obj,
                np.array([0.0, 0.0, self.grasp_offset_z], dtype=np.float32),
                gripper=-1.0,
            )

        if self.phase == "transfer":
            if obj_to_goal is not None and np.linalg.norm(obj_to_goal) < 0.025:
                self._set_phase("settle")
                return self._base_action(gripper=-1.0)
            return self._transfer_action(obj_to_goal, gripper=-1.0)

        if self.phase == "settle":
            return self._base_action(gripper=-1.0)

        return self._base_action(gripper=1.0)


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
        if isinstance(uncertainty, dict):
            score = float(uncertainty.get("mean_uncertainty", uncertainty.get("uncertainty", 0.0)))
        else:
            score = float(context.get("mean_uncertainty", 0.0))
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

    def __init__(
        self,
        base_policy,
        action_space,
        probe_amplitude: float = 0.12,
        max_probe_steps: int = 2,
    ):
        self.base_policy = base_policy
        self.action_space = action_space
        self.probe_amplitude = probe_amplitude
        self.max_probe_steps = max_probe_steps
        self.probe_count = 0

    def predict(self, obs, step: int = 0, context: dict | None = None):
        context = context or {}
        action = self.base_policy.predict(obs, step=step, context=context)
        if not context.get("active_probe", False):
            return action
        if self.probe_count >= self.max_probe_steps:
            return action

        probe = np.zeros_like(action, dtype=np.float32)
        if probe.size:
            probe.flat[0] = self.probe_amplitude * ((-1.0) ** step)
        self.probe_count += 1
        return np.clip(action + probe, self.action_space.low, self.action_space.high)
