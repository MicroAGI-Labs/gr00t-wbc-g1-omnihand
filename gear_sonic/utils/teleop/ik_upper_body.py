"""Retarget calibrated PICO wrist poses to G1 arm joint targets."""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation

from gear_sonic.utils.teleop.vis.vr3pt_pose_visualizer import G1_KEY_FRAME_OFFSETS

ARM_JOINT_TYPES = (
    "shoulder_pitch",
    "shoulder_roll",
    "shoulder_yaw",
    "elbow",
    "wrist_roll",
    "wrist_pitch",
    "wrist_yaw",
)
ARM_JOINT_NAMES = tuple(
    f"{side}_{joint}_joint" for side in ("left", "right") for joint in ARM_JOINT_TYPES
)
ARM_OUTPUT_JOINT_NAMES = tuple(
    f"{side}_{joint}_joint" for joint in ARM_JOINT_TYPES for side in ("left", "right")
)
UPPER_BODY_FEEDBACK_INDICES = np.array(
    [12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int64,
)
ARM_JOINT_MASK = np.arange(17) >= 3


def wrist_link_targets(vr_3pt_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert SONIC wrist key points to wrist-yaw link SE(3) targets."""
    pose = np.asarray(vr_3pt_pose, dtype=np.float64)
    if pose.shape != (3, 7) or not np.all(np.isfinite(pose)):
        raise ValueError("vr_3pt_pose must be a finite (3, 7) array")

    targets = []
    for side, key in enumerate(("left_wrist", "right_wrist")):
        quaternion = pose[side, 3:]
        norm = float(np.linalg.norm(quaternion))
        if not 0.99 <= norm <= 1.01:
            raise ValueError(f"{key} quaternion is not unit length")
        rotation = Rotation.from_quat(quaternion / norm, scalar_first=True).as_matrix()
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = rotation
        target[:3, 3] = pose[side, :3] - rotation @ G1_KEY_FRAME_OFFSETS[key]
        targets.append(target)
    return targets[0], targets[1]


def rate_limited_arm_step(
    previous: np.ndarray, target: np.ndarray, dt: float, max_speed: float
) -> tuple[np.ndarray, np.ndarray]:
    """Advance only masked arm joints and return the resulting velocity."""
    old = np.asarray(previous, dtype=np.float64)
    goal = np.asarray(target, dtype=np.float64)
    if old.shape != (17,) or goal.shape != (17,) or not np.all(np.isfinite((old, goal))):
        raise ValueError("upper-body position arrays must contain 17 finite values")
    if not np.isfinite(dt) or dt <= 0.0 or not np.isfinite(max_speed) or max_speed <= 0.0:
        raise ValueError("dt and max_speed must be positive and finite")
    result = old.copy()
    max_delta = max_speed * dt
    delta = np.clip(goal[ARM_JOINT_MASK] - old[ARM_JOINT_MASK], -max_delta, max_delta)
    result[ARM_JOINT_MASK] += delta
    return result, (result - old) / dt


class IkUpperBodyRetargeter:
    """Stateful 20 Hz Pink IK with a rate-limited 50 Hz arm output."""

    IK_PERIOD_SECONDS = 1.0 / 20.0
    MAX_ARM_SPEED_RAD_S = 1.5

    def __init__(self, robot_model) -> None:
        try:
            from decoupled_wbc.control.robot_model.robot_model import ReducedRobotModel
            from decoupled_wbc.control.teleop.solver.body.body_ik_solver import BodyIKSolver
            from decoupled_wbc.control.teleop.solver.body.body_ik_solver_settings import (
                BodyIKSolverSettings,
            )
        except ImportError as exc:
            raise RuntimeError(
                "IK upper-body planner mode requires pin-pink and qpsolvers; "
                "rerun install_scripts/install_pico.sh"
            ) from exc

        import qpsolvers

        if not qpsolvers.available_solvers:
            raise RuntimeError("IK upper-body planner mode requires a qpsolvers backend")

        self.robot = robot_model
        self.reduced_robot = ReducedRobotModel.from_active_groups(self.robot, ["arms"])
        self.solver = BodyIKSolver(BodyIKSolverSettings())
        self.solver.register_robot(self.reduced_robot)
        self.frame_names = tuple(
            self.robot.supplemental_info.hand_frame_names[side] for side in ("left", "right")
        )
        self._arm_full_index = np.asarray(
            [self.robot.dof_index(name) for name in ARM_OUTPUT_JOINT_NAMES], dtype=np.int64
        )
        self._last_solution_time: float | None = None
        self._last_output_time: float | None = None
        self._target = np.zeros(17, dtype=np.float64)
        self._output = np.zeros(17, dtype=np.float64)

    def reset(self, measured_body_q: np.ndarray, *, now: float | None = None) -> None:
        """Seed both IK and the output limiter from the measured robot pose."""
        measured = np.asarray(measured_body_q, dtype=np.float64)
        if measured.shape != (29,) or not np.all(np.isfinite(measured)):
            raise ValueError("measured_body_q must be a finite 29-DOF array")

        full_q = self.robot.default_body_pose.copy()
        for name, value in zip(ARM_JOINT_NAMES, measured[15:]):
            full_q[self.robot.dof_index(name)] = value
        reduced_q = self.reduced_robot.full_to_reduced_configuration(full_q)
        reduced_q = self.reduced_robot.clip_configuration(reduced_q)
        self.solver.configuration.q = reduced_q.copy()
        self.solver.configuration.update()
        self.reduced_robot.cache_forward_kinematics(reduced_q, auto_clip=False)
        for task in self.solver.tasks.values():
            task.set_target_from_configuration(self.solver.configuration)

        measured_upper = measured[UPPER_BODY_FEEDBACK_INDICES]
        self._target = measured_upper.copy()
        self._output = measured_upper.copy()
        timestamp = time.monotonic() if now is None else now
        self._last_solution_time = None
        self._last_output_time = timestamp

    def update(
        self, vr_3pt_pose: np.ndarray, *, now: float | None = None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return position, velocity, and mask arrays in SONIC upper-body order."""
        timestamp = time.monotonic() if now is None else now
        if self._last_output_time is None:
            raise RuntimeError("retargeter must be reset from robot feedback before use")

        if (
            self._last_solution_time is None
            or timestamp - self._last_solution_time >= self.IK_PERIOD_SECONDS
        ):
            left_target, right_target = wrist_link_targets(vr_3pt_pose)
            reduced_q = self.solver(
                {self.frame_names[0]: left_target, self.frame_names[1]: right_target}
            )
            full_q = self.reduced_robot.reduced_to_full_configuration(reduced_q)
            candidate = np.asarray(full_q[self._arm_full_index], dtype=np.float64)
            if not np.all(np.isfinite(candidate)):
                raise RuntimeError("upper-body IK returned an invalid joint target")
            self._target[3:] = candidate
            self._last_solution_time = timestamp

        dt = float(np.clip(timestamp - self._last_output_time, 1e-4, 0.1))
        self._output, velocity = rate_limited_arm_step(
            self._output,
            self._target,
            dt,
            self.MAX_ARM_SPEED_RAD_S,
        )
        self._last_output_time = timestamp
        return self._output.copy(), velocity, ARM_JOINT_MASK.copy()
