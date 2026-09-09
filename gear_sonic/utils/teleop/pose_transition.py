"""Smooth joint and VR-target transitions used by the teleoperation mode manager."""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


def validate_vr_pose(pose: np.ndarray) -> np.ndarray:
    """Copy a finite [left wrist, right wrist, head] XYZ + wxyz target."""
    result = np.asarray(pose, dtype=np.float64).copy()
    if result.shape != (3, 7) or not np.all(np.isfinite(result)):
        raise ValueError("VR target must be a finite (3, 7) pose")
    norms = np.linalg.norm(result[:, 3:], axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("VR target has a zero quaternion")
    result[:, 3:] /= norms[:, None]
    return result


class VRPoseTransition:
    """Frozen Cartesian endpoints with quintic timing and shortest rotation paths."""

    def __init__(self, start, goal, *, started_at: float, duration_s: float):
        self.start = validate_vr_pose(start)
        self.goal = validate_vr_pose(goal)
        self.progress = JointPoseTransition(
            np.zeros(1), np.ones(1), started_at=started_at, duration_s=duration_s
        )
        self.rotation = Rotation.from_quat(self.start[:, 3:], scalar_first=True)
        end_rotation = Rotation.from_quat(self.goal[:, 3:], scalar_first=True)
        self.rotation_delta = (end_rotation * self.rotation.inv()).as_rotvec()

    def sample(self, now: float) -> tuple[np.ndarray, bool]:
        progress = self.progress.sample(now)
        blend = float(progress.position[0])
        pose = self.start.copy()
        pose[:, :3] += blend * (self.goal[:, :3] - self.start[:, :3])
        pose[:, 3:] = (
            Rotation.from_rotvec(blend * self.rotation_delta) * self.rotation
        ).as_quat(scalar_first=True)
        return pose, progress.complete


UPPER_BODY_WIDTH = 17
IDLE_BASE_SHOULDER_PITCH_RAD = np.deg2rad(20.0)
# The G1's zero elbow command is the physical 90-degree pose. The requested
# 110-degree pose bends in the negative joint direction on this model.
IDLE_BASE_ELBOW_RAD = np.deg2rad(-20.0)

# Upper-body order on the planner wire:
# waist yaw/roll/pitch, interleaved left/right shoulder pitch/roll/yaw,
# elbows, then interleaved wrist roll/pitch/yaw.
#
# Shoulder pitch moves the complete arm behind the torso for balance. The
# negative elbow offset produces the requested 110-degree bend without driving
# the forearms downward. Neutral wrists keep both hands collinear with their
# forearms; the mirrored palms face each other.
IDLE_BASE_UPPER_BODY_RAD = np.asarray(
    [
        0.0,
        0.0,
        0.0,
        IDLE_BASE_SHOULDER_PITCH_RAD,
        IDLE_BASE_SHOULDER_PITCH_RAD,
        0.0,
        0.0,
        0.0,
        0.0,
        IDLE_BASE_ELBOW_RAD,
        IDLE_BASE_ELBOW_RAD,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)

# The waist remains owned by the locomotion planner. Only the fourteen arm
# joints are overridden while moving to or holding the calibration pose.
IDLE_BASE_UPPER_BODY_MASK = np.asarray(
    [False, False, False] + [True] * 14,
    dtype=bool,
)


@dataclass(frozen=True)
class JointTransitionSample:
    """One position/velocity sample from a finite joint transition."""

    position: np.ndarray
    velocity: np.ndarray
    complete: bool


class JointPoseTransition:
    """Quintic joint interpolation with zero endpoint velocity/acceleration."""

    def __init__(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        *,
        started_at: float,
        duration_s: float,
    ) -> None:
        start_array = np.asarray(start, dtype=np.float64).reshape(-1)
        goal_array = np.asarray(goal, dtype=np.float64).reshape(-1)
        if start_array.shape != goal_array.shape:
            raise ValueError("transition endpoints must have the same shape")
        if start_array.size == 0 or not np.all(np.isfinite(start_array)):
            raise ValueError("transition start must be finite and non-empty")
        if not np.all(np.isfinite(goal_array)):
            raise ValueError("transition goal must be finite")
        if not np.isfinite(started_at):
            raise ValueError("started_at must be finite")
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be positive and finite")

        self.start = start_array.copy()
        self.goal = goal_array.copy()
        self.started_at = float(started_at)
        self.duration_s = float(duration_s)

    def sample(self, now: float) -> JointTransitionSample:
        """Evaluate the transition at monotonic time ``now``."""
        if not np.isfinite(now):
            raise ValueError("sample time must be finite")
        u = float(np.clip((now - self.started_at) / self.duration_s, 0.0, 1.0))
        blend = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        blend_rate = (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / self.duration_s
        delta = self.goal - self.start
        return JointTransitionSample(
            position=self.start + blend * delta,
            velocity=blend_rate * delta,
            complete=u >= 1.0,
        )
