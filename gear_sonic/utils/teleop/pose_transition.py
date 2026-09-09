"""Smooth joint-space transitions used by the teleoperation mode manager."""

from dataclasses import dataclass

import numpy as np

# Alignment pose from PR #3, in planner wire order: waist yaw/roll/pitch,
# interleaved shoulder pitch/roll/yaw, elbows, then wrist roll/pitch/yaw.
IDLE_BASE_UPPER_BODY_RAD = np.deg2rad([0, 0, 0, 20, 20, 0, 0, 0, 0, -20, -20, 0, 0, 0, 0, 0, 0])
# Override only the fourteen arm joints; the planner retains the waist.
IDLE_BASE_UPPER_BODY_MASK = np.asarray([False] * 3 + [True] * 14)


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
