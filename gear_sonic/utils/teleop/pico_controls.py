"""Direct Pico poses and the operator's exclusive locomotion/hand controls."""

import numpy as np
from scipy.spatial.transform import Rotation

from .pose_transition import validate_vr_pose

XR_TO_ROBOT = np.array(((0., 0., -1.), (-1., 0., 0.), (0., 1., 0.)))
CONTROLLER_STALE_SECONDS = 0.200


def controller_poses(left, right, head):
    """Convert SDK XYZ/xyzw poses into robot-axis XYZ/wxyz, without body inference."""
    raw = np.asarray((left, right, head), dtype=np.float64)
    pose = validate_vr_pose(np.concatenate((raw[:, :3], raw[:, -1:], raw[:, 3:6]), axis=1))
    pose[:, :3] = pose[:, :3] @ XR_TO_ROBOT.T
    pose[:, 3:] = Rotation.from_matrix(
        XR_TO_ROBOT @ Rotation.from_quat(pose[:, 3:], scalar_first=True).as_matrix() @ XR_TO_ROBOT.T
    ).as_quat(scalar_first=True)
    return pose


def fresh_controller_sample(sample, now):
    if sample is None or sample.get("controller_poses") is None:
        return False
    received_at = sample.get("timestamp_monotonic", -np.inf)
    return received_at <= now < received_at + CONTROLLER_STALE_SECONDS


class PicoLocomotion:
    def __init__(self):
        self.enabled = False
        self.slow = False
        self.invalidate()

    def invalidate(self):
        self.neutral_required = True
        self.clicked = True

    def update(self, axes, click, action, *, fresh):
        if not fresh or not np.all(np.isfinite(axes)):
            self.invalidate()
            return 0., 0.
        if click and not self.clicked:
            self.enabled = not self.enabled
            self.neutral_required = True
        self.clicked = click
        if self.neutral_required:
            if not click and max(abs(v) for v in axes) <= 0.15:
                self.neutral_required = False
            return 0., 0.
        forward = axes[1] if abs(axes[1]) > 0.15 else 0.
        turn = axes[2] if abs(axes[2]) > 0.15 else 0.
        if not self.enabled or (forward and turn):
            return 0., 0.
        return forward, turn


class PicoHandGate:
    def __init__(self):
        self.tracking = np.zeros(2, dtype=bool)
        self.trigger_released = np.zeros(2, dtype=bool)
        self.home_open = np.zeros(2, dtype=bool)

    def update(self, triggers, engaged, *, valid, force_open=False):
        holds = np.ones(2, dtype=bool)
        for side, trigger in enumerate(triggers):
            if not valid or force_open or not engaged[side]:
                self.tracking[side] = self.trigger_released[side] = False
            if not valid:
                self.home_open[side] = False
                continue
            if force_open:
                self.home_open[side] = True
            elif engaged[side]:
                self.home_open[side] = False
                if trigger <= 0.4:
                    self.trigger_released[side] = True
                elif trigger >= 0.6 and self.trigger_released[side]:
                    self.tracking[side] = True
            holds[side] = not (self.tracking[side] or self.home_open[side])
        return holds, self.home_open.copy()
