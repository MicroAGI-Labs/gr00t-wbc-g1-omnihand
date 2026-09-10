"""Independent middle-finger clutches in calibrated VR target coordinates."""

import numpy as np
from scipy.spatial.transform import Rotation

from .pose_transition import validate_vr_pose


class VRArmClutch:
    def __init__(self, *, controller_frame=False):
        self.controller_frame = controller_frame
        self.require_release = np.full(2, controller_frame, dtype=bool)
        self.heading = np.tile(np.eye(3), (2, 1, 1))
        self.pressed = np.zeros(2, dtype=bool)
        self.tracking = np.zeros(2, dtype=bool)
        self.position_offset = np.zeros((2, 3))
        self.rotation_offset = Rotation.identity(2).as_quat(scalar_first=True)

    def update(self, pose, held, grips, *, source_fresh, stopped):
        """Anchor each press to the emitted target; wait for braking to settle.

        Returns the target and three-point brake/re-anchor masks. Direct
        controllers capture headset heading per arm and hold the head target;
        legacy SMPL mode retains head tracking. Commit this state only after
        sending the corresponding command successfully.
        """
        pose = validate_vr_pose(pose)
        held = validate_vr_pose(held)
        reanchored = np.zeros(3, dtype=bool)
        for side, grip in enumerate(grips):
            if self.require_release[side]:
                if source_fresh and np.isfinite(grip) and grip <= 0.4:
                    self.require_release[side] = False
                grip = 0.
            # Hysteresis prevents repeated calibration around the threshold.
            if not np.isfinite(grip) or grip <= 0.4:
                self.pressed[side] = False
            elif grip >= 0.6:
                self.pressed[side] = True
            if not self.pressed[side]:
                self.tracking[side] = False
            if self.pressed[side] and not self.tracking[side] and source_fresh and stopped[side]:
                if self.controller_frame:
                    head = Rotation.from_quat(pose[2, 3:], scalar_first=True).as_matrix()
                    yaw = np.arctan2(-head[0, 1], head[0, 0])
                    self.heading[side] = Rotation.from_euler("z", -yaw).as_matrix()
            if self.controller_frame:
                pose[side, :3] = self.heading[side] @ pose[side, :3]
            if self.pressed[side] and not self.tracking[side] and source_fresh and stopped[side]:
                self.position_offset[side] = held[side, :3] - pose[side, :3]
                self.rotation_offset[side] = (
                    Rotation.from_quat(held[side, 3:], scalar_first=True)
                    * Rotation.from_quat(pose[side, 3:], scalar_first=True).inv()
                ).as_quat(scalar_first=True)
                self.tracking[side] = True
                reanchored[side] = True
            if self.tracking[side] and not reanchored[side]:
                pose[side, :3] += self.position_offset[side]
                pose[side, 3:] = (
                    Rotation.from_quat(self.rotation_offset[side], scalar_first=True)
                    * Rotation.from_quat(pose[side, 3:], scalar_first=True)
                ).as_quat(scalar_first=True)
            else:
                pose[side] = held[side]
        if self.controller_frame:
            pose[2] = held[2]
        return pose, np.r_[~self.tracking, self.controller_frame], reanchored
