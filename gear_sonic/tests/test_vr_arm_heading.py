"""Controller displacement must retain its direction when the headset tips down."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.teleop.pico_controls import XR_TO_ROBOT, controller_poses
from gear_sonic.utils.teleop.vr_arm_clutch import VRArmClutch


@pytest.mark.parametrize("side", [0, 1])
@pytest.mark.parametrize("yaw", [-170, -90, 0, 45, 90, 180])
@pytest.mark.parametrize("pitch", [-110, -90, 0, 75, 89, 90, 91, 110])
def test_grip_heading_preserves_motion_through_headset_pitch(side, yaw, pitch):
    # Build SDK-frame poses, including a nonzero tracking origin, then use the
    # same conversion and grip calibration as the live PICO path.
    heading = Rotation.from_euler("z", yaw, degrees=True)
    head = heading * Rotation.from_euler("y", pitch, degrees=True)
    raw = np.array([
        [1.2, 1.0, -0.2, 0, 0, 0, 1],
        [1.6, 1.0, -0.2, 0, 0, 0, 1],
        [1.4, 1.5, 0.0, 0, 0, 0, 1],
    ])
    raw[2, 3:] = Rotation.from_matrix(XR_TO_ROBOT.T @ head.as_matrix() @ XR_TO_ROBOT).as_quat()
    held = np.array([
        [0.3, 0.2, 0.1, 1, 0, 0, 0],
        [0.3, -0.2, 0.1, 1, 0, 0, 0],
        [0.0, 0.0, 0.5, 1, 0, 0, 0],
    ])
    clutch = VRArmClutch(controller_frame=True)
    grips = np.zeros(2)
    clutch.update(controller_poses(*raw), held, grips, source_fresh=True)
    grips[side] = 1
    np.testing.assert_array_equal(
        clutch.update(controller_poses(*raw), held, grips, source_fresh=True), held
    )

    for displacement in ([0, 0.05, 0], [0.04, 0, 0], [0, 0, 0.03]):
        moved = raw.copy()
        moved[side, :3] += XR_TO_ROBOT.T @ heading.apply(displacement)
        result = clutch.update(controller_poses(*moved), held, grips, source_fresh=True)
        np.testing.assert_allclose(result[side, :3], held[side, :3] + displacement, atol=1e-12)
        np.testing.assert_allclose(result[side, 3:], held[side, 3:], atol=1e-12)
        np.testing.assert_array_equal(result[1 - side], held[1 - side])
        np.testing.assert_array_equal(result[2], held[2])
