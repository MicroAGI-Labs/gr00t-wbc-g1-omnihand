import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.utils.teleop.pose_transition import (
    IDLE_BASE_ELBOW_RAD,
    IDLE_BASE_SHOULDER_PITCH_RAD,
    IDLE_BASE_UPPER_BODY_MASK,
    IDLE_BASE_UPPER_BODY_RAD,
    JointPoseTransition,
    VRPoseTransition,
    validate_vr_pose,
)


def test_idle_base_pose_has_rearward_elbows_and_neutral_wrists():
    assert IDLE_BASE_UPPER_BODY_RAD.shape == (17,)
    np.testing.assert_allclose(IDLE_BASE_SHOULDER_PITCH_RAD, np.deg2rad(20.0))
    np.testing.assert_allclose(IDLE_BASE_ELBOW_RAD, np.deg2rad(-20.0))
    np.testing.assert_allclose(
        IDLE_BASE_UPPER_BODY_RAD[3:11],
        [
            IDLE_BASE_SHOULDER_PITCH_RAD,
            IDLE_BASE_SHOULDER_PITCH_RAD,
            0.0,
            0.0,
            0.0,
            0.0,
            IDLE_BASE_ELBOW_RAD,
            IDLE_BASE_ELBOW_RAD,
        ],
    )
    np.testing.assert_allclose(IDLE_BASE_UPPER_BODY_RAD[11:], 0.0)
    np.testing.assert_array_equal(
        IDLE_BASE_UPPER_BODY_MASK,
        np.asarray([False, False, False] + [True] * 14),
    )


def test_quintic_transition_holds_endpoints_with_zero_velocity():
    transition = JointPoseTransition(
        np.asarray([1.0, -1.0]),
        np.asarray([3.0, 2.0]),
        started_at=10.0,
        duration_s=2.0,
    )

    before = transition.sample(9.0)
    start = transition.sample(10.0)
    end = transition.sample(12.0)
    after = transition.sample(13.0)

    np.testing.assert_allclose(before.position, [1.0, -1.0])
    np.testing.assert_allclose(start.velocity, [0.0, 0.0])
    assert not start.complete
    np.testing.assert_allclose(end.position, [3.0, 2.0])
    np.testing.assert_allclose(end.velocity, [0.0, 0.0], atol=1e-14)
    assert end.complete
    np.testing.assert_allclose(after.position, end.position)
    np.testing.assert_allclose(after.velocity, end.velocity)
    assert after.complete


def test_transition_is_symmetric_in_both_directions():
    start = np.asarray([-0.4, 0.8, 1.2])
    goal = np.asarray([0.6, -0.2, 0.2])
    forward = JointPoseTransition(start, goal, started_at=0.0, duration_s=2.0)
    backward = JointPoseTransition(goal, start, started_at=0.0, duration_s=2.0)

    for now in np.linspace(0.0, 2.0, 9):
        forward_sample = forward.sample(now)
        backward_sample = backward.sample(now)
        np.testing.assert_allclose(
            forward_sample.position + backward_sample.position,
            start + goal,
        )
        np.testing.assert_allclose(
            forward_sample.velocity + backward_sample.velocity,
            np.zeros_like(start),
        )


@pytest.mark.parametrize("duration", [0.0, -1.0, np.inf, np.nan])
def test_transition_rejects_invalid_duration(duration):
    with pytest.raises(ValueError, match="duration_s"):
        JointPoseTransition(
            np.zeros(1),
            np.ones(1),
            started_at=0.0,
            duration_s=duration,
        )


def test_vr_transition_is_frozen_smooth_and_takes_short_rotation_path():
    start = np.zeros((3, 7))
    start[:, 3:] = Rotation.from_euler("z", 170, degrees=True).as_quat(scalar_first=True)
    goal = start.copy()
    goal[:, :3] = 1
    goal[:, 3:] = Rotation.from_euler("z", -170, degrees=True).as_quat(scalar_first=True)
    transition = VRPoseTransition(start, goal, started_at=0, duration_s=2)
    start[:] = 99
    goal[:] = -99  # Later input changes must not modify either endpoint.
    previous = transition.sample(0)[0]
    for now in np.linspace(0.02, 2, 100):
        pose, complete = transition.sample(now)
        delta = pose[:, :3] - previous[:, :3]
        assert np.all(delta >= 0)
        assert np.max(delta) < 0.019  # Bounded steps, no discontinuity.
        np.testing.assert_allclose(np.linalg.norm(pose[:, 3:], axis=1), 1)
        previous = pose
    assert complete
    midpoint, _ = transition.sample(1)
    np.testing.assert_allclose(midpoint[:, :3], 0.5)
    rotated_x = Rotation.from_quat(midpoint[:, 3:], scalar_first=True).apply([1, 0, 0])
    np.testing.assert_allclose(rotated_x, np.tile([-1, 0, 0], (3, 1)), atol=1e-12)
    assert np.max(transition.sample(0.001)[0][:, :3]) < 2e-9
    np.testing.assert_allclose(transition.sample(3)[0], transition.sample(2)[0])


@pytest.mark.parametrize("fault", ["shape", "nan", "zero_quaternion"])
def test_vr_target_rejects_invalid_data(fault):
    pose = np.zeros((3, 7))
    pose[:, 3] = 1
    if fault == "shape":
        pose = pose[:2]
    elif fault == "nan":
        pose[0, 0] = np.nan
    else:
        pose[0, 3:] = 0
    with pytest.raises(ValueError):
        validate_vr_pose(pose)
