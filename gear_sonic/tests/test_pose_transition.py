import numpy as np
import pytest

from gear_sonic.utils.teleop.pose_transition import (
    IDLE_BASE_UPPER_BODY_MASK,
    IDLE_BASE_UPPER_BODY_RAD,
    JointPoseTransition,
)


def test_idle_base_pose_has_forward_forearms_and_neutral_wrists():
    assert IDLE_BASE_UPPER_BODY_RAD.shape == (17,)
    # Zero is the G1's physical 90-degree elbow pose. Neutral wrists keep the
    # fingers pointing forward, parallel to the forearms, with opposing palms.
    np.testing.assert_allclose(IDLE_BASE_UPPER_BODY_RAD, 0.0)
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
