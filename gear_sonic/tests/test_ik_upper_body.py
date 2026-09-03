import json

import numpy as np
import pytest

from gear_sonic.utils.teleop.ik_upper_body import (
    ARM_JOINT_MASK,
    rate_limited_arm_step,
    wrist_link_targets,
)
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (
    HEADER_SIZE,
    build_planner_message,
)


def test_wrist_keypoints_are_converted_to_link_targets():
    pose = np.zeros((3, 7), dtype=np.float64)
    pose[:, 3] = 1.0
    pose[0, :3] = [0.18, -0.025, 0.0]
    pose[1, :3] = [0.18, 0.025, 0.0]

    left, right = wrist_link_targets(pose)

    np.testing.assert_allclose(left, np.eye(4), atol=1e-7)
    np.testing.assert_allclose(right, np.eye(4), atol=1e-7)


def test_wrist_target_rejects_non_unit_quaternion():
    pose = np.zeros((3, 7), dtype=np.float64)
    pose[:, 3] = 1.0
    pose[1, 3:] = 0.0

    with pytest.raises(ValueError, match="unit length"):
        wrist_link_targets(pose)


def test_rate_limiter_never_changes_planner_owned_waist():
    previous = np.zeros(17)
    target = np.full(17, 10.0)

    position, velocity = rate_limited_arm_step(previous, target, 0.02, 1.5)

    np.testing.assert_array_equal(position[:3], np.zeros(3))
    np.testing.assert_array_equal(velocity[:3], np.zeros(3))
    np.testing.assert_allclose(position[ARM_JOINT_MASK], 0.03)
    np.testing.assert_allclose(velocity[ARM_JOINT_MASK], 1.5)


def test_planner_message_carries_explicit_arm_mask():
    mask = [False, False, False] + [True] * 14
    message = build_planner_message(
        1,
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        upper_body_position=np.zeros(17),
        upper_body_velocity=np.zeros(17),
        upper_body_mask=mask,
    )

    header_start = len(b"planner")
    header = json.loads(
        message[header_start : header_start + HEADER_SIZE].rstrip(b"\x00").decode("utf-8")
    )
    assert header["fields"][-1] == {
        "name": "upper_body_mask",
        "dtype": "bool",
        "shape": [17],
    }
    assert message[-17:] == bytes(mask)


def test_planner_mask_requires_matching_position():
    with pytest.raises(ValueError, match="requires upper_body_position"):
        build_planner_message(
            1,
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            upper_body_mask=[True] * 17,
        )
