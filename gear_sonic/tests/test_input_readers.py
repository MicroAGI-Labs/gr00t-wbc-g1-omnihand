"""Body schema conversion without headset or ROS dependencies."""
from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.utils.teleop.input_readers import _body_data_to_24x7


def test_wire_body_pose_preserves_joint_order_and_padding():
    payload = {
        "joint_positions": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        "joint_orientations": [[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0]],
    }
    pose = _body_data_to_24x7(payload)
    assert pose.shape == (24, 7)
    assert pose.dtype == np.float32
    np.testing.assert_allclose(pose[0], [0.1, 0.2, 0.3, 0, 0, 0, 1])
    np.testing.assert_allclose(pose[1], [0.4, 0.5, 0.6, 0, 0, 1, 0])
    np.testing.assert_array_equal(pose[2:], 0)


@pytest.mark.parametrize("valid", [False, True])
def test_deviceio_body_pose_uses_joint_validity(valid):
    joint = SimpleNamespace(
        is_valid=valid,
        pose=SimpleNamespace(
            position=SimpleNamespace(x=1, y=2, z=3),
            orientation=SimpleNamespace(x=0, y=0, z=0, w=1),
        ),
    )
    data = SimpleNamespace(joints=SimpleNamespace(joints=lambda index: joint if index == 3 else None))
    pose = _body_data_to_24x7(data)
    if not valid:
        assert pose is None
    else:
        np.testing.assert_array_equal(pose[3], [1, 2, 3, 0, 0, 0, 1])
        np.testing.assert_array_equal(np.delete(pose, 3, axis=0), 0)


@pytest.mark.parametrize("data", [None, {}, {"joint_positions": [], "joint_orientations": []}])
def test_missing_body_pose_does_not_create_fake_tracking(data):
    assert _body_data_to_24x7(data) is None
