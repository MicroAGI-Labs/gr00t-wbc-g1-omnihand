from __future__ import annotations

import numpy as np

from gear_sonic.data.features_sonic_vla import (
    assemble_dataset_configuration,
    dataset_joint_names,
    get_features_sonic_vla,
    get_g1_robot_model,
    get_modality_config_sonic_vla,
)
from gear_sonic.end_effectors.profiles import OMNIHAND_O10


def test_omnihand_dataset_is_49_wide_in_declared_group_order():
    model = get_g1_robot_model()
    body = np.arange(29, dtype=np.float64)
    left = np.arange(10, dtype=np.float64) + 100
    right = np.arange(10, dtype=np.float64) + 200
    result = assemble_dataset_configuration(model, body, left, right, OMNIHAND_O10)
    names = dataset_joint_names(model, OMNIHAND_O10)
    assert result.shape == (49,)
    assert len(names) == 49
    assert names[22:32] == list(OMNIHAND_O10.left.joint_names)
    assert names[39:49] == list(OMNIHAND_O10.right.joint_names)
    np.testing.assert_array_equal(result[22:32], left)
    np.testing.assert_array_equal(result[39:49], right)


def test_features_and_modality_follow_dynamic_hand_width():
    model = get_g1_robot_model()
    features = get_features_sonic_vla(model, OMNIHAND_O10)
    modality = get_modality_config_sonic_vla(model, OMNIHAND_O10)
    assert features["observation.state"]["shape"] == (49,)
    assert features["action.wbc"]["shape"] == (49,)
    assert features["control.hand_applied_position"]["shape"] == (20,)
    assert features["observation.body_joint_velocity"]["shape"] == (29,)
    assert features["observation.base_angular_velocity"]["shape"] == (3,)
    assert features["observation.omnihand_left_raw"]["shape"] == (10,)
    assert features["action.omnihand_right_raw"]["shape"] == (10,)
    assert features["action.motion_token"]["dtype"] == "float32"
    assert features["episode.success"]["dtype"] == "uint8"
    assert features["teleop.vr_3pt_orientation_wxyz"]["shape"] == (12,)
    assert {
        "capture.robot_state_publish_monotonic_ns",
        "capture.robot_state_received_monotonic_ns",
        "capture.camera_sample_monotonic_ns",
        "capture.camera_publish_monotonic_ns",
        "capture.camera_received_monotonic_ns",
        "capture.hand_state_publish_monotonic_ns",
        "capture.hand_state_received_monotonic_ns",
        "capture.pico_pose_publish_monotonic_ns",
        "capture.pico_pose_received_monotonic_ns",
        "capture.planner_publish_monotonic_ns",
        "capture.planner_received_monotonic_ns",
        "capture.manager_publish_monotonic_ns",
        "capture.manager_received_monotonic_ns",
    } <= features.keys()
    assert modality["state"]["left_hand"] == {"start": 22, "end": 32}
    assert modality["state"]["right_hand"] == {"start": 39, "end": 49}
    assert modality["state"]["left_wrist_abs_quat"]["end"] == 7
    assert modality["action"]["left_hand_joints"]["end"] == 10
    assert modality["action"]["right_hand_joints"]["end"] == 10
    assert "base_angular_velocity" not in modality["state"]
    assert "body_joint_velocity" not in modality["state"]
