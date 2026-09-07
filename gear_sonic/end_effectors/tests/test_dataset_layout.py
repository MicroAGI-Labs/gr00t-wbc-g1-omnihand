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
    assert modality["state"]["left_hand"] == {"start": 22, "end": 32}
    assert modality["state"]["right_hand"] == {"start": 39, "end": 49}
    assert modality["state"]["left_wrist_abs_quat"]["end"] == 7
    assert modality["action"]["left_hand_joints"]["end"] == 10
    assert modality["action"]["right_hand_joints"]["end"] == 10
