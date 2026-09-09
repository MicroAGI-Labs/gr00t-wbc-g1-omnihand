"""
Dataset configuration for the Sonic VLA pipeline.

Provides feature/modality config dicts and a convenience function to
instantiate the G1 RobotModel needed for FK and joint configuration
assembly during data collection.

Joint names, counts, and group indices are derived at runtime from the
``RobotModel`` (via its ``supplemental_info``).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from gear_sonic.data.robot_model import RobotModel
from gear_sonic.end_effectors.profiles import HandProfile, raw_hand_name

EGO_VIEW_HEIGHT: int = 480
EGO_VIEW_WIDTH: int = 640
WRIST_VIEW_HEIGHT: int = 480
WRIST_VIEW_WIDTH: int = 640
FPS: int = 50


_JOINT_GROUPS_FOR_STATE: list[str] = [
    "left_leg",
    "right_leg",
    "waist",
    "left_arm",
    "left_hand",
    "right_arm",
    "right_hand",
]


def _get_joint_group_slices(
    robot_model: RobotModel, hand_profile: HandProfile | None = None
) -> dict[str, dict[str, int]]:
    """Derive ``{group_name: {"start": ..., "end": ...}}`` from the robot model."""
    slices: dict[str, dict[str, int]] = {}
    cursor = 0
    for group in _JOINT_GROUPS_FOR_STATE:
        if hand_profile is not None and group in {"left_hand", "right_hand"}:
            width = hand_profile.width
        else:
            width = len(robot_model.get_joint_group_indices(group))
        slices[group] = {"start": cursor, "end": cursor + width}
        cursor += width
    return slices


def get_modality_config_sonic_vla(
    robot_model: RobotModel, hand_profile: HandProfile | None = None
) -> dict:
    """Return the modality config for the Sonic VLA dataset.

    Produces the exact content of meta/modality.json.
    """
    group_slices = _get_joint_group_slices(robot_model, hand_profile)

    return {
        "state": {
            **group_slices,
            "left_wrist_pos": {
                "start": 0,
                "end": 3,
                "original_key": "observation.eef_state",
            },
            "left_wrist_abs_quat": {
                "start": 3,
                "end": 7,
                "original_key": "observation.eef_state",
                "rotation_type": "quaternion",
            },
            "right_wrist_pos": {
                "start": 7,
                "end": 10,
                "original_key": "observation.eef_state",
            },
            "right_wrist_abs_quat": {
                "start": 10,
                "end": 14,
                "original_key": "observation.eef_state",
                "rotation_type": "quaternion",
            },
            "root_orientation": {
                "start": 0,
                "end": 4,
                "original_key": "observation.root_orientation",
                "rotation_type": "quaternion",
            },
            "projected_gravity": {
                "start": 0,
                "end": 3,
                "original_key": "observation.projected_gravity",
            },
            "cpp_rotation_offset": {
                "start": 0,
                "end": 4,
                "original_key": "observation.cpp_rotation_offset",
                "rotation_type": "quaternion",
            },
            "init_base_quat": {
                "start": 0,
                "end": 4,
                "original_key": "observation.init_base_quat",
                "rotation_type": "quaternion",
            },
        },
        "action": {
            "delta_heading": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.delta_heading",
            },
            "motion_token": {
                "start": 0,
                "end": 64,
                "original_key": "action.motion_token",
            },
            "smpl_joints": {
                "start": 0,
                "end": 72,
                "original_key": "teleop.smpl_joints",
            },
            "smpl_pose": {
                "start": 0,
                "end": 63,
                "original_key": "teleop.smpl_pose",
            },
            "body_quat_w": {
                "start": 0,
                "end": 4,
                "original_key": "teleop.body_quat_w",
                "rotation_type": "quaternion",
            },
            "target_body_orientation": {
                "start": 0,
                "end": 6,
                "original_key": "teleop.target_body_orientation",
                "rotation_type": "rotation_6d",
            },
            "left_hand_joints": {
                "start": 0,
                "end": hand_profile.width if hand_profile is not None else 7,
                "original_key": "teleop.left_hand_joints",
            },
            "right_hand_joints": {
                "start": 0,
                "end": hand_profile.width if hand_profile is not None else 7,
                "original_key": "teleop.right_hand_joints",
            },
            "left_wrist_joints": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.left_wrist_joints",
            },
            "right_wrist_joints": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.right_wrist_joints",
            },
            "stream_mode": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.stream_mode",
            },
            "planner_mode": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.planner_mode",
            },
            "planner_movement": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.planner_movement",
            },
            "planner_facing": {
                "start": 0,
                "end": 3,
                "original_key": "teleop.planner_facing",
            },
            "planner_speed": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.planner_speed",
            },
            "planner_height": {
                "start": 0,
                "end": 1,
                "original_key": "teleop.planner_height",
            },
            "vr_3pt_position": {
                "start": 0,
                "end": 9,
                "original_key": "teleop.vr_3pt_position",
            },
            "vr_3pt_orientation": {
                "start": 0,
                "end": 18,
                "original_key": "teleop.vr_3pt_orientation",
                "rotation_type": "rotation_6d",
            },
        },
        "video": {
            "ego_view": {"original_key": "observation.images.ego_view"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }


def dataset_joint_names(
    robot_model: RobotModel, hand_profile: HandProfile | None = None
) -> list[str]:
    """Return canonical body/hand names, replacing only the hand channels."""
    if hand_profile is None:
        return robot_model.joint_names
    names: list[str] = []
    for group in _JOINT_GROUPS_FOR_STATE:
        if group == "left_hand":
            names.extend(hand_profile.left.joint_names)
        elif group == "right_hand":
            names.extend(hand_profile.right.joint_names)
        else:
            indices = robot_model.get_joint_group_indices(group)
            names.extend(robot_model.joint_names[index] for index in indices)
    return names


def assemble_dataset_configuration(
    robot_model: RobotModel,
    body_actuated_joint_values,
    left_hand_values,
    right_hand_values,
    hand_profile: HandProfile,
) -> np.ndarray:
    """Assemble 43/49 values by declared group/name order, never width alone."""
    left = np.asarray(left_hand_values, dtype=np.float64).reshape(-1)
    right = np.asarray(right_hand_values, dtype=np.float64).reshape(-1)
    if left.shape != (hand_profile.width,) or right.shape != (hand_profile.width,):
        raise ValueError(
            f"{hand_profile.name} requires {hand_profile.width} values per side"
        )
    neutral = np.zeros(7, dtype=np.float64)
    fk_q = robot_model.get_configuration_from_actuated_joints(
        body_actuated_joint_values=np.asarray(body_actuated_joint_values),
        left_hand_actuated_joint_values=neutral,
        right_hand_actuated_joint_values=neutral,
    )
    parts = []
    for group in _JOINT_GROUPS_FOR_STATE:
        if group == "left_hand":
            parts.append(left)
        elif group == "right_hand":
            parts.append(right)
        else:
            parts.append(fk_q[robot_model.get_joint_group_indices(group)])
    return np.concatenate(parts)


def get_features_sonic_vla(
    robot_model: RobotModel, hand_profile: HandProfile | None = None
) -> dict:
    """Return the dataset features for the Sonic VLA dataset.

    The returned dict populates the "features" key of meta/info.json.
    """
    joint_names = dataset_joint_names(robot_model, hand_profile)
    num_joints = len(joint_names)
    hand_width = hand_profile.width if hand_profile is not None else 7

    return {
        "observation.images.ego_view": {
            "dtype": "video",
            "shape": [EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3],
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": joint_names,
        },
        "observation.body_joint_velocity": {
            "dtype": "float32",
            "shape": (29,),
            "names": [f"body_joint_velocity_{index:02d}" for index in range(29)],
        },
        "observation.eef_state": {
            "dtype": "float32",
            "shape": (14,),
            "names": [
                "left_wrist_pos",
                "left_wrist_abs_quat",
                "right_wrist_pos",
                "right_wrist_abs_quat",
            ],
        },
        "action.wbc": {
            "dtype": "float32",
            "shape": (num_joints,),
            "names": joint_names,
        },
        "control.hand_applied_position": {
            "dtype": "float32",
            "shape": (2 * hand_width,),
            "names": (
                list(hand_profile.left.joint_names + hand_profile.right.joint_names)
                if hand_profile is not None
                else [f"hand_applied_{index}" for index in range(2 * hand_width)]
            ),
        },
        "teleop.hand_closed": {
            "dtype": "bool",
            "shape": (2,),
            "names": ["left", "right"],
        },
        "observation.root_orientation": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["base_qw", "base_qx", "base_qy", "base_qz"],
        },
        "observation.projected_gravity": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["gravity_x", "gravity_y", "gravity_z"],
        },
        "observation.cpp_rotation_offset": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["rot_offset_qw", "rot_offset_qx", "rot_offset_qy", "rot_offset_qz"],
        },
        "observation.init_base_quat": {
            "dtype": "float32",
            "shape": (4,),
            "names": ["init_base_qw", "init_base_qx", "init_base_qy", "init_base_qz"],
        },
        "teleop.delta_heading": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["delta_heading"],
        },
        "action.motion_token": {
            "dtype": "float32",
            "shape": (64,),
            "names": [f"motion_token_{index:02d}" for index in range(64)],
        },
        "action.motion_token_valid": {
            "dtype": "uint8",
            "shape": (1,),
            "names": ["motion_token_valid"],
        },
        "observation.base_angular_velocity": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["angular_velocity_x", "angular_velocity_y", "angular_velocity_z"],
        },
        f"observation.{raw_hand_name(hand_profile)}_left_raw": {
            "dtype": "float32",
            "shape": (hand_width,),
            "names": (
                list(hand_profile.left.joint_names)
                if hand_profile is not None
                else [f"left_hand_{index}" for index in range(hand_width)]
            ),
        },
        f"observation.{raw_hand_name(hand_profile)}_right_raw": {
            "dtype": "float32",
            "shape": (hand_width,),
            "names": (
                list(hand_profile.right.joint_names)
                if hand_profile is not None
                else [f"right_hand_{index}" for index in range(hand_width)]
            ),
        },
        f"action.{raw_hand_name(hand_profile)}_left_raw": {
            "dtype": "float32",
            "shape": (hand_width,),
            "names": (
                list(hand_profile.left.joint_names)
                if hand_profile is not None
                else [f"left_hand_{index}" for index in range(hand_width)]
            ),
        },
        f"action.{raw_hand_name(hand_profile)}_right_raw": {
            "dtype": "float32",
            "shape": (hand_width,),
            "names": (
                list(hand_profile.right.joint_names)
                if hand_profile is not None
                else [f"right_hand_{index}" for index in range(hand_width)]
            ),
        },
        "observation.left_hand_valid": {
            "dtype": "uint8",
            "shape": (1,),
            "names": ["left_hand_valid"],
        },
        "observation.right_hand_valid": {
            "dtype": "uint8",
            "shape": (1,),
            "names": ["right_hand_valid"],
        },
        "episode.success": {
            "dtype": "uint8",
            "shape": (1,),
            "names": ["success"],
        },
        "teleop.smpl_joints": {
            "dtype": "float32",
            "shape": (72,),
            "names": "smpl_joints",
        },
        "teleop.smpl_pose": {
            "dtype": "float32",
            "shape": (63,),
            "names": "smpl_pose",
        },
        "teleop.body_quat_w": {
            "dtype": "float32",
            "shape": (4,),
            "names": "body_quat_w",
        },
        "teleop.target_body_orientation": {
            "dtype": "float32",
            "shape": (6,),
            "names": [
                "target_body_r00",
                "target_body_r10",
                "target_body_r01",
                "target_body_r11",
                "target_body_r02",
                "target_body_r12",
            ],
        },
        "teleop.left_hand_joints": {
            "dtype": "float32",
            "shape": (hand_width,),
            "names": "left_hand_joints",
        },
        "teleop.right_hand_joints": {
            "dtype": "float32",
            "shape": (hand_width,),
            "names": "right_hand_joints",
        },
        "teleop.smpl_frame_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["smpl_frame_index"],
        },
        "teleop.left_wrist_joints": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"],
        },
        "teleop.right_wrist_joints": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw"],
        },
        "teleop.stream_mode": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["stream_mode"],
        },
        "teleop.planner_mode": {
            "dtype": "int32",
            "shape": (1,),
            "names": ["locomotion_mode"],
        },
        "teleop.planner_movement": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["movement_x", "movement_y", "movement_z"],
        },
        "teleop.planner_facing": {
            "dtype": "float32",
            "shape": (3,),
            "names": ["facing_x", "facing_y", "facing_z"],
        },
        "teleop.planner_speed": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["speed"],
        },
        "teleop.planner_height": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["height"],
        },
        "teleop.vr_3pt_position": {
            "dtype": "float32",
            "shape": (9,),
            "names": [
                "lwrist_x", "lwrist_y", "lwrist_z",
                "rwrist_x", "rwrist_y", "rwrist_z",
                "neck_x", "neck_y", "neck_z",
            ],
        },
        "teleop.vr_3pt_orientation_wxyz": {
            "dtype": "float32",
            "shape": (12,),
            "names": [
                f"{source}_q{component}"
                for source in ("lwrist", "rwrist", "head")
                for component in ("w", "x", "y", "z")
            ],
        },
        "teleop.vr_3pt_valid": {
            "dtype": "uint8",
            "shape": (1,),
            "names": ["vr_3pt_valid"],
        },
        "teleop.smpl_valid": {
            "dtype": "uint8",
            "shape": (1,),
            "names": ["smpl_valid"],
        },
        "teleop.control_mode": {
            "dtype": "uint8",
            "shape": (1,),
            "names": ["manipulation_0_locomotion_1_combined_2_unknown_255"],
        },
        "teleop.locomotion_speed_m_s": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["configured_locomotion_speed_m_s_or_minus_1"],
        },
        "capture.robot_state_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["robot_state_sequence"],
        },
        "capture.robot_state_source_timestamp_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["robot_state_source_timestamp_ns"],
        },
        "capture.robot_state_sample_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["robot_state_sample_monotonic_ns"],
        },
        "capture.robot_state_publish_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["robot_state_publish_monotonic_ns"],
        },
        "capture.robot_state_received_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["robot_state_received_monotonic_ns"],
        },
        "capture.camera_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["camera_sequence"],
        },
        "capture.camera_source_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["camera_source_monotonic_ns"],
        },
        "capture.camera_sample_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["camera_sample_monotonic_ns"],
        },
        "capture.camera_publish_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["camera_publish_monotonic_ns"],
        },
        "capture.camera_received_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["camera_received_monotonic_ns"],
        },
        "capture.hand_state_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["hand_controller_sequence"],
        },
        "capture.hand_state_publish_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["hand_state_publish_sequence"],
        },
        "capture.hand_state_source_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["hand_state_source_monotonic_ns"],
        },
        "capture.hand_state_publish_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["hand_state_publish_monotonic_ns"],
        },
        "capture.hand_state_received_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["hand_state_received_monotonic_ns"],
        },
        "capture.hand_intent_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["hand_intent_sequence"],
        },
        "capture.hand_intent_source_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["hand_intent_source_monotonic_ns"],
        },
        "capture.hand_intent_received_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["hand_intent_received_monotonic_ns"],
        },
        "capture.pico_pose_sequence": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["pico_pose_sequence"],
        },
        "capture.pico_pose_sample_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["pico_pose_sample_monotonic_ns"],
        },
        "capture.pico_pose_publish_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["pico_pose_publish_monotonic_ns"],
        },
        "capture.pico_pose_received_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["pico_pose_received_monotonic_ns"],
        },
        "capture.planner_publish_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["planner_publish_monotonic_ns"],
        },
        "capture.planner_received_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["planner_received_monotonic_ns"],
        },
        "capture.manager_publish_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["manager_publish_monotonic_ns"],
        },
        "capture.manager_received_monotonic_ns": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["manager_received_monotonic_ns"],
        },
        "teleop.vr_3pt_orientation": {
            "dtype": "float32",
            "shape": (18,),
            "names": [
                "lwrist_r00", "lwrist_r10", "lwrist_r01", "lwrist_r11", "lwrist_r02", "lwrist_r12",
                "rwrist_r00", "rwrist_r10", "rwrist_r01", "rwrist_r11", "rwrist_r02", "rwrist_r12",
                "neck_r00", "neck_r10", "neck_r01", "neck_r11", "neck_r02", "neck_r12",
            ],
        },
    }


def get_wrist_camera_features() -> dict:
    """Optional wrist videos and their original host capture timestamps."""
    features = {}
    for name in ("left_wrist", "right_wrist"):
        features[f"observation.images.{name}"] = {
            "dtype": "video",
            "shape": (WRIST_VIEW_HEIGHT, WRIST_VIEW_WIDTH, 3),
            "names": ["height", "width", "channel"],
        }
        features[f"capture.{name}_source_timestamp_ns"] = {
            "dtype": "int64", "shape": (1,), "names": ["timestamp_ns"],
        }
    return features


def get_zed_stereo_features() -> dict:
    """Optional native-orientation ZED left/right RGB and float32 depth streams."""
    return {
        "observation.images.ego_view_left": {
            "dtype": "video", "shape": (EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.depth.ego_view": {
            "dtype": "float32", "shape": (EGO_VIEW_HEIGHT, EGO_VIEW_WIDTH),
            "names": ["height", "width"],
        },
        "capture.ego_view_left_source_timestamp_ns": {
            "dtype": "int64", "shape": (1,), "names": ["timestamp_ns"],
        },
        "capture.ego_view_depth_source_timestamp_ns": {
            "dtype": "int64", "shape": (1,), "names": ["timestamp_ns"],
        },
    }


def get_zed_stereo_modality_config() -> dict:
    return {
        "video": {"ego_view_left": {"original_key": "observation.images.ego_view_left"}},
    }


def get_wrist_camera_modality_config() -> dict:
    """Modality config entries for optional wrist cameras."""
    return {
        "video": {
            "left_wrist": {"original_key": "observation.images.left_wrist"},
            "right_wrist": {"original_key": "observation.images.right_wrist"},
        },
    }


def get_g1_robot_model(
    waist_location: Literal[
        "lower_body", "upper_body", "lower_and_upper_body"
    ] = "lower_and_upper_body",
    high_elbow_pose: bool = False,
):
    """Instantiate the G1 + ThreeFinger RobotModel for Sonic VLA."""
    from gear_sonic.data.robot_model.instantiation.g1 import instantiate_g1_robot_model

    return instantiate_g1_robot_model(
        waist_location=waist_location,
        high_elbow_pose=high_elbow_pose,
    )
