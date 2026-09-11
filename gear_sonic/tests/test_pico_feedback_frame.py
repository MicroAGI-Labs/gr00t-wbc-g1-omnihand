"""Robot visualization feedback must not rotate teleop command targets."""

from types import SimpleNamespace

import msgpack
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.tests.test_commanded_pose_transition import vr_pose
from gear_sonic.utils.teleop.vr_arm_clutch import VRArmClutch


def feedback_packet(pose, root_rotation):
    # Match OutputInterface::create_output_data_map: only positions are rotated.
    return dict(
        vr_3point_position=root_rotation.apply(pose[:, :3]).ravel().tolist(),
        vr_3point_orientation=pose[:, 3:].ravel().tolist(),
        base_quat_target=root_rotation.as_quat(scalar_first=True).tolist(),
        body_q_measured=np.linspace(-0.5, 0.5, 29).tolist(),
        body_q_target=np.linspace(-0.4, 0.4, 29).tolist(),
        left_hand_q_measured=[0.0] * 7,
        right_hand_q_measured=[0.0] * 7,
    )


def read_feedback(packet, monkeypatch):
    monkeypatch.setattr(manager, "ZMQPoller", lambda **kwargs: SimpleNamespace(
        get_data=lambda: msgpack.packb(packet, use_bin_type=True)))
    reader = manager.FeedbackReader()
    assert reader.poll_feedback()
    return reader


@pytest.mark.parametrize("angles", [[0, 0, 0], [0, 0, 30], [12, -18, 75], [-20, 25, -135]])
def test_feedback_recovers_command_positions_and_keeps_orientation(monkeypatch, angles):
    pose = vr_pose()
    rotation = Rotation.from_euler("xyz", angles, degrees=True)
    reader = read_feedback(feedback_packet(pose, rotation), monkeypatch)
    np.testing.assert_allclose(reader.vr_pose, pose, atol=1e-12)
    np.testing.assert_allclose(reader.full_body_q_measured, np.linspace(-0.5, 0.5, 29))


def test_repeated_reconnect_with_released_grips_does_not_accumulate_rotation(monkeypatch):
    original = vr_pose()
    held = original.copy()
    for angles in ([5, -8, 30], [-10, 6, -40], [8, 12, 120]):
        reader = read_feedback(feedback_packet(
            held, Rotation.from_euler("xyz", angles, degrees=True)), monkeypatch)
        headset_pose = vr_pose()
        headset_pose[:, :3] += 2  # Headset/controllers moved during the outage.
        held = VRArmClutch(controller_frame=True).update(
            headset_pose, reader.vr_pose, (0, 0), source_fresh=True)
        np.testing.assert_allclose(held, original, atol=1e-12)


@pytest.mark.parametrize("quaternion", [None, [], [1, 0, 0], [0, 0, 0, 0],
                                        [float("nan"), 0, 0, 0], [float("inf"), 0, 0, 0]])
def test_missing_or_invalid_feedback_frame_cannot_become_a_vr_target(monkeypatch, quaternion):
    packet = feedback_packet(vr_pose(), Rotation.identity())
    if quaternion is None:
        packet.pop("base_quat_target")
    else:
        packet["base_quat_target"] = quaternion
    reader = read_feedback(packet, monkeypatch)
    assert reader.vr_pose is None
    assert reader.full_body_q_measured is not None
