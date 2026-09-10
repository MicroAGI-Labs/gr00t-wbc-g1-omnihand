"""Offline clutch geometry and planner/hand publication regressions."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.end_effectors.protocol import decode_intent
from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.tests.test_commanded_pose_transition import decode_vr, streamer, vr_pose
from gear_sonic.utils.teleop.vr_arm_clutch import VRArmClutch


@pytest.mark.parametrize("side", [0, 1])
def test_press_reanchors_translation_and_rotation_without_changing_other_targets(side):
    clutch = VRArmClutch()
    held, source = vr_pose(), vr_pose()
    source[:, :3] += 2
    source[:, 3:] = Rotation.from_euler("xyz", [0.4, -0.6, 1.2]).as_quat(scalar_first=True)
    grips = np.zeros(2)
    grips[side] = 1
    options = dict(source_fresh=True)
    output = clutch.update(source, held, grips, **options)
    np.testing.assert_array_equal(output[:2], held[:2])
    np.testing.assert_allclose(output[2], source[2])
    assert clutch.tracking[side] and not clutch.tracking[1 - side]
    start = source.copy()
    source[side, :3] += [0.03, -0.02, 0.01]
    delta = Rotation.from_rotvec([0.03, 0.05, -0.01])
    source[side, 3:] = (Rotation.from_quat(start[side, 3:], scalar_first=True) * delta).as_quat(
        scalar_first=True
    )
    output = clutch.update(source, held, grips, **options)
    np.testing.assert_allclose(output[side, :3], held[side, :3] + [0.03, -0.02, 0.01])
    expected_rotation = Rotation.from_quat(held[side, 3:], scalar_first=True) * delta
    np.testing.assert_allclose(output[side, 3:], expected_rotation.as_quat(scalar_first=True))
    np.testing.assert_array_equal(output[1 - side], held[1 - side])
    held = output.copy()
    clutch.update(source, held, (0, 0), **options)
    source[side, :3] -= 3  # Reposition freely while released.
    source[side, 3:] = Rotation.from_euler("x", 2).as_quat(scalar_first=True)
    output = clutch.update(source, held, grips, **options)
    np.testing.assert_array_equal(output[:2], held[:2])
    assert clutch.tracking[side]


def test_hysteresis_invalid_grip_and_waiting_for_fresh_reference():
    clutch = VRArmClutch()
    pose = vr_pose()

    def update(value, fresh=True):
        clutch.update(pose, pose, (value, 0), source_fresh=fresh)
        return bool(clutch.tracking[0])

    assert not update(0.5)
    assert not update(0.7, fresh=False)
    assert update(0.7)
    assert update(0.5)  # Still held through the hysteresis band.
    assert not update(float("nan"))
    assert update(0.7)
    assert not update(0.4)


@pytest.fixture
def live_streamer(streamer, monkeypatch):
    state = SimpleNamespace(now=0.0, stamp=1_000_000_000, pose=vr_pose(), grips=[0, 0])
    streamer.last_vr_pose = vr_pose()
    streamer.left_hand_ik_solver = streamer.right_hand_ik_solver = None
    streamer.three_point = SimpleNamespace(process_smpl_pose=lambda pose: pose.copy())
    streamer.reader.get_latest = lambda: {"body_poses_np": state.pose, "timestamp_ns": state.stamp}
    monkeypatch.setattr(manager.time, "monotonic", lambda: state.now)
    monkeypatch.setattr(manager, "get_controller_inputs", lambda reader: (False, 0, 0, *state.grips))
    monkeypatch.setattr(manager, "compute_hand_joints_from_inputs", lambda *args: (np.zeros(7), np.zeros(7)))

    def tick(fresh=True):
        state.now += 0.02
        if fresh:
            state.stamp += 20_000_000
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
        return decode_vr(streamer.packets[-1])

    return streamer, state, tick


def test_streamed_release_reposition_repress_is_independent_and_continuous(live_streamer):
    streamer, state, tick = live_streamer
    initial = streamer.last_vr_pose.copy()
    state.pose[:2, 0] += 1
    np.testing.assert_allclose(tick()[:2], initial[:2], atol=1e-7)
    state.grips[:] = [1, 1]
    np.testing.assert_allclose(tick()[:2], initial[:2], atol=1e-7)
    for _ in range(30):
        state.pose[:2, 0] += 0.001
        tick()
    assert streamer.last_vr_pose[0, 0] > initial[0, 0] + 0.005
    state.grips[0] = 0
    for _ in range(100):
        state.pose[0, 0] -= 0.05  # A released arm keeps its last command.
        state.pose[1, 0] += 0.001
        tick()
    held = streamer.last_vr_pose.copy()
    state.grips[0] = 1
    output = tick()
    np.testing.assert_allclose(output[0], held[0], atol=1e-7)
    assert output[1, 0] > held[1, 0]
    for _ in range(30):
        state.pose[0, 0] += 0.001
        tick()
    assert streamer.last_vr_pose[0, 0] > held[0, 0] + 0.005


def test_failed_send_does_not_commit_press_or_release(live_streamer):
    streamer, state, tick = live_streamer
    tick()
    for grip in (1, 0, 1):
        state.grips[0] = grip
        before = deepcopy(streamer.vr_arm_clutch)

        def fail(packet):
            raise RuntimeError("send failed")

        streamer.socket.send = fail
        with pytest.raises(RuntimeError, match="send failed"):
            tick()
        np.testing.assert_array_equal(streamer.vr_arm_clutch.tracking, before.tracking)
        np.testing.assert_array_equal(streamer.vr_arm_clutch.position_offset, before.position_offset)
        state.pose[0, 0] += 1
        held = streamer.last_vr_pose.copy()
        streamer.socket.send = streamer.packets.append
        np.testing.assert_allclose(tick()[0], held[0], atol=1e-7)


def test_invalid_pose_rejects_packet_without_consuming_a_press(live_streamer):
    streamer, state, tick = live_streamer
    tick()
    state.grips[:] = [1, 1]
    state.pose[0, 0] = np.nan
    sent = len(streamer.packets)
    held = streamer.last_vr_pose.copy()
    with pytest.raises(ValueError):
        tick()
    assert len(streamer.packets) == sent
    np.testing.assert_array_equal(streamer.last_vr_pose, held)
    assert not streamer.vr_arm_clutch.tracking.any()


@pytest.mark.parametrize("triggers, expected", [((0, 0), (False, False)), ((1, 0), (True, False)),
                                               ((0, 1), (False, True))])
def test_grips_never_close_external_hands(monkeypatch, triggers, expected):
    monkeypatch.setattr(manager, "get_controller_inputs", lambda reader: (False, *triggers, 1, 1))
    packets = []
    manager.HandIntentStream().publish(
        SimpleNamespace(send=packets.append), SimpleNamespace(get_timestamp_ns=lambda: 123)
    )
    intent = decode_intent(packets[-1])
    assert (intent["left"]["closed"], intent["right"]["closed"]) == expected
