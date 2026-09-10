"""Offline clutch geometry, braking, and planner/hand publication regressions."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.end_effectors.protocol import decode_intent
from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.tests.test_commanded_pose_transition import decode_vr, streamer, vr_pose
from gear_sonic.utils.teleop.vr_arm_clutch import VRArmClutch
from gear_sonic.utils.teleop.vr_motion_conditioner import VRMotionConditioner, rotation_error


@pytest.mark.parametrize("side", [0, 1])
def test_press_reanchors_translation_and_rotation_without_changing_other_targets(side):
    clutch = VRArmClutch()
    held, source = vr_pose(), vr_pose()
    source[:, :3] += 2
    source[:, 3:] = Rotation.from_euler("xyz", [0.4, -0.6, 1.2]).as_quat(scalar_first=True)
    grips = np.zeros(2)
    grips[side] = 1
    options = dict(source_fresh=True, stopped=(True,) * 3)
    output, brake, anchored = clutch.update(source, held, grips, **options)
    np.testing.assert_array_equal(output[:2], held[:2])
    np.testing.assert_allclose(output[2], source[2])
    assert anchored[side] and not brake[side] and brake[1 - side]
    start = source.copy()
    source[side, :3] += [0.03, -0.02, 0.01]
    delta = Rotation.from_rotvec([0.03, 0.05, -0.01])
    source[side, 3:] = (Rotation.from_quat(start[side, 3:], scalar_first=True) * delta).as_quat(
        scalar_first=True
    )
    output, _, anchored = clutch.update(source, held, grips, **options)
    np.testing.assert_allclose(output[side, :3], held[side, :3] + [0.03, -0.02, 0.01])
    expected_rotation = Rotation.from_quat(held[side, 3:], scalar_first=True) * delta
    np.testing.assert_allclose(output[side, 3:], expected_rotation.as_quat(scalar_first=True))
    np.testing.assert_array_equal(output[1 - side], held[1 - side])
    assert not anchored.any()  # Holding must not repeatedly calibrate.
    held = output.copy()
    clutch.update(source, held, (0, 0), **options)
    source[side, :3] -= 3  # Reposition freely while released.
    source[side, 3:] = Rotation.from_euler("x", 2).as_quat(scalar_first=True)
    output, _, anchored = clutch.update(source, held, grips, **options)
    np.testing.assert_array_equal(output[:2], held[:2])
    assert anchored[side]


def test_hysteresis_invalid_grip_and_waiting_for_fresh_stopped_reference():
    clutch = VRArmClutch()
    pose = vr_pose()

    def update(value, fresh=True, stopped=True):
        return clutch.update(pose, pose, (value, 0), source_fresh=fresh, stopped=(stopped,) * 3)

    assert not update(0.5)[2].any()
    assert not update(0.7, fresh=False)[2].any()
    assert not update(0.7, stopped=False)[2].any()
    assert update(0.7)[2][0]
    assert not update(0.5)[1][0]  # Still held through hysteresis band.
    assert update(float("nan"))[1][0]
    assert update(0.7)[2][0]
    assert update(0.4)[1][0]


def test_release_brakes_only_that_arm_with_bounded_derivatives_and_no_old_goal_pursuit():
    clutch = VRArmClutch()
    limiter = VRMotionConditioner()
    source = vr_pose()
    limiter.seed(source, 0)
    previous = source.copy()
    previous_v = np.zeros((3, 3))
    previous_w = np.zeros((3, 3))
    previous_a = np.zeros((3, 3))
    previous_alpha = np.zeros((3, 3))
    for i in range(1, 251):
        source[:, 0] += 0.004
        source[:, 3:] = (Rotation.from_rotvec(np.tile([0, 0, 0.008], (3, 1))) *
                        Rotation.from_quat(source[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
        target, brake, anchor = clutch.update(
            source, limiter.pose, (1 if i < 51 else 0, 1), source_fresh=True,
            stopped=limiter.stopped_points,
        )
        output = limiter.update(target, i * 0.02, brake_mask=brake, reanchor_mask=anchor)
        v = (output[:, :3] - previous[:, :3]) / 0.02
        w = rotation_error(output[:, 3:], previous[:, 3:]) / 0.02
        a, alpha = (v - previous_v) / 0.02, (w - previous_w) / 0.02
        limits = limiter.limits
        for value, maximum in (
            (v, limits.speed * limits.hard_limit_factor),
            (w, limits.angular_speed * limits.hard_limit_factor),
            (a, limits.acceleration * limits.hard_limit_factor),
            (alpha, limits.angular_acceleration * limits.hard_limit_factor),
            ((a - previous_a) / 0.02, limits.jerk),
            ((alpha - previous_alpha) / 0.02, limits.angular_jerk),
        ):
            assert np.max(np.linalg.norm(value, axis=1)) <= maximum + 1e-7
        previous, previous_v, previous_w, previous_a, previous_alpha = output, v, w, a, alpha
        assert not limiter.fault
        if i == 50:
            released_at = output.copy()
        if i == 200:
            settled = output.copy()
    assert limiter.stopped_points[0] and not limiter.stopped_points[1]
    np.testing.assert_allclose(output[0], settled[0], atol=1e-10)
    assert 0 < output[0, 0] - released_at[0, 0] < 0.06
    assert output[1, 0] - released_at[1, 0] > 0.4


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


@pytest.mark.parametrize("limited", [False, True])
def test_streamed_release_reposition_repress_is_independent_and_continuous(live_streamer, limited):
    streamer, state, tick = live_streamer
    if limited:
        streamer.vr_conditioner = VRMotionConditioner()
        streamer.vr_conditioner.seed(streamer.last_vr_pose, state.now)
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
        state.pose[0, 0] -= 0.05  # Released motion cannot trip jump detection.
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
    if limited:
        assert not streamer.vr_conditioner.fault


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


def test_stale_tracking_still_faults_with_both_clutches_released(live_streamer):
    streamer, state, tick = live_streamer
    streamer.vr_conditioner = VRMotionConditioner()
    streamer.vr_conditioner.seed(streamer.last_vr_pose, state.now)
    tick()
    for _ in range(51):
        tick(fresh=False)
    assert streamer.vr_conditioner.fault == "stale 3PT input"


def test_repress_during_braking_waits_and_anchors_to_latest_pose(live_streamer):
    streamer, state, tick = live_streamer
    streamer.vr_conditioner = VRMotionConditioner()
    streamer.vr_conditioner.seed(streamer.last_vr_pose, state.now)
    state.grips[:] = [1, 1]
    tick()
    for _ in range(40):
        state.pose[:2, 0] += 0.002
        tick()
    state.grips[0] = 0
    tick()
    assert not streamer.vr_conditioner.stopped_points[0]
    state.grips[0] = 1
    tick()
    assert not streamer.vr_arm_clutch.tracking[0]
    for _ in range(150):
        state.pose[0, 0] -= 0.1
        state.pose[1, 0] += 0.002
        previous = streamer.last_vr_pose.copy()
        output = tick()
        if streamer.vr_arm_clutch.tracking[0]:
            np.testing.assert_allclose(output[0], previous[0], atol=1e-6)
            assert output[1, 0] > previous[1, 0]
            break
    else:
        pytest.fail("Held press did not engage after braking settled")
    np.testing.assert_allclose(
        streamer.vr_arm_clutch.position_offset[0], previous[0, :3] - state.pose[0, :3]
    )
    assert streamer.vr_conditioner.seed_generation == 1
    assert not streamer.vr_conditioner.fault


def test_invalid_pose_brakes_without_consuming_a_press(live_streamer):
    streamer, state, tick = live_streamer
    streamer.vr_conditioner = VRMotionConditioner()
    streamer.vr_conditioner.seed(streamer.last_vr_pose, state.now)
    tick()
    state.grips[:] = [1, 1]
    state.pose[0, 0] = np.nan
    output = tick()
    assert np.all(np.isfinite(output))
    assert streamer.vr_conditioner.fault == "invalid 3PT pose"
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
