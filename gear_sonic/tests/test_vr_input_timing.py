"""Tracking cadence regressions, using fake SDK data and offline commands only."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.tests.test_commanded_pose_transition import streamer, vr_pose, decode_vr
from gear_sonic.utils.teleop.vr_motion_conditioner import VRMotionConditioner, rotation_error


def test_sdk_snapshot_retries_when_body_changes_during_pose_read(monkeypatch):
    stamp = [10]
    calls = []

    def pose():
        calls.append(stamp[0])
        result = np.full((24, 7), stamp[0])
        if len(calls) == 1:
            stamp[0] += 1
        return result

    monkeypatch.setattr(manager, "xrt", SimpleNamespace(
        get_time_stamp_ns=lambda: 100, get_body_timestamp_ns=lambda: stamp[0],
        get_body_joints_pose=pose,
    ))
    reader = manager.PicoReader()
    sample = reader._read_sample()
    assert calls == [10, 11]
    assert sample["body_timestamp_ns"] == 11
    np.testing.assert_array_equal(sample["body_poses_np"], 11)
    reader._latest = sample
    assert reader._read_sample() is sample  # No repeated pose allocations.
    stamp[0] += 1  # Body updates even if the packet clock is unchanged.
    assert reader._read_sample()["body_timestamp_ns"] == 12


def test_sdk_snapshot_discards_continuously_torn_reads(monkeypatch):
    stamps = iter(range(100))
    monkeypatch.setattr(manager, "xrt", SimpleNamespace(
        get_time_stamp_ns=lambda: 100, get_body_timestamp_ns=lambda: next(stamps),
        get_body_joints_pose=lambda: np.zeros((24, 7)),
    ))
    assert manager.PicoReader()._read_sample() is None


def test_sdk_without_body_clock_retains_packet_timestamp(monkeypatch):
    monkeypatch.setattr(manager, "xrt", SimpleNamespace(
        get_time_stamp_ns=lambda: 123, get_body_joints_pose=lambda: np.zeros((24, 7)),
    ))
    sample = manager.PicoReader()._read_sample()
    assert sample["body_timestamp_ns"] is None
    assert sample["timestamp_ns"] == 123
    assert sample["source_timestamp_ns"] == 123


def test_legacy_sdk_retries_when_packet_changes_during_pose_read(monkeypatch):
    stamps = iter((100, 101, 101, 101))
    poses = iter((np.zeros((24, 7)), np.ones((24, 7))))
    monkeypatch.setattr(manager, "xrt", SimpleNamespace(
        get_time_stamp_ns=lambda: next(stamps), get_body_joints_pose=lambda: next(poses),
    ))
    sample = manager.PicoReader()._read_sample()
    assert sample["timestamp_ns"] == 101
    np.testing.assert_array_equal(sample["body_poses_np"], 1)


def prepare_streamer(streamer, monkeypatch):
    monkeypatch.setattr(manager, "get_controller_inputs", lambda reader: (False, 0, 0, 0, 0))
    monkeypatch.setattr(manager, "compute_hand_joints_from_inputs", lambda *args: (np.zeros(7), np.zeros(7)))
    streamer.left_hand_ik_solver = streamer.right_hand_ik_solver = None
    streamer.three_point = SimpleNamespace(process_smpl_pose=lambda pose: pose.copy())


def test_zero_body_clock_sdk_can_teleoperate_and_still_detect_packet_loss(streamer, monkeypatch):
    prepare_streamer(streamer, monkeypatch)
    origin = vr_pose()
    target = origin.copy()
    now = [0.0]
    packet_stamp = [1_000_000_000]
    monkeypatch.setattr(manager.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(manager, "xrt", SimpleNamespace(
        get_time_stamp_ns=lambda: packet_stamp[0], get_body_timestamp_ns=lambda: 0,
        get_body_joints_pose=lambda: target.copy(),
    ))
    streamer.reader = manager.PicoReader()
    streamer.vr_conditioner = VRMotionConditioner()
    streamer.vr_conditioner.seed(origin, 0)
    for i in range(1, 101):
        now[0] = i * 0.02
        packet_stamp[0] += 20_000_000
        target[0, 0] += 0.0004
        streamer.reader._latest = streamer.reader._read_sample()
        assert streamer.reader._latest["body_timestamp_ns"] == 0
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
        assert not streamer.vr_conditioner.fault
    assert streamer.last_vr_pose[0, 0] > origin[0, 0] + 0.01
    for i in range(101, 153):
        now[0] = i * 0.02
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert streamer.vr_conditioner.fault == "stale 3PT input"


def test_verified_body_clock_cannot_fall_back_to_packet_traffic(monkeypatch):
    body_stamp = [10]
    packet_stamp = [100]
    monkeypatch.setattr(manager, "xrt", SimpleNamespace(
        get_time_stamp_ns=lambda: packet_stamp[0], get_body_timestamp_ns=lambda: body_stamp[0],
        get_body_joints_pose=vr_pose,
    ))
    reader = manager.PicoReader()
    reader._latest = reader._read_sample()
    assert reader._latest["source_timestamp_ns"] == 10
    packet_stamp[0] = 200
    assert reader._read_sample()["source_timestamp_ns"] == 10
    body_stamp[0] = 0
    assert reader._read_sample()["source_timestamp_ns"] == 0


def test_planner_uses_one_snapshot_and_body_freshness(streamer, monkeypatch):
    prepare_streamer(streamer, monkeypatch)
    pose = vr_pose()
    calls = []

    def latest():
        calls.append(1)
        return {"body_poses_np": pose.copy(), "timestamp_ns": len(calls), "body_timestamp_ns": 100}

    streamer.reader.get_latest = latest
    streamer.reader.get_timestamp_ns = lambda: pytest.fail("Separate timestamp read")
    assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    np.testing.assert_allclose(decode_vr(streamer.packets[-1]), pose, atol=1e-7)
    pose[0, 0] += 0.01
    # Packet updates alone must not masquerade as new body poses.
    assert not streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert len(calls) == 2
    assert len(streamer.packets) == 1


@pytest.mark.parametrize("body_stamp", [0, 100])
def test_packet_traffic_cannot_hide_stale_body(streamer, monkeypatch, body_stamp):
    prepare_streamer(streamer, monkeypatch)
    origin = vr_pose()
    streamer.vr_conditioner = VRMotionConditioner()
    streamer.vr_conditioner.seed(origin, 0)
    now = [0.0]
    monkeypatch.setattr(manager.time, "monotonic", lambda: now[0])
    streamer.reader.get_latest = lambda: {
        "body_poses_np": origin, "timestamp_ns": int(now[0] * 1e9), "body_timestamp_ns": body_stamp,
    }
    for i in range(1, 54):
        now[0] = i * 0.02
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert streamer.vr_conditioner.fault == "stale 3PT input"


def test_failed_send_does_not_consume_body_timestamp(streamer, monkeypatch):
    prepare_streamer(streamer, monkeypatch)
    streamer.reader.get_latest = lambda: {
        "body_poses_np": vr_pose(), "timestamp_ns": 100, "body_timestamp_ns": 10,
    }
    streamer.last_vr_source_timestamp_ns = 9

    def fail(packet):
        raise RuntimeError("send failed")

    streamer.socket.send = fail
    with pytest.raises(RuntimeError, match="send failed"):
        streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert streamer.last_vr_source_timestamp_ns == 9
    streamer.socket.send = streamer.packets.append
    assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert streamer.last_vr_source_timestamp_ns == 10


def test_velocity_uses_device_intervals_and_survives_repeated_samples():
    limiter = VRMotionConditioner()
    origin = vr_pose()
    limiter.seed(origin, 0)
    stamp = 1_800_000_000_000_000_000  # Preserve integer ns before subtraction.
    target = origin.copy()
    limiter.update(target, 0.02, source_timestamp_ns=stamp)
    for i, arrival in enumerate((0.05, 0.10, 0.13), start=1):
        target = origin.copy()
        target[:, 0] += i * 0.002
        target[:, 3:] = (Rotation.from_rotvec(np.tile([0, 0, i * 0.004], (3, 1))) *
                        Rotation.from_quat(origin[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
        limiter.update(target, arrival, source_timestamp_ns=stamp + i * 40_000_000)
    np.testing.assert_allclose(limiter.raw_velocity[0, :, 0], 0.05)
    np.testing.assert_allclose(limiter.raw_velocity[1, :, 2], 0.1, atol=1e-12)
    estimate = limiter.target_velocity.copy()
    for now in (0.15, 0.17, 0.19, 0.21):
        limiter.update(target, now, source_timestamp_ns=stamp + 120_000_000)
        np.testing.assert_array_equal(limiter.target_velocity, estimate)
    limiter.update(target, 0.24, source_timestamp_ns=stamp + 120_000_000)
    np.testing.assert_array_equal(limiter.target_velocity, 0)  # Stale braking.
    assert not limiter.fault
    limiter.update(target, 0.26, source_timestamp_ns=stamp + 160_000_000)
    np.testing.assert_array_equal(limiter.raw_velocity, 0)  # A fresh stationary sample.


def test_invalid_velocity_is_not_revived_by_repeated_samples():
    limiter = VRMotionConditioner()
    origin = vr_pose()
    limiter.seed(origin, 0)
    limiter.update(origin, 0.02, source_timestamp_ns=1_000_000_000)
    target = origin.copy()
    target[:, 0] += 0.05
    limiter.update(target, 0.04, source_timestamp_ns=1_020_000_000)
    assert np.any(limiter.raw_velocity)
    for i in range(3, 7):
        limiter.update(target, i * 0.02, source_timestamp_ns=1_020_000_000)
        np.testing.assert_array_equal(limiter.target_velocity, 0)


def test_one_plausible_interval_does_not_enable_retained_velocity():
    limiter = VRMotionConditioner()
    origin = vr_pose()
    limiter.seed(origin, 0)
    limiter.update(origin, 0.02, source_timestamp_ns=1_000_000_000)
    for i in range(1, 4):
        target = origin.copy()
        target[:, 0] += i * 0.0004
        limiter.update(target, (i + 1) * 0.02, source_timestamp_ns=1_000_000_000 + i * 20_000_000)
        # The first velocity change exceeds the acceleration threshold. One
        # subsequent plausible interval cannot yet establish steady movement.
        if i < 3:
            np.testing.assert_array_equal(limiter.target_velocity, 0)
        else:
            np.testing.assert_allclose(limiter.target_velocity[0, :, 0], 0.02)


def test_steady_reach_with_repeated_samples_does_not_accumulate_following_gap():
    limiter = VRMotionConditioner()
    target = vr_pose()
    limiter.seed(target, 0)
    origin_x = target[:, 0].copy()
    errors = []
    stamp = 1_000_000_000
    for i in range(1, 501):
        now = i * 0.02
        if i % 3 == 1:
            target[:, 0] = origin_x + (now - 0.02) * 0.08
            stamp += 60_000_000
        output = limiter.update(target, now, source_timestamp_ns=stamp)
        if i > 100:
            errors.append(abs(target[0, 0] - output[0, 0]))
        assert not limiter.fault
    assert np.mean(errors) < 0.003


def test_clock_reset_brakes_until_reseed_and_clears_velocity_estimate():
    limiter = VRMotionConditioner()
    origin = vr_pose()
    limiter.seed(origin, 0)
    limiter.update(origin, 0.02, source_timestamp_ns=100)
    limiter.update(origin, 0.04, source_timestamp_ns=90)
    assert limiter.fault == "3PT source clock moved backwards"
    limiter.seed(origin, 0.05)
    limiter.update(origin, 0.06, source_timestamp_ns=90)
    assert not limiter.fault
    np.testing.assert_array_equal(limiter.target_velocity, 0)


def test_uneven_source_and_arrival_cadence_preserve_output_limits():
    limiter = VRMotionConditioner()
    origin = vr_pose()
    limiter.seed(origin, 0)
    now, source_time = 0.0, 0.0
    previous = origin.copy()
    velocity = np.zeros((2, 3, 3))
    acceleration = velocity.copy()
    for i in range(1, 401):
        dt = (0.012, 0.024, 0.016, 0.028)[i % 4]
        now += dt
        if i % 3 == 1:
            source_time += 0.06
            target = origin.copy()
            target[:, 0] += 0.06 * np.sin(source_time * 3)
            target[:, 3:] = (Rotation.from_rotvec(np.tile([0, 0, 0.4 * np.sin(source_time * 3)], (3, 1))) *
                            Rotation.from_quat(origin[:, 3:], scalar_first=True)).as_quat(scalar_first=True)
        output = limiter.update(target, now, source_timestamp_ns=1_000_000_000 + round(source_time * 1e9))
        v = np.array([(output[:, :3] - previous[:, :3]) / dt, rotation_error(output[:, 3:], previous[:, 3:]) / dt])
        a = (v - velocity) / dt
        j = (a - acceleration) / dt
        for values, bounds in (
            (v, [limiter.limits.speed, limiter.limits.angular_speed]),
            (a, [limiter.limits.acceleration, limiter.limits.angular_acceleration]),
            (j, [limiter.limits.jerk, limiter.limits.angular_jerk]),
        ):
            reserve = 1 if values is j else limiter.limits.hard_limit_factor
            assert np.all(np.linalg.norm(values, axis=-1) <= np.array(bounds)[:, None] * reserve + 1e-8)
        previous, velocity, acceleration = output, v, a
        assert not limiter.fault
