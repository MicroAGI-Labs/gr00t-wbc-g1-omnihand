"""Tracking cadence regressions, using fake SDK data and offline commands only."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.tests.test_commanded_pose_transition import streamer, vr_pose, decode_vr


@pytest.mark.parametrize("tracking_delay", [0.3, 10.0])
def test_controller_connect_waits_for_async_tracking_without_resubscribing(monkeypatch, tracking_delay):
    clock = SimpleNamespace(now=0.)
    subscriptions = []
    def sleep(seconds):
        clock.now += seconds
        assert clock.now < tracking_delay + 1.0
    monkeypatch.setattr(manager, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    monkeypatch.setattr(manager, "_ensure_robotics_service", lambda: None)
    monkeypatch.setattr(manager, "_close_xrt", lambda: pytest.fail("Must retain a live subscription"))
    monkeypatch.setattr(manager, "xrt", SimpleNamespace(
        init=lambda: subscriptions.append(clock.now),
        get_time_stamp_ns=lambda: int(clock.now * 1e9),
        get_left_controller_pose=lambda: [0., 0., 0., 0., 0., 0., 1.],
        get_right_controller_pose=lambda: [0., 0., 0., 0., 0., 0., 1.],
        get_headset_pose=lambda: ([0.] * 7 if clock.now < tracking_delay
                                  else [0., 0., 0., 0., 0., 0., 1.]),
    ))
    manager._connect_xrt_body_stream(controller_tracking=True)
    assert subscriptions == [0.]
    assert clock.now > tracking_delay  # Require an advancing, valid pose stream.


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
    monkeypatch.setattr(manager, "get_controller_inputs", lambda reader: (False, 0, 0, 1, 1))
    monkeypatch.setattr(manager, "compute_hand_joints_from_inputs", lambda *args: (np.zeros(7), np.zeros(7)))
    streamer.left_hand_ik_solver = streamer.right_hand_ik_solver = None
    streamer.three_point = SimpleNamespace(process_smpl_pose=lambda pose: pose.copy())
    streamer.last_vr_pose = vr_pose()
    streamer.vr_arm_clutch.update(vr_pose(), vr_pose(), (1, 1), source_fresh=True)


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
    for i in range(1, 101):
        now[0] = i * 0.02
        packet_stamp[0] += 20_000_000
        target[0, 0] += 0.0004
        streamer.reader._latest = streamer.reader._read_sample()
        assert streamer.reader._latest["body_timestamp_ns"] == 0
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert streamer.last_vr_pose[0, 0] > origin[0, 0] + 0.01
    sent = len(streamer.packets)
    for i in range(101, 153):
        now[0] = i * 0.02
        assert not streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
    assert len(streamer.packets) == sent  # No old body pose refreshes the receiver watchdog.


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
    now = [0.0]
    monkeypatch.setattr(manager.time, "monotonic", lambda: now[0])
    streamer.reader.get_latest = lambda: {
        "body_poses_np": origin, "timestamp_ns": int(now[0] * 1e9), "body_timestamp_ns": body_stamp,
    }
    for i in range(1, 54):
        now[0] = i * 0.02
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT) == (i == 1 and body_stamp > 0)
    assert len(streamer.packets) == int(body_stamp > 0)


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
