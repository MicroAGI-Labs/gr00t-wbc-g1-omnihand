"""Exercise mode handoffs without a headset, robot, or command sockets."""

from types import SimpleNamespace

import numpy as np
import pytest

from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.utils.teleop.pose_transition import (
    IDLE_BASE_UPPER_BODY_MASK,
    IDLE_BASE_UPPER_BODY_RAD,
    JointPoseTransition,
)


@pytest.fixture
def streamer(monkeypatch):
    instance = manager.PlannerStreamer.__new__(manager.PlannerStreamer)
    reference = np.linspace(-0.4, 0.4, 17)
    instance.feedback_reader = SimpleNamespace(
        upper_body_planner_target=reference.copy(),
        upper_body_position_target=reference + 0.15,
        full_body_q_measured=np.linspace(-0.5, 0.5, 29),
        upper_body_joint_indices=[12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
        last_body_feedback_monotonic=manager.time.monotonic(),
        poll_feedback=lambda **kwargs: True,
        left_hand_position_target=None,
        right_hand_position_target=None,
    )
    instance.last_upper_body_override = None
    instance.reader = SimpleNamespace(get_timestamp_ns=lambda: 123)
    instance.last_xrt_timestamp = None
    instance.dt = 0.02
    instance.mode = manager.LocomotionMode.IDLE
    instance.yaw_accumulator = SimpleNamespace(update=lambda *args: [1.0, 0.0])
    instance.socket = SimpleNamespace(send=lambda data: None)
    monkeypatch.setattr(manager, "get_controller_axes", lambda reader: (0, 0, 0, 0))
    monkeypatch.setattr(manager, "build_planner_message", lambda *args, **kwargs: b"planner")
    return instance


def test_planner_handoff_ignores_measured_tracking_error(streamer):
    reference = streamer.feedback_reader.upper_body_planner_target.copy()
    start = streamer.commanded_transition_start()
    transition = JointPoseTransition(
        start, IDLE_BASE_UPPER_BODY_RAD, started_at=0.0, duration_s=2.0
    )
    np.testing.assert_array_equal(transition.sample(0).position, reference)
    # A 0.15-rad measured/reference offset must not become a command step.
    assert np.max(np.abs(transition.sample(0.02).position - reference)) < 1e-5
    np.testing.assert_allclose(transition.sample(2).position, IDLE_BASE_UPPER_BODY_RAD)


@pytest.mark.parametrize("source", ["base_pose", "ik_override"])
def test_handoff_preserves_last_sent_arm_override_and_planner_waist(streamer, source):
    position = IDLE_BASE_UPPER_BODY_RAD.copy()
    if source == "base_pose":
        assert streamer.run_once(manager.StreamMode.PLANNER_IDLE_BASE_POSE)
    else:
        position += 0.2
        assert streamer.run_once(
            manager.StreamMode.PLANNER,
            upper_body_override=(position, np.zeros(17), IDLE_BASE_UPPER_BODY_MASK),
        )
    expected = position.copy()
    # Retaining the command must not alias the IK solver's mutable buffers.
    position[:] = 99
    start = streamer.commanded_transition_start()
    np.testing.assert_array_equal(start[3:], expected[3:])
    np.testing.assert_array_equal(start[:3], streamer.feedback_reader.upper_body_planner_target[:3])


def test_plain_planner_packet_releases_cached_override_only_after_send(streamer):
    streamer.run_once(manager.StreamMode.PLANNER_IDLE_BASE_POSE)
    assert streamer.last_upper_body_override is not None
    # A repeated headset sample does not send anything or change the cache.
    assert not streamer.run_once(manager.StreamMode.PLANNER)
    assert streamer.last_upper_body_override is not None
    streamer.reader.get_timestamp_ns = lambda: 124
    assert streamer.run_once(manager.StreamMode.PLANNER)
    assert streamer.last_upper_body_override is None
    np.testing.assert_array_equal(
        streamer.commanded_transition_start(), streamer.feedback_reader.upper_body_planner_target
    )


def test_failed_send_keeps_previous_command_for_handoff(streamer):
    streamer.run_once(manager.StreamMode.PLANNER_IDLE_BASE_POSE)

    def fail(data):
        raise RuntimeError("socket failed")

    streamer.socket.send = fail
    with pytest.raises(RuntimeError, match="socket failed"):
        streamer.run_once(
            manager.StreamMode.PLANNER,
            upper_body_override=(np.ones(17), np.zeros(17), IDLE_BASE_UPPER_BODY_MASK),
        )
    np.testing.assert_array_equal(
        streamer.commanded_transition_start()[3:], IDLE_BASE_UPPER_BODY_RAD[3:]
    )


@pytest.mark.parametrize("fault", ["stale", "missing_reference", "malformed_reference"])
def test_handoff_requires_fresh_reference_without_measured_fallback(streamer, fault):
    feedback = streamer.feedback_reader
    if fault == "stale":
        feedback.last_body_feedback_monotonic -= 1
    elif fault == "missing_reference":
        feedback.upper_body_planner_target = None
    else:
        feedback.upper_body_planner_target = np.zeros(16)
    assert streamer.commanded_transition_start() is None


def test_vr_calibration_still_uses_measured_joints(streamer):
    captured = []
    streamer.three_point = SimpleNamespace(reset_with_measured_q=captured.append)
    assert streamer.recalibrate_for_vr3pt()
    expected = streamer.feedback_reader.full_body_q_measured.copy()
    expected[streamer.feedback_reader.upper_body_joint_indices] = (
        streamer.feedback_reader.upper_body_planner_target
    )
    np.testing.assert_array_equal(captured[0], expected)
