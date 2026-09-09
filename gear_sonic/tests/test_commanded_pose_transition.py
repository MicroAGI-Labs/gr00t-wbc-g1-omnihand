"""Exercise mode handoffs without a headset, robot, or command sockets."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from gear_sonic.scripts import pico_manager_thread_server as manager
from gear_sonic.scripts.run_data_exporter import unpack_pose_message
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
        upper_body_last_action=reference + 0.25,
        upper_body_position_target=reference + 0.15,
        full_body_q_measured=np.linspace(-0.5, 0.5, 29),
        upper_body_joint_indices=[12, 13, 14, 15, 22, 16, 23, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
        last_body_feedback_monotonic=manager.time.monotonic(),
        poll_feedback=lambda **kwargs: True,
        left_hand_position_target=None,
        right_hand_position_target=None,
        vr_pose=None,
    )
    instance.last_upper_body_override = None
    instance.last_vr_pose = None
    instance.held_vr_pose = None
    instance.first_vr_pose = None
    instance.disconnect_idle_pose = vr_pose()
    # These geometry/FSM tests isolate calibration from motion conditioning;
    # dedicated conditioner integration tests exercise the production filter.
    instance.vr_conditioner = None
    instance.reader = SimpleNamespace(get_timestamp_ns=lambda: 123, get_latest=lambda: None)
    instance.last_xrt_timestamp = None
    instance.dt = 0.02
    instance.mode = manager.LocomotionMode.IDLE
    instance.yaw_accumulator = SimpleNamespace(update=lambda *args: [1.0, 0.0, 0.0])
    instance.packets = []
    instance.socket = SimpleNamespace(send=instance.packets.append)
    monkeypatch.setattr(manager, "get_controller_axes", lambda reader: (0, 0, 0, 0))
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


def vr_pose():
    return np.column_stack((
        [[0.3, 0.2, 0.1], [0.3, -0.2, 0.1], [0.0, 0.0, 0.4]],
        Rotation.from_euler("xyz", [[0.1, 0.2, 0.3], [-0.1, 0.2, -0.3], [0.1, 0.1, 0.2]])
        .as_quat(scalar_first=True),
    ))


def decode_vr(packet):
    data = unpack_pose_message(packet, topic="planner")
    assert not {"upper_body_position", "upper_body_velocity", "upper_body_mask"} & data.keys()
    return np.column_stack((data["vr_position"].reshape(3, 3), data["vr_orientation"].reshape(3, 4)))


def forbidden_pico_read(*args):
    raise AssertionError("return/base hold must ignore Pico poses, triggers, and sticks")


@pytest.mark.parametrize("to_base", [True, False])
def test_vr_return_and_hold_ignore_pico_and_keep_one_controller(streamer, monkeypatch, to_base):
    start = vr_pose()
    streamer.last_vr_pose = start.copy()
    # Deliberately unrelated measured joints, policy output and underlying motion target.
    streamer.feedback_reader.upper_body_last_action[:] = 8
    goal = vr_pose()
    goal[:2, :3] += [0.2, 0.1, -0.1]
    goal[2, :3] += 1  # Head must remain at the last sent target during an arm return.
    monkeypatch.setattr(streamer, "vr_pose_from_upper_body", lambda joints: goal.copy())
    streamer.reader.get_latest = forbidden_pico_read
    monkeypatch.setattr(manager, "get_controller_axes", forbidden_pico_read)
    monkeypatch.setattr(manager, "get_controller_inputs", forbidden_pico_read)
    transition = streamer.begin_vr_return(to_base=to_base, duration_s=2)
    assert transition is not None
    began = transition.progress.started_at
    for elapsed in np.linspace(0, 2, 101):
        sent, complete = streamer.send_vr_return_sample(transition, began + elapsed)
        assert sent
        assert complete == (elapsed == 2)
        packet = unpack_pose_message(streamer.packets[-1], topic="planner")
        np.testing.assert_array_equal(packet["movement"], [0, 0, 0])
        np.testing.assert_array_equal(packet["facing"], [1, 0, 0])
        np.testing.assert_array_equal(packet["mode"], [manager.LocomotionMode.IDLE.value])
        pose = decode_vr(streamer.packets[-1])
        np.testing.assert_allclose(pose[2], start[2], atol=1e-7)
    np.testing.assert_allclose(decode_vr(streamer.packets[0]), start, atol=1e-7)
    np.testing.assert_allclose(decode_vr(streamer.packets[-1])[:2], goal[:2], atol=1e-7)
    endpoint = decode_vr(streamer.packets[-1])
    for _ in range(5):
        mode = manager.StreamMode.PLANNER_IDLE_BASE_POSE if to_base else manager.StreamMode.PLANNER
        assert streamer.run_once(mode, force_locomotion_idle=True)
        np.testing.assert_array_equal(decode_vr(streamer.packets[-1]), endpoint)


@pytest.mark.parametrize("fault", ["stale", "missing_vr"])
def test_vr_return_rejects_missing_reference(streamer, fault):
    streamer.feedback_reader.vr_pose = vr_pose()
    if fault == "stale":
        streamer.feedback_reader.last_body_feedback_monotonic -= 1
    else:
        streamer.feedback_reader.vr_pose = None
        streamer.feedback_reader.upper_body_planner_target = None
    assert streamer.begin_vr_return(to_base=True, duration_s=2) is None
    assert streamer.held_vr_pose is None
    assert not streamer.packets


def test_initial_vr_return_uses_planner_command_and_real_base_fk(streamer):
    streamer.three_point = manager.ThreePointPose()
    streamer.feedback_reader.last_body_feedback_monotonic = manager.time.monotonic()
    streamer.feedback_reader.vr_pose = vr_pose()
    transition = streamer.begin_vr_return(to_base=True, duration_s=2)
    expected_start = streamer.vr_pose_from_upper_body(
        streamer.feedback_reader.upper_body_planner_target
    )
    np.testing.assert_allclose(transition.start, expected_start)
    # Base wrist FK is symmetric about the robot's sagittal plane.
    np.testing.assert_allclose(transition.goal[0, :3], transition.goal[1, :3] * [1, -1, 1], atol=1e-5)
    assert np.isfinite(transition.goal).all()
    assert streamer.send_vr_return_sample(transition, transition.progress.started_at)[0]
    packet = unpack_pose_message(streamer.packets[-1], "planner")
    # Automatic recovery returns all the way to planner idle, not calibration.
    np.testing.assert_allclose(packet["vr_base_pose"].reshape(2, 7), expected_start[:2], atol=1e-7)
    assert not np.allclose(expected_start[:2], transition.goal[:2])

    # The reconnect completion must use that same cached idle destination,
    # even if fresh planner feedback has since changed during locomotion.
    streamer.last_vr_pose = transition.goal.copy()
    streamer.feedback_reader.upper_body_planner_target += 0.3
    recovery = streamer.begin_vr_return(
        to_base=True, duration_s=2, goal_override=streamer.disconnect_idle_pose
    )
    np.testing.assert_allclose(recovery.goal[:2], expected_start[:2])
    np.testing.assert_array_equal(recovery.goal[2], streamer.last_vr_pose[2])

    # A deliberate second B+Y refreshes the destination from the planner.
    streamer.feedback_reader.last_body_feedback_monotonic = manager.time.monotonic()
    idle_return = streamer.begin_vr_return(to_base=False, duration_s=2)
    np.testing.assert_allclose(streamer.disconnect_idle_pose[:2], idle_return.goal[:2])
    assert not np.allclose(streamer.disconnect_idle_pose[:2], expected_start[:2])


def test_every_reentry_reanchors_pico_and_preserves_first_packet(streamer, monkeypatch):
    # Exercise real calibration math with three-point extraction isolated from SMPL conversion.
    monkeypatch.setattr(manager, "_process_3pt_pose", lambda sample: sample.copy())
    streamer.three_point = manager.ThreePointPose(robot_model=object())
    streamer.left_hand_ik_solver = streamer.right_hand_ik_solver = None
    monkeypatch.setattr(manager, "get_controller_inputs", lambda reader: (False, 0, 0, 0, 0))
    monkeypatch.setattr(manager, "compute_hand_joints_from_inputs", lambda *args: (np.zeros(7), np.zeros(7)))
    for entry in range(2):
        target = vr_pose()
        target[:2, :3] += entry * 0.05
        streamer.last_vr_pose = target.copy()
        streamer.held_vr_pose = target.copy()
        sample = vr_pose()
        sample[:, :3] += (entry + 1) * 0.4
        sample[:, 3:] = Rotation.from_euler("xyz", [0.3, -0.5, 0.7 + entry]).as_quat(scalar_first=True)
        streamer.reader.get_latest = lambda: {"body_poses_np": sample}
        assert streamer.recalibrate_for_vr3pt()
        assert streamer.held_vr_pose is None
        calibrated = streamer.three_point.process_smpl_pose(sample)
        np.testing.assert_allclose(calibrated, target, atol=1e-12)
        # A newer frame arriving after calibration cannot replace the first held command.
        sample[0, :3] += [0.1, 0, 0]
        streamer.reader.get_timestamp_ns = lambda: entry * 10
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
        np.testing.assert_allclose(decode_vr(streamer.packets[-1]), target, atol=1e-7)
        streamer.reader.get_timestamp_ns = lambda: entry * 10 + 1
        assert streamer.run_once(manager.StreamMode.PLANNER_VR_3PT)
        expected = target.copy()
        expected[0, :3] += Rotation.from_quat(sample[2, 3:], scalar_first=True).inv().apply([0.1, 0, 0])
        np.testing.assert_allclose(decode_vr(streamer.packets[-1]), expected, atol=1e-7)


def test_failed_vr_return_send_retains_previous_hold(streamer, monkeypatch):
    streamer.last_vr_pose = vr_pose()
    monkeypatch.setattr(streamer, "vr_pose_from_upper_body", lambda joints: vr_pose())
    transition = streamer.begin_vr_return(to_base=True, duration_s=2)
    hold = streamer.held_vr_pose.copy()
    monkeypatch.setattr(streamer, "run_once", lambda *args, **kwargs: False)
    assert streamer.send_vr_return_sample(transition, transition.progress.started_at + 3) == (False, False)
    np.testing.assert_array_equal(streamer.held_vr_pose, hold)


def test_calibration_maps_actual_smpl_sample_to_held_target():
    three_point = manager.ThreePointPose(robot_model=object())
    sample = np.zeros((24, 7))
    sample[:, 6] = 1
    sample[22, :3] = [-0.4, 1, 0.2]
    sample[23, :3] = [0.4, 1, 0.2]
    sample[12, :3] = [0, 1.4, 0]
    target = vr_pose()
    original = sample.copy()
    three_point.calibrate_to_vr_target(sample, target)
    np.testing.assert_allclose(three_point.process_smpl_pose(sample), target, atol=1e-7)
    np.testing.assert_array_equal(sample, original)


@pytest.mark.parametrize("disconnect_frame,outage_seconds", [(None, 0), (123, 5), (123, 14.5), (123, 20), (150, 5), (150, 20), (232, 20)])
def test_manager_gesture_flow_keeps_pico_out_of_return_and_base(streamer, monkeypatch, disconnect_frame, outage_seconds):
    # Run the real FSM, gesture trackers, calibration and packet sender offline.
    # Replace only time, hardware, pose extraction and unused full-body/hand workers.
    clock = SimpleNamespace(now=100.0)
    monkeypatch.setattr(manager, "time", SimpleNamespace(
        monotonic=lambda: clock.now,
        monotonic_ns=lambda: int(clock.now * 1e9),
        time=lambda: clock.now,
        sleep=lambda seconds: setattr(clock, "now", clock.now + seconds),
    ))
    streamer.feedback_reader.last_body_feedback_monotonic = clock.now
    streamer.feedback_reader.vr_pose = vr_pose()
    streamer.feedback_reader.poll_feedback = lambda **kwargs: (
        setattr(streamer.feedback_reader, "last_body_feedback_monotonic", clock.now) or True
    )
    three_point = manager.ThreePointPose(robot_model=object())
    monkeypatch.setattr(manager, "ThreePointPose", lambda **kwargs: three_point)
    monkeypatch.setattr(manager, "_process_3pt_pose", lambda sample: sample.copy())
    streamer.three_point = three_point

    def fk(joints):
        pose = vr_pose()
        if np.array_equal(joints, IDLE_BASE_UPPER_BODY_RAD):
            pose[:2, 2] += 0.12  # Calibration and resting idle are distinct.
        return pose

    monkeypatch.setattr(streamer, "vr_pose_from_upper_body", fk)
    streamer.left_hand_ik_solver = streamer.right_hand_ik_solver = None
    streamer.reset_yaw = lambda: None
    streamer.reader.disconnected = False
    streamer.reader.stop = lambda: None
    streamer.reader.get_timestamp_ns = lambda: int(clock.now * 1e9)
    monkeypatch.setattr(manager, "PicoReader", type(streamer.reader))
    reconnects = []
    recovery_targets = []
    reconnect_packet_counts = []

    def reconnect():
        reconnects.append(frame[0])
        reconnect_packet_counts.append(len(emitted))
        # Native reconnect may block far longer than the robot's 1s watchdog.
        clock.now += outage_seconds
        target = streamer.last_vr_pose.copy()
        if outage_seconds > 17:
            target[:2] = vr_pose()[:2]
        elif outage_seconds > 14:
            # Reconnect during SONIC's return (its watchdog starts before
            # Python's 2s disconnect detector). Finish from current feedback.
            target[:2, :3] = (target[:2, :3] + vr_pose()[:2, :3]) / 2
        streamer.feedback_reader.vr_pose = target.copy()
        recovery_targets.append(target)
        streamer.reader.disconnected = False

    streamer.reader.reconnect = reconnect
    source = vr_pose()
    reads = []

    def latest():
        reads.append(frame[0])
        return {"body_poses_np": source.copy()}

    streamer.reader.get_latest = latest
    monkeypatch.setattr(manager, "_init_input_source", lambda *args: streamer.reader)
    monkeypatch.setattr(manager, "PlannerStreamer", lambda **kwargs: streamer)
    monkeypatch.setattr(manager, "PoseStreamer", lambda **kwargs: SimpleNamespace())
    monkeypatch.setattr(manager, "HandIntentStream", lambda: SimpleNamespace(publish=lambda *args, **kwargs: None))
    monkeypatch.setattr(manager, "get_controller_inputs", lambda reader: (False, 0, 0, 0, 0))
    monkeypatch.setattr(manager, "get_axis_clicks", lambda reader: (False, False))
    monkeypatch.setattr(manager, "compute_hand_joints_from_inputs", lambda *args: (np.zeros(7), np.zeros(7)))
    neutral, ax, by = (False,) * 4, (True, False, True, False), (False, True, False, True)
    # First AX pair reaches base; second enters VR. BY returns, then another AX pair reenters.
    schedule = [(True,) * 4, neutral, ax, neutral, ax, neutral] + [neutral] * 110
    schedule += [ax, neutral, ax, neutral] + [neutral] * 5
    return_frame = len(schedule) + 1
    schedule += [neutral if disconnect_frame == 123 else by, neutral] + [neutral] * 110
    reentry_frame = len(schedule) + 3
    schedule += [ax, neutral, ax, neutral] + [neutral] * 5
    frame = [-1]

    def buttons(reader):
        frame[0] += 1
        if frame[0] == len(schedule):
            raise KeyboardInterrupt
        source[0, 0] = 0.3 + frame[0] * 0.01  # Pico never stops moving.
        if frame[0] == disconnect_frame:
            streamer.reader.disconnected = True
            streamer.mode = manager.LocomotionMode.WALK
            # Reconnected/stale sticks cannot start walking before re-arming.
            monkeypatch.setattr(manager, "get_controller_axes", lambda reader: (1, 0, 1, 0))
        return schedule[frame[0]]

    monkeypatch.setattr(manager, "get_abxy_buttons", buttons)
    emitted = []
    socket = SimpleNamespace(
        bind=lambda *args: None, connect=lambda *args: None,
        setsockopt=lambda *args: None, setsockopt_string=lambda *args: None,
        poll=lambda *args: False, close=lambda: None,
        send=lambda data: emitted.append((frame[0], data)),
    )
    streamer.socket = socket
    monkeypatch.setattr(
        manager.zmq, "Context", lambda: SimpleNamespace(socket=lambda *args: socket, term=lambda: None)
    )
    manager.run_pico_manager(
        teleop_mode="vr3pt", input_source="isaac", target_fps=50,
        idle_base_transition_duration=2.0,  # Fixed timeline for this gesture scenario.
    )
    if disconnect_frame is not None:
        assert reconnects == [disconnect_frame]
        # Only the robot owns the outage timer. Python must not keep sending
        # stale planner packets or overwrite the robot's return on reconnect.
        before = emitted[:reconnect_packet_counts[0]]
        assert len([data for index, data in before if index == disconnect_frame and data.startswith(b"planner")]) == 1
        packets = [(index, data) for index, data in emitted[reconnect_packet_counts[0]:]
                   if data.startswith(b"planner") and index < reentry_frame]
        expected = recovery_targets[0]
        assert packets
        np.testing.assert_allclose(decode_vr(packets[0][1]), expected, atol=1e-7)
        if outage_seconds > 14:
            expected = expected.copy()
            expected[:2] = vr_pose()[:2]
        for index, data in packets:
            if index > disconnect_frame:
                np.testing.assert_allclose(decode_vr(data), expected, atol=1e-7)
            payload = unpack_pose_message(data, "planner")
            np.testing.assert_array_equal(payload["movement"], [0, 0, 0])
            assert payload["mode"][0] == manager.LocomotionMode.IDLE.value
        # Re-entry recalibrates from the held target, without a pose jump.
        resumed = next(data for index, data in emitted
                       if index == reentry_frame and data.startswith(b"planner"))
        np.testing.assert_array_equal(decode_vr(resumed), decode_vr(packets[-1][1]))
        return
    states = [int(unpack_pose_message(data, "manager_state")["stream_mode"][0])
              for _, data in emitted if data.startswith(b"manager_state")]
    assert states[return_frame - 1] == manager.StreamMode.PLANNER_VR_3PT.value
    assert states[reentry_frame - 1] == manager.StreamMode.PLANNER_IDLE_BASE_POSE.value
    assert states[-1] == manager.StreamMode.PLANNER_VR_3PT.value
    assert not any(return_frame <= index < reentry_frame for index in reads)
    vr_packets = [(index, decode_vr(data)) for index, data in emitted
                  if data.startswith(b"planner") and index >= 5]
    for calibration_frame in (119, reentry_frame):
        before = next(pose for index, pose in vr_packets if index == calibration_frame - 1)
        after = next(pose for index, pose in vr_packets if index == calibration_frame)
        np.testing.assert_array_equal(after, before)
    held = [pose for index, pose in vr_packets if return_frame + 102 <= index < reentry_frame]
    assert len(held) > 1
    for pose in held[1:]:
        np.testing.assert_array_equal(pose, held[0])


def test_generated_hold_does_not_call_disconnected_timestamp_api(streamer):
    streamer.reader.get_timestamp_ns = forbidden_pico_read
    streamer.reader.get_latest = forbidden_pico_read
    streamer.held_vr_pose = vr_pose()
    for _ in range(100):
        assert streamer.run_once(manager.StreamMode.PLANNER_IDLE_BASE_POSE, force_locomotion_idle=True)
        np.testing.assert_allclose(decode_vr(streamer.packets[-1]), vr_pose(), atol=1e-7)


def test_pico_timestamp_is_from_last_complete_sample():
    reader = manager.PicoReader()
    assert reader.get_timestamp_ns() == 0
    reader._latest = {"timestamp_ns": 123}
    reader._disconnected.set()
    assert reader.get_timestamp_ns() == 123
